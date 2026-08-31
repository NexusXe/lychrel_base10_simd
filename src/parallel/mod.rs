#[cfg(any(test, feature = "reference-impl"))]
use crate::impossible;
use crate::integer_limb::Integer;
#[cfg(any(test, feature = "reference-impl"))]
use crate::integer_limb::{LV_LEN, Limb, LimbVec, add_block, increment_block};
use std::alloc::{Allocator, Global};
#[cfg(any(test, feature = "reference-impl"))]
use std::hint::{cold_path, likely, unlikely};
use std::hint::spin_loop;
#[cfg(any(test, feature = "reference-impl"))]
use std::sync::Arc;
#[cfg(any(test, feature = "reference-impl"))]
use std::sync::atomic::{AtomicBool, AtomicPtr};
use std::sync::atomic::{
    AtomicUsize,
    Ordering::{AcqRel, Acquire, Relaxed},
};
#[cfg(any(test, feature = "reference-impl"))]
use std::sync::mpsc::Sender;
use std::time::Instant;

pub struct IterationResult<T: Allocator + Clone + Copy> {
    pub(crate) last_iteration: usize,
    pub(crate) start_time: Instant,
    pub(crate) end_integer: Integer<T>,
}

pub struct StatusReport {
    pub(crate) iteration: usize,
    pub(crate) current_value: Option<Integer<Global>>,
}

pub const LOG_FREQUENCY_EXP: usize = 14;

pub const LOG_MASK: usize = 2usize.pow(LOG_FREQUENCY_EXP as u32);

/// Limb count below which one iteration is too short to amortize spreading
/// across cores, so `iterate_parallel` runs the engine with one participant.
pub const PAR_THRESHOLD_LIMBS: usize = 4096;

/// Limb count below which more than one CCD's worth of threads loses to its
/// own barrier latency, so `iterate_parallel` caps the engine at 8 threads
/// until the integer outgrows it. Measured crossover on the 9950X3D is
/// between 32k and 64k limbs.
pub const PAR_FULL_THREADS_LIMBS: usize = 49152;

/// Threads for one CCD; the cap applied between the two size thresholds.
pub(crate) const ONE_CCD_THREADS: usize = 8;

/// Centralized sense-reversing barrier. All waiters spin; the engine never
/// parks a thread, since iterations are microseconds apart.
pub(crate) struct SpinBarrier {
    participants: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
}

impl SpinBarrier {
    pub(crate) fn new(participants: usize) -> Self {
        Self {
            participants,
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub(crate) fn wait(&self) {
        let generation = self.generation.load(Acquire);
        if self.count.fetch_add(1, AcqRel) == self.participants - 1 {
            self.count.store(0, Relaxed);
            self.generation.fetch_add(1, AcqRel);
        } else {
            while self.generation.load(Acquire) == generation {
                spin_loop();
            }
        }
    }
}

#[repr(align(64))]
pub(crate) struct Padded<T>(pub(crate) T);

/// Per-iteration state published by the coordinator before the start barrier
/// and read by every worker after it. All accesses are Relaxed: the barriers
/// provide the acquire/release edges.
#[cfg(any(test, feature = "reference-impl"))]
struct Shared {
    barrier: SpinBarrier,
    num_threads: usize,
    limbs_ptr: AtomicPtr<LimbVec>,
    total_limbs: AtomicUsize,
    skip_len: AtomicUsize,
    stop: AtomicBool,
    ever_carried: AtomicBool,
    /// 2 * num_threads + 1 entries: the add pass's block boundaries.
    block_bounds: Box<[AtomicUsize]>,
    /// num_threads + 1 entries: the zipper pass's pair-index boundaries.
    zip_bounds: Box<[AtomicUsize]>,
    /// 2 * num_threads entries: each block's speculative carry-out.
    block_carry: Box<[Padded<AtomicBool>]>,
}

// The raw pointer is to the limb buffer, whose accesses are partitioned by
// block and ordered by the barrier.
#[cfg(any(test, feature = "reference-impl"))]
unsafe impl Send for Shared {}
#[cfg(any(test, feature = "reference-impl"))]
unsafe impl Sync for Shared {}

#[cfg(any(test, feature = "reference-impl"))]
impl Shared {
    fn new(num_threads: usize) -> Self {
        Self {
            barrier: SpinBarrier::new(num_threads),
            num_threads,
            limbs_ptr: AtomicPtr::new(std::ptr::null_mut()),
            total_limbs: AtomicUsize::new(0),
            skip_len: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            ever_carried: AtomicBool::new(false),
            block_bounds: (0..=num_threads * 2).map(|_| AtomicUsize::new(0)).collect(),
            zip_bounds: (0..=num_threads).map(|_| AtomicUsize::new(0)).collect(),
            block_carry: (0..num_threads * 2)
                .map(|_| Padded(AtomicBool::new(false)))
                .collect(),
        }
    }

    /// The data-parallel phases of one iteration, for thread `t`. Thread `t`
    /// owns add blocks `t` and `2P - 1 - t`: the mirrored pair keeps the
    /// limbs a thread zips and the limbs it then adds in the same caches.
    /// Entered right after the start barrier; returns after the add barrier.
    #[inline]
    fn run_phases(&self, t: usize) {
        let limbs_ptr = self.limbs_ptr.load(Relaxed);
        let total_limbs = self.total_limbs.load(Relaxed);
        let skip_len = self.skip_len.load(Relaxed);
        let num_blocks = self.num_threads * 2;

        let zip_lb = self.zip_bounds[t].load(Relaxed);
        let zip_ub = self.zip_bounds[t + 1].load(Relaxed);
        if likely(zip_lb < zip_ub) {
            unsafe {
                Limb::zipper(limbs_ptr, limbs_ptr.add(total_limbs - 1), zip_lb, zip_ub);
            }
        }

        self.barrier.wait();

        let lo_start = self.block_bounds[t].load(Relaxed);
        let lo_end = self.block_bounds[t + 1].load(Relaxed);
        let hi_block = num_blocks - 1 - t;
        let hi_start = self.block_bounds[hi_block].load(Relaxed);
        let hi_end = self.block_bounds[hi_block + 1].load(Relaxed);

        // The limb just past a block feeds the block's last unaligned reload,
        // and its owner overwrites it during the add pass; between these two
        // barriers every thread only reads, so the snapshots are race-free.
        // The topmost block's boundary limb is the padding limb.
        let boundary_lo = unsafe { *limbs_ptr.add(lo_end) };
        let boundary_hi = unsafe { *limbs_ptr.add(hi_end) };

        self.barrier.wait();

        let (carry_lo, carried_lo) = if likely(lo_start < lo_end) {
            unsafe { add_block(limbs_ptr, lo_start, lo_end, skip_len, boundary_lo) }
        } else {
            (false, false)
        };
        let (carry_hi, carried_hi) = if likely(hi_start < hi_end) {
            unsafe { add_block(limbs_ptr, hi_start, hi_end, skip_len, boundary_hi) }
        } else {
            (false, false)
        };

        self.block_carry[t].0.store(carry_lo, Relaxed);
        self.block_carry[hi_block].0.store(carry_hi, Relaxed);
        if likely(carried_lo || carried_hi) {
            self.ever_carried.store(true, Relaxed);
        }

        self.barrier.wait();
    }
}

/// The CPUs this process may run on, in id order. On this machine's Zen 5
/// topology that order puts one CPU per physical core first (SMT siblings are
/// the upper ids) with the V-cache CCD's cores lowest, so pinning participant
/// t to the t-th allowed CPU spreads threads across distinct cores and fills
/// the large-L3 CCD before the frequency CCD.
///
/// The set is captured once, on the first call, which happens before any
/// participant is pinned. A later engine (the 8-to-16 thread upgrade) is
/// created from a coordinator already pinned to one CPU, and reading the
/// affinity mask again there would collapse the whole pool onto that CPU.
#[cfg(target_family = "unix")]
pub(crate) fn allowed_cpus() -> &'static [usize] {
    static ALLOWED: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    ALLOWED.get_or_init(|| unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &raw mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
            .collect()
    })
}

#[cfg(target_family = "unix")]
fn pin_to_cpu(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        // a failed pin only costs placement, so the result is ignored
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &raw const set);
    }
}

#[cfg(target_family = "unix")]
pub(crate) fn pin_participant(t: usize, cpus: &[usize]) {
    if !cpus.is_empty() {
        pin_to_cpu(cpus[t % cpus.len()]);
    }
}

#[cfg(not(target_family = "unix"))]
pub(crate) fn pin_participant(_t: usize, _cpus: &[usize]) {}

#[cfg(not(target_family = "unix"))]
pub(crate) fn allowed_cpus() -> &'static [usize] {
    &[]
}

/// The number of distinct physical cores among `cpus`, from sysfs topology.
/// Falls back to the plain CPU count where topology is unreadable.
#[cfg(target_os = "linux")]
fn physical_cores_among(cpus: &[usize]) -> usize {
    let mut cores = std::collections::HashSet::new();
    for &cpu in cpus {
        let read = |leaf: &str| {
            std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/{leaf}"))
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
        };
        match (read("physical_package_id"), read("core_id")) {
            (Some(package), Some(core)) => {
                cores.insert((package, core));
            }
            _ => return cpus.len(),
        }
    }
    cores.len().max(1)
}

/// Resolves `--threads 0`: one thread per physical core available to the
/// process. SMT siblings share the vector pipes and the L1/L2 this kernel
/// saturates; measured on the 9950X3D, 32 threads lose to 16 by 20-40%
/// cache-resident and by 13% at DRAM-bound sizes.
#[must_use]
pub fn auto_threads() -> usize {
    #[cfg(target_os = "linux")]
    {
        let cpus = allowed_cpus();
        if !cpus.is_empty() {
            return physical_cores_among(cpus);
        }
    }

    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

/// A persistent pool of worker threads executing the fused reverse-and-add in
/// lockstep with the calling thread. The caller is participant 0; `step` is
/// a drop-in equivalent of `Integer::fused_reverse_add_asm_interleave`.
#[cfg(any(test, feature = "reference-impl"))]
pub struct ParallelEngine {
    shared: Arc<Shared>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

#[cfg(any(test, feature = "reference-impl"))]
impl ParallelEngine {
    #[must_use]
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads >= 1);
        let shared = Arc::new(Shared::new(num_threads));
        let cpus = allowed_cpus();

        pin_participant(0, cpus);

        let handles = (1..num_threads)
            .map(|t| {
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || {
                    pin_participant(t, cpus);
                    loop {
                        shared.barrier.wait();
                        if unlikely(shared.stop.load(Relaxed)) {
                            break;
                        }
                        shared.run_phases(t);
                    }
                })
            })
            .collect();

        Self { shared, handles }
    }

    /// One fused reverse-and-add step. Returns whether any digit carried,
    /// with the same meaning as `fused_reverse_add_asm_interleave`.
    pub fn step<T: Allocator + Clone + Copy>(&self, integer: &mut Integer<T>) -> bool {
        let shared = &*self.shared;
        let limbs = &mut integer.0;

        let total_limbs = limbs.len();
        if total_limbs == 0 {
            impossible!("Tried to reverse and add empty integer");
        }

        let skip_len = LV_LEN - usize::from(limbs[total_limbs - 1].len());

        limbs.push(Limb::new()); // padding

        let limbs_ptr = limbs.as_mut_ptr().cast::<LimbVec>();
        shared.limbs_ptr.store(limbs_ptr, Relaxed);
        shared.total_limbs.store(total_limbs, Relaxed);
        shared.skip_len.store(skip_len, Relaxed);
        shared.ever_carried.store(false, Relaxed);

        let num_blocks = shared.num_threads * 2;
        for k in 0..=num_blocks {
            shared.block_bounds[k].store(k * total_limbs / num_blocks, Relaxed);
        }
        let zip_pairs = total_limbs.div_ceil(2);
        for t in 0..=shared.num_threads {
            shared.zip_bounds[t].store(t * zip_pairs / shared.num_threads, Relaxed);
        }

        shared.barrier.wait();
        shared.run_phases(0);

        // Serial carry resolution across blocks: a block whose true carry-in
        // turned out to be one gets a decimal increment at its base, which
        // propagates past the block only if the whole block is nines.
        let mut carry = false;
        for k in 0..num_blocks {
            let mut carry_out = shared.block_carry[k].0.load(Relaxed);
            if unlikely(carry) {
                cold_path();
                let start = shared.block_bounds[k].load(Relaxed);
                let end = shared.block_bounds[k + 1].load(Relaxed);
                carry_out |= unsafe { increment_block(limbs_ptr, start, end) };
            }
            carry = carry_out;
        }

        if likely(carry) {
            unsafe {
                (*limbs_ptr.add(total_limbs).cast::<Limb>())
                    .0
                    .as_mut_array()[0] = 1;
            }
        } else {
            limbs.pop();
        }

        shared.ever_carried.load(Relaxed)
    }
}

#[cfg(any(test, feature = "reference-impl"))]
impl Drop for ParallelEngine {
    fn drop(&mut self) {
        self.shared.stop.store(true, Relaxed);
        self.shared.barrier.wait();
        for handle in self.handles.drain(..) {
            handle.join().expect("parallel engine worker died");
        }
    }
}

/// The iteration loop. The engine runs with one participant while the
/// integer is small, and widens at the two size thresholds above.
#[cfg(any(test, feature = "reference-impl"))]
#[inline]
pub fn iterate_parallel<T: Allocator + Clone + Copy>(
    range: std::ops::Range<usize>,
    starting_integer: Integer<T>,
    tx: Option<&Sender<StatusReport>>,
    num_threads: usize,
) -> IterationResult<T> {
    let mut current_iteration = starting_integer;

    current_iteration.0.reserve(2048.min(range.end / 100));

    #[allow(unused_variables)]
    let mut carried: bool = true; // ignore palindrome check on the first loop
    let mut i: usize = range.start;
    let mut engine: Option<ParallelEngine> = None;
    let mut engine_threads: usize = 0;

    let start_time = Instant::now();

    #[allow(unused_assignments)]
    while likely(i < range.end) {
        #[cfg(not(feature = "no-verify"))]
        if unlikely(!carried) {
            cold_path();
            let mut reverse = Integer(Vec::with_capacity(current_iteration.0.len()));
            current_iteration.reverse_into_integer(&mut reverse);
            if current_iteration.0 == reverse.0 {
                cold_path();
                break;
            }
        }

        let num_limbs = current_iteration.0.len();
        let target_threads = if likely(num_limbs >= PAR_FULL_THREADS_LIMBS) {
            num_threads
        } else if num_limbs >= PAR_THRESHOLD_LIMBS {
            num_threads.min(ONE_CCD_THREADS)
        } else {
            1
        };

        if unlikely(engine_threads != target_threads) {
            cold_path();
            engine = None; // join the smaller pool before its cores are re-pinned
            engine = Some(ParallelEngine::new(target_threads));
            engine_threads = target_threads;
            eprintln!(
                "Parallel engine: {target_threads} thread(s) at iteration {i} ({num_limbs} limbs, {:.3} s elapsed)",
                start_time.elapsed().as_secs_f64()
            );
        }

        carried = unsafe { engine.as_ref().unwrap_unchecked() }.step(&mut current_iteration);

        if unlikely(i.is_multiple_of(LOG_MASK)) {
            let report = StatusReport {
                iteration: i,
                current_value: {
                    if unlikely(i.is_multiple_of(2usize.pow(18))) {
                        cold_path();
                        // manually clone the current iteration into a new vector using the global allocator
                        let mut output_vec =
                            Vec::with_capacity_in(current_iteration.0.len(), Global);
                        output_vec.extend_from_slice(&current_iteration.0);
                        Some(Integer(output_vec))
                    } else {
                        None
                    }
                },
            };

            if likely(tx.is_some()) {
                if unlikely(
                    unsafe { tx.as_ref().unwrap_unchecked() }
                        .send(report)
                        .is_err(),
                ) {
                    break;
                }
            } else {
                cold_path();
            }
        }
        i += 1;
    }
    IterationResult {
        last_iteration: i,
        start_time,
        end_integer: current_iteration,
    }
}

#[cfg(test)]
mod tests;
