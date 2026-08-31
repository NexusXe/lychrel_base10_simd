use crate::impossible;
use crate::integer_limb::{Integer, LV_LEN, Limb, LimbVec, add_block, increment_block};
use crate::iterate::{IterationResult, LOG_MASK, StatusReport};
use std::alloc::{Allocator, Global};
use std::hint::{cold_path, likely, spin_loop, unlikely};
use std::sync::Arc;
use std::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicUsize,
    Ordering::{AcqRel, Acquire, Relaxed},
};
use std::sync::mpsc::Sender;
use std::time::Instant;

/// Limb count below which one iteration is too short to amortize the engine's
/// barriers, so `iterate_parallel` stays on the single-threaded kernel.
pub const PAR_THRESHOLD_LIMBS: usize = 4096;

/// Limb count below which more than one CCD's worth of threads loses to its
/// own barrier latency, so `iterate_parallel` caps the engine at 8 threads
/// until the integer outgrows it. Measured crossover on the 9950X3D is
/// between 32k and 64k limbs.
pub const PAR_FULL_THREADS_LIMBS: usize = 49152;

/// Threads for one CCD; the cap applied between the two size thresholds.
const ONE_CCD_THREADS: usize = 8;

/// Centralized sense-reversing barrier. All waiters spin; the engine never
/// parks a thread, since iterations are microseconds apart.
struct SpinBarrier {
    participants: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
}

impl SpinBarrier {
    fn new(participants: usize) -> Self {
        Self {
            participants,
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn wait(&self) {
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
struct Padded<T>(T);

/// Per-iteration state published by the coordinator before the start barrier
/// and read by every worker after it. All accesses are Relaxed: the barriers
/// provide the acquire/release edges.
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
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

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
#[cfg(target_family = "unix")]
fn allowed_cpus() -> Vec<usize> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &raw mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
            .collect()
    }
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
fn pin_participant(t: usize, cpus: &[usize]) {
    if !cpus.is_empty() {
        pin_to_cpu(cpus[t % cpus.len()]);
    }
}

#[cfg(not(target_family = "unix"))]
fn pin_participant(_t: usize, _cpus: &[usize]) {}

#[cfg(not(target_family = "unix"))]
fn allowed_cpus() -> Vec<usize> {
    Vec::new()
}

/// A persistent pool of worker threads executing the fused reverse-and-add in
/// lockstep with the calling thread. The caller is participant 0; `step` is
/// a drop-in equivalent of `Integer::fused_reverse_add_asm_interleave`.
pub struct ParallelEngine {
    shared: Arc<Shared>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl ParallelEngine {
    #[must_use]
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads >= 1);
        let shared = Arc::new(Shared::new(num_threads));
        let cpus = allowed_cpus();

        pin_participant(0, &cpus);

        let handles = (1..num_threads)
            .map(|t| {
                let shared = Arc::clone(&shared);
                let cpus = cpus.clone();
                std::thread::spawn(move || {
                    pin_participant(t, &cpus);
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

impl Drop for ParallelEngine {
    fn drop(&mut self) {
        self.shared.stop.store(true, Relaxed);
        self.shared.barrier.wait();
        for handle in self.handles.drain(..) {
            handle.join().expect("parallel engine worker died");
        }
    }
}

/// Drop-in replacement for `iterate::iterate` that switches from the serial
/// kernel to a `ParallelEngine` once the integer is large enough for the
/// per-iteration barriers to amortize.
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
    let mut engine_threads: usize = 1;

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

        if unlikely(target_threads > 1 && engine_threads != target_threads) {
            cold_path();
            engine = None; // join the smaller pool before its cores are re-pinned
            engine = Some(ParallelEngine::new(target_threads));
            engine_threads = target_threads;
        }

        carried = if likely(target_threads > 1) {
            unsafe { engine.as_ref().unwrap_unchecked() }.step(&mut current_iteration)
        } else {
            current_iteration.fused_reverse_add_asm_interleave()
        };

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
