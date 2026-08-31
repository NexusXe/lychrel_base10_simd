//! The dual nibble-packed representation: the number and its digit-reversal
//! are both kept packed (two digits per byte, the checkpoint format), which
//! halves the DRAM traffic of an iteration. The reverse-and-add pass reads
//! the two copies slot-aligned with no skip offset, resolves carries in one
//! ascending sweep, and produces both next-iteration copies.
//!
//! Packed line layout (one 64-byte `Limb` holding 128 digits): digit `p` of
//! the line lives in byte `p` low nibble for `p < 64`, byte `p - 64` high
//! nibble for `p >= 64`. Lines are LSD-first like the byte-per-digit form,
//! and a packed line is exactly `Limb::pack` of two adjacent unpacked limbs.

use crate::impossible;
use crate::integer_limb::{Integer, LV_LEN, Limb, LimbVec, resolve_digits};
use std::alloc::Allocator;
use std::hint::{cold_path, likely};

/// Digits per packed line.
pub(crate) const DPL: usize = 2 * LV_LEN;

const LO_MASK: LimbVec = LimbVec::splat(0x0F);

/// The two 64-digit halves of a packed line, low digits first.
#[inline(always)]
fn unpack_line(v: LimbVec) -> (LimbVec, LimbVec) {
    (v & LO_MASK, v >> 4)
}

/// Packs two vectors of clean digits (0..=9) into one line.
#[inline(always)]
fn pack_line(lo: LimbVec, hi: LimbVec) -> LimbVec {
    lo | (hi << 4)
}

/// The result of adding and resolving one packed line: the packed output,
/// its two unpacked digit halves (low digits first), the carry out of the
/// top digit, and whether any digit carried.
struct LineSum {
    packed: LimbVec,
    lo: LimbVec,
    hi: LimbVec,
    carry_out: bool,
    carried: bool,
}

/// Adds two packed lines digit-wise and resolves every decimal carry inside
/// the line, with `carry` into the line's lowest digit.
#[inline(always)]
fn add_resolve_line(a: LimbVec, r: LimbVec, carry: bool) -> LineSum {
    let (a_lo, a_hi) = unpack_line(a);
    let (r_lo, r_hi) = unpack_line(r);

    let (lo, carry_mid, carried_lo) = resolve_digits(a_lo + r_lo, carry);
    let (hi, carry_out, carried_hi) = resolve_digits(a_hi + r_hi, carry_mid);

    LineSum {
        packed: pack_line(lo, hi),
        lo,
        hi,
        carry_out,
        carried: carried_lo || carried_hi,
    }
}

/// The digit at slot `d` of a packed line slice.
#[inline]
pub(crate) fn digit_at(lines: &[Limb], d: usize) -> u8 {
    let line = lines[d / DPL].0;
    let p = d % DPL;
    if p < LV_LEN {
        line[p] & 0x0F
    } else {
        line[p - LV_LEN] >> 4
    }
}

/// Overwrites the digit at slot `d` of a packed line slice.
#[inline]
pub(crate) fn set_digit(lines: &mut [Limb], d: usize, digit: u8) {
    debug_assert!(digit <= 9);
    let line = &mut lines[d / DPL].0;
    let p = d % DPL;
    if p < LV_LEN {
        line[p] = (line[p] & 0xF0) | digit;
    } else {
        line[p - LV_LEN] = (line[p - LV_LEN] & 0x0F) | (digit << 4);
    }
}

/// The 128 digits of packed line `m` as ascending bytes; lines outside
/// `0..lines.len()` read as zeros (the virtual padding below digit 0 and
/// above the top line).
#[inline(always)]
fn line_digit_bytes(lines: &[Limb], m: isize) -> [u8; DPL] {
    let mut out = [0u8; DPL];
    if m >= 0 && (m as usize) < lines.len() {
        let (lo, hi) = unpack_line(lines[m as usize].0);
        out[..LV_LEN].copy_from_slice(lo.as_array());
        out[LV_LEN..].copy_from_slice(hi.as_array());
    }
    out
}

/// Rebuilds `dst` as the digit-reversal of `lines` (`digits` significant
/// digits): dst slot `s` holds digit `digits - 1 - s`. Slots at and above
/// `digits` in the top line are zero.
pub(crate) fn mirror_into<T: Allocator + Clone + Copy>(
    lines: &[Limb],
    digits: usize,
    dst: &mut Vec<Limb, T>,
) {
    let lines_out = digits.div_ceil(DPL);
    dst.clear();
    dst.reserve(lines_out);

    // dst line l covers slots [DPL*l, DPL*(l+1)) = source digits
    // [digits - DPL*(l+1), digits - DPL*l) in descending order, so the source
    // window walks the array downward at the fixed intra-line offset
    // digits % DPL. The window's two source lines roll: the lower line of
    // step l is the upper line of step l + 1.
    let mut upper_digits = line_digit_bytes(lines, (digits as isize - 1).div_euclid(DPL as isize));
    for l in 0..lines_out {
        let w = digits as isize - (DPL * (l + 1)) as isize;
        let m1 = w.div_euclid(DPL as isize);
        let phi = w.rem_euclid(DPL as isize) as usize;

        let lower_digits = line_digit_bytes(lines, m1);
        let mut buf = [0u8; 2 * DPL];
        buf[..DPL].copy_from_slice(&lower_digits);
        buf[DPL..].copy_from_slice(&upper_digits);

        // window = source digits [w, w + DPL) ascending; the dst line wants
        // them descending, so its low half is the reversed upper window half.
        let window = &buf[phi..phi + DPL];
        let lo = LimbVec::from_slice(&window[LV_LEN..]).reverse();
        let hi = LimbVec::from_slice(&window[..LV_LEN]).reverse();
        dst.push(Limb(pack_line(lo, hi)));

        upper_digits = lower_digits;
    }
}

/// A number held as two packed copies: `a` is the value LSD-first, and
/// `rev[cur]` is its digit-reversal, slot-aligned so that
/// `rev[d] == a[digits - 1 - d]`. The other rev buffer is the write target
/// of the next iteration.
pub struct PackedInt<T: Allocator + Clone + Copy> {
    pub(crate) a: Vec<Limb, T>,
    rev: [Vec<Limb, T>; 2],
    cur: usize,
    pub(crate) digits: usize,
}

impl<T: Allocator + Clone + Copy> PackedInt<T> {
    /// Builds the dual representation from a byte-per-digit integer.
    pub fn from_integer(integer: &Integer<T>, allocator: T) -> Self {
        if integer.0.is_empty() {
            impossible!("Tried to pack an empty integer");
        }
        let digits = integer.len() as usize;
        let mut a = Vec::with_capacity_in(integer.0.len().div_ceil(2), allocator);
        for pair in integer.0.chunks(2) {
            let hi = if pair.len() == 2 { pair[1].0 } else { LimbVec::splat(0) };
            a.push(Limb(pack_line(pair[0].0, hi)));
        }
        let mut rev0 = Vec::new_in(allocator);
        mirror_into(&a, digits, &mut rev0);
        let rev1 = rev0.clone();
        Self {
            a,
            rev: [rev0, rev1],
            cur: 0,
            digits,
        }
    }

    /// The value as a byte-per-digit integer (for reports and checkpoints).
    pub fn to_integer<G: Allocator + Clone + Copy>(&self, allocator: G) -> Integer<G> {
        let mut out = Vec::with_capacity_in(self.a.len() * 2, allocator);
        let limbs = self.digits.div_ceil(LV_LEN);
        for line in &self.a {
            let (lo, hi) = unpack_line(line.0);
            out.push(Limb(lo));
            out.push(Limb(hi));
        }
        out.truncate(limbs);
        Integer(out)
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn rev_cur(&self) -> &[Limb] {
        &self.rev[self.cur]
    }

    /// Whether the value is a palindrome: the two copies are slot-identical.
    #[inline]
    pub fn is_palindrome(&self) -> bool {
        let lines = self.digits.div_ceil(DPL);
        self.a[..lines] == self.rev[self.cur][..lines]
    }

    /// Whether the next reverse-and-add gains a digit, decided exactly before
    /// the pass: descending from the top, skip digit sums equal to nine (they
    /// propagate whatever comes from below); the first other sum decides.
    /// All-nines sums generate no carry at all and do not grow.
    pub(crate) fn prescan_grow(&self) -> bool {
        let l = self.digits;
        for d in (0..l).rev() {
            let s = digit_at(&self.a, d) + digit_at(&self.a, l - 1 - d);
            if s != 9 {
                return s > 9;
            }
        }
        false
    }

    /// One reverse-and-add step, single-threaded: an ascending fused add over
    /// the two copies with an exact running carry, then the mirror rebuild
    /// into the spare rev buffer. Returns whether any digit carried.
    #[cfg(test)]
    pub fn step(&mut self) -> bool {
        let l = self.digits;
        let grew = self.prescan_grow();
        let lp = l + grew as usize;
        let lines = l.div_ceil(DPL);

        let mut carry = false;
        let mut any_carried = false;
        for k in 0..lines {
            let sum = add_resolve_line(self.a[k].0, self.rev[self.cur][k].0, carry);
            self.a[k] = Limb(sum.packed);
            carry = sum.carry_out;
            any_carried |= sum.carried;
        }
        if carry {
            // only reachable when the top line was full to the brim
            let mut top = LimbVec::splat(0);
            top[0] = 1;
            self.a.push(Limb(top));
        }

        debug_assert_eq!(lp.div_ceil(DPL), self.a.len().min(lp.div_ceil(DPL)));
        debug_assert!(digit_at(&self.a, lp - 1) != 0, "prescan missed growth");
        debug_assert!(
            lp % DPL == 0 || self.a.len() * DPL <= lp + DPL,
            "prescan over-grew"
        );
        if lp < self.a.len() * DPL {
            debug_assert_eq!(
                (lp..(lines.max(lp.div_ceil(DPL)) * DPL).min(self.a.len() * DPL))
                    .map(|d| digit_at(&self.a, d))
                    .max()
                    .unwrap_or(0),
                0,
                "dirty padding above the top digit"
            );
        }

        self.digits = lp;
        let (a, rev) = (&self.a, &mut self.rev);
        let next = 1 - self.cur;
        mirror_into(a, lp, &mut rev[next]);
        self.cur = next;

        likely(any_carried)
    }
}

use crate::parallel::{Padded, SpinBarrier, allowed_cpus, pin_participant};
use std::cell::UnsafeCell;
use std::hint::unlikely;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize};

/// Per-iteration state published by the coordinator before the start barrier
/// and read by every worker after it. All accesses are Relaxed: the barriers
/// provide the acquire/release edges.
struct SharedPacked {
    barrier: SpinBarrier,
    num_threads: usize,
    a_ptr: AtomicPtr<Limb>,
    rev_src: AtomicPtr<Limb>,
    rev_dst: AtomicPtr<Limb>,
    /// Input lines: ceil(digits / DPL).
    lines: AtomicUsize,
    /// The output digit count, prescanned exactly before the pass; every
    /// rev_dst write slot derives from it.
    out_digits: AtomicUsize,
    stop: AtomicBool,
    ever_carried: AtomicBool,
    /// 2 * num_threads + 1 entries: block boundaries in lines.
    bounds: Box<[AtomicUsize]>,
    /// 2 * num_threads entries: each block's speculative carry-out.
    block_carry: Box<[Padded<AtomicBool>]>,
    /// Reversed digit windows of each block's first and last chunk, for the
    /// coordinator to assemble the rev_dst lines that straddle block seams.
    stash_first: Box<[Padded<UnsafeCell<[u8; DPL]>>]>,
    stash_last: Box<[Padded<UnsafeCell<[u8; DPL]>>]>,
}

// The raw pointers partition by block, the stashes are written only by the
// block's owner, and the barrier orders every access.
unsafe impl Send for SharedPacked {}
unsafe impl Sync for SharedPacked {}

/// Writes one rev_dst line assembled from the reversed windows of chunk `k`
/// (`cur`) and chunk `k - 1` (`prev`). Chunk `k` covers rev slots
/// [out_digits - DPL*(k+1), out_digits - DPL*k); the line completed by its
/// arrival is the one holding the top of that window. Targets outside
/// [0, lines_out) are the virtual lines above the top or below slot zero.
#[inline(always)]
unsafe fn emit_rev_line(
    rev_dst: *mut Limb,
    out_digits: usize,
    lines_out: usize,
    k: usize,
    cur: &[u8; DPL],
    prev: &[u8; DPL],
) {
    let w = out_digits as isize - (DPL * (k + 1)) as isize;
    let target = w.div_euclid(DPL as isize) + 1;
    if target < 0 || target >= lines_out as isize {
        return;
    }
    let phi = out_digits % DPL;

    let mut buf = [0u8; 2 * DPL];
    buf[..DPL].copy_from_slice(cur);
    buf[DPL..].copy_from_slice(prev);
    let window = &buf[DPL - phi..2 * DPL - phi];
    let lo = LimbVec::from_slice(&window[..LV_LEN]);
    let hi = LimbVec::from_slice(&window[LV_LEN..]);
    unsafe {
        *rev_dst.add(target as usize) = Limb(pack_line(lo, hi));
    }
}

impl SharedPacked {
    fn new(num_threads: usize) -> Self {
        Self {
            barrier: SpinBarrier::new(num_threads),
            num_threads,
            a_ptr: AtomicPtr::new(std::ptr::null_mut()),
            rev_src: AtomicPtr::new(std::ptr::null_mut()),
            rev_dst: AtomicPtr::new(std::ptr::null_mut()),
            lines: AtomicUsize::new(0),
            out_digits: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            ever_carried: AtomicBool::new(false),
            bounds: (0..=num_threads * 2).map(|_| AtomicUsize::new(0)).collect(),
            block_carry: (0..num_threads * 2)
                .map(|_| Padded(AtomicBool::new(false)))
                .collect(),
            stash_first: (0..num_threads * 2)
                .map(|_| Padded(UnsafeCell::new([0; DPL])))
                .collect(),
            stash_last: (0..num_threads * 2)
                .map(|_| Padded(UnsafeCell::new([0; DPL])))
                .collect(),
        }
    }

    /// The fused pass over block `j`: add the two copies line by line with a
    /// speculative carry-in of zero, write the sums over `a` in place, and
    /// scatter the reversed digits into rev_dst. A chunk's reversed window
    /// straddles two rev_dst lines, so each chunk's arrival completes one
    /// line from itself and its predecessor; the windows at the block's
    /// edges are stashed for the coordinator's seam assembly.
    fn run_block(&self, j: usize) {
        let a = self.a_ptr.load(Relaxed);
        let rsrc = self.rev_src.load(Relaxed);
        let rdst = self.rev_dst.load(Relaxed);
        let lines = self.lines.load(Relaxed);
        let out_digits = self.out_digits.load(Relaxed);
        let lines_out = out_digits.div_ceil(DPL);

        let start = self.bounds[j].load(Relaxed);
        let end = self.bounds[j + 1].load(Relaxed);
        if start >= end {
            self.block_carry[j].0.store(false, Relaxed);
            return;
        }

        let mut carry = false;
        let mut any_carried = false;
        let mut prev = [0u8; DPL];
        let mut cur = [0u8; DPL];

        for k in start..end {
            #[cfg(all(target_arch = "x86_64", not(feature = "no-prefetch")))]
            unsafe {
                use std::arch::x86_64::{_MM_HINT_ET0, _MM_HINT_T0, _mm_prefetch};
                _mm_prefetch::<_MM_HINT_ET0>(a.add(k + 16).cast());
                _mm_prefetch::<_MM_HINT_T0>(rsrc.add(k + 16).cast());
            }

            let sum = unsafe {
                let out = add_resolve_line((*a.add(k)).0, (*rsrc.add(k)).0, carry);
                (*a.add(k)).0 = out.packed;
                out
            };
            carry = sum.carry_out;
            any_carried |= sum.carried;

            // the window is the chunk's digits in descending order
            cur[..LV_LEN].copy_from_slice(sum.hi.reverse().as_array());
            cur[LV_LEN..].copy_from_slice(sum.lo.reverse().as_array());

            if likely(k > start) {
                unsafe { emit_rev_line(rdst, out_digits, lines_out, k, &cur, &prev) };
            } else if k == 0 {
                // the top rev_dst line: nothing above the window but padding
                unsafe { emit_rev_line(rdst, out_digits, lines_out, k, &cur, &[0; DPL]) };
            } else {
                unsafe { *self.stash_first[j].0.get() = cur };
            }
            prev = cur;
        }

        if end == lines {
            // the bottom rev_dst line: nothing below the last window
            unsafe { emit_rev_line(rdst, out_digits, lines_out, end, &[0; DPL], &prev) };
        } else {
            unsafe { *self.stash_last[j].0.get() = prev };
        }

        self.block_carry[j].0.store(carry, Relaxed);
        if likely(any_carried) {
            self.ever_carried.store(true, Relaxed);
        }
    }

    /// Both blocks of participant `t`, mirror-owned like the byte engine:
    /// the rev_dst lines a block emits are the mirror of its own line range,
    /// which is the participant's other block.
    #[inline]
    fn run_blocks(&self, t: usize) {
        self.run_block(t);
        self.run_block(self.num_threads * 2 - 1 - t);
        self.barrier.wait();
    }
}

/// Adds one to digit `d0` and propagates the decimal carry upward through
/// `a`, stopping before `end_digit`. Returns the slot one past the last
/// changed digit, and whether the carry escaped the range, which requires
/// every digit in it to be nine.
fn increment_digits(a: &mut [Limb], d0: usize, end_digit: usize) -> (usize, bool) {
    let mut d = d0;
    while d < end_digit {
        let digit = digit_at(a, d);
        if digit == 9 {
            set_digit(a, d, 0);
            d += 1;
        } else {
            set_digit(a, d, digit + 1);
            return (d + 1, false);
        }
    }
    (d, true)
}

/// A persistent pool of worker threads executing the packed fused
/// reverse-and-add in lockstep with the calling thread, which is
/// participant 0. `step` is a drop-in equivalent of `PackedInt::step`.
pub struct PackedEngine {
    shared: Arc<SharedPacked>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl PackedEngine {
    #[must_use]
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads >= 1);
        let shared = Arc::new(SharedPacked::new(num_threads));
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
                        shared.run_blocks(t);
                    }
                })
            })
            .collect();

        Self { shared, handles }
    }

    /// One packed reverse-and-add step. Returns whether any digit carried.
    pub fn step<T: Allocator + Clone + Copy>(&self, x: &mut PackedInt<T>) -> bool {
        let shared = &*self.shared;
        let num_blocks = shared.num_threads * 2;

        let digits = x.digits;
        let grew = x.prescan_grow();
        let out_digits = digits + grew as usize;
        let lines = digits.div_ceil(DPL);
        let lines_out = out_digits.div_ceil(DPL);

        // Growth past the top line is decided by the prescan, so the line
        // can be appended before the pass; its single digit is set after
        // the carry scan confirms it.
        if lines_out > x.a.len() {
            x.a.push(Limb::new());
        }
        let next = 1 - x.cur;
        x.rev[next].resize(lines_out, Limb::new());

        shared.a_ptr.store(x.a.as_mut_ptr(), Relaxed);
        shared
            .rev_src
            .store(x.rev[x.cur].as_ptr().cast_mut(), Relaxed);
        shared.rev_dst.store(x.rev[next].as_mut_ptr(), Relaxed);
        shared.lines.store(lines, Relaxed);
        shared.out_digits.store(out_digits, Relaxed);
        shared.ever_carried.store(false, Relaxed);
        for k in 0..=num_blocks {
            shared.bounds[k].store(k * lines / num_blocks, Relaxed);
        }

        shared.barrier.wait();
        shared.run_blocks(0);

        // Seam lines: rev_dst lines straddling a block boundary are built
        // from the last window below the seam and the first window above it.
        let mut below: Option<(&[u8; DPL], usize)> = None;
        for j in 0..num_blocks {
            let start = shared.bounds[j].load(Relaxed);
            let end = shared.bounds[j + 1].load(Relaxed);
            if start >= end {
                continue;
            }
            if let Some((prev, k)) = below
                && start == k
                && start != 0
            {
                let cur = unsafe { &*shared.stash_first[j].0.get() };
                unsafe {
                    emit_rev_line(
                        x.rev[next].as_mut_ptr(),
                        out_digits,
                        lines_out,
                        start,
                        cur,
                        prev,
                    );
                }
            }
            below = Some((unsafe { &*shared.stash_last[j].0.get() }, end));
        }

        // Serial carry resolution across blocks: a block whose true carry-in
        // turned out to be one gets a decimal increment at its base. Any
        // digit the increment changes is mirrored into rev_dst.
        let mut carry = false;
        for j in 0..num_blocks {
            let start = shared.bounds[j].load(Relaxed);
            let end = shared.bounds[j + 1].load(Relaxed);
            if start >= end {
                continue; // an empty block passes the carry through
            }
            let mut carry_out = shared.block_carry[j].0.load(Relaxed);
            if unlikely(carry) {
                cold_path();
                let d0 = start * DPL;
                let (changed_until, escaped) =
                    increment_digits(&mut x.a, d0, (end * DPL).min(out_digits));
                for d in d0..changed_until {
                    set_digit(&mut x.rev[next], out_digits - 1 - d, digit_at(&x.a, d));
                }
                carry_out |= escaped;
            }
            carry = carry_out;
        }

        if unlikely(carry) {
            // only reachable when the input's top line was full to the brim
            cold_path();
            debug_assert!(grew && digits % DPL == 0);
            set_digit(&mut x.a, out_digits - 1, 1);
            set_digit(&mut x.rev[next], 0, 1);
        }

        x.digits = out_digits;
        x.cur = next;

        debug_assert!(digit_at(&x.a, out_digits - 1) != 0, "prescan missed growth");
        shared.ever_carried.load(Relaxed)
    }
}

impl Drop for PackedEngine {
    fn drop(&mut self) {
        self.shared.stop.store(true, Relaxed);
        self.shared.barrier.wait();
        for handle in self.handles.drain(..) {
            handle.join().expect("packed engine worker died");
        }
    }
}

/// The iteration loop over the dual packed representation. The engine runs
/// with one participant while the number is small, and widens at the same
/// working-set thresholds as the byte engine (expressed there in 64-digit
/// limbs).
#[inline]
pub fn iterate_packed<T: Allocator + Clone + Copy>(
    range: std::ops::Range<usize>,
    starting_integer: Integer<T>,
    tx: Option<&std::sync::mpsc::Sender<crate::parallel::StatusReport>>,
    num_threads: usize,
) -> crate::parallel::IterationResult<T> {
    use crate::parallel::{
        IterationResult, LOG_MASK, PAR_FULL_THREADS_LIMBS, PAR_THRESHOLD_LIMBS, StatusReport,
    };
    use std::alloc::Global;
    use std::time::Instant;

    let allocator = *starting_integer.0.allocator();
    let mut current = PackedInt::from_integer(&starting_integer, allocator);
    drop(starting_integer);

    #[allow(unused_variables)]
    let mut carried: bool = true; // ignore palindrome check on the first loop
    let mut i: usize = range.start;
    let mut engine: Option<PackedEngine> = None;
    let mut engine_threads: usize = 0;

    let start_time = Instant::now();

    #[allow(unused_assignments)]
    while likely(i < range.end) {
        #[cfg(not(feature = "no-verify"))]
        if unlikely(!carried) {
            cold_path();
            if current.is_palindrome() {
                cold_path();
                break;
            }
        }

        let num_limbs = current.digits.div_ceil(LV_LEN);
        let target_threads = if likely(num_limbs >= PAR_FULL_THREADS_LIMBS) {
            num_threads
        } else if num_limbs >= PAR_THRESHOLD_LIMBS {
            num_threads.min(crate::parallel::ONE_CCD_THREADS)
        } else {
            1
        };

        if unlikely(engine_threads != target_threads) {
            cold_path();
            engine = None; // join the smaller pool before its cores are re-pinned
            engine = Some(PackedEngine::new(target_threads));
            engine_threads = target_threads;
            eprintln!(
                "Packed engine: {target_threads} thread(s) at iteration {i} ({num_limbs} limbs, {:.3} s elapsed)",
                start_time.elapsed().as_secs_f64()
            );
        }

        carried = unsafe { engine.as_ref().unwrap_unchecked() }.step(&mut current);

        if unlikely(i.is_multiple_of(LOG_MASK)) {
            let report = StatusReport {
                iteration: i,
                current_value: {
                    if unlikely(i.is_multiple_of(2usize.pow(18))) {
                        cold_path();
                        Some(current.to_integer(Global))
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
        end_integer: current.to_integer(allocator),
    }
}

#[cfg(test)]
mod tests;
