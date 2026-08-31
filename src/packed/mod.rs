//! The single-copy nibble-packed representation: only the number itself is
//! kept packed (two digits per byte, the checkpoint format), ping-ponged
//! between two buffers. The reverse-and-add pass reads the current buffer
//! with two streams -- forward for a[d], backward from the top for
//! a[L-1-d], the reversed operand assembled in registers by a descending
//! funnel permutation -- and writes the sum slot-aligned into the other
//! buffer. One read pair plus one write per line is 510MB moved per
//! iteration on a 340MB number, against 680MB for the dual-copy scheme it
//! replaces.
//!
//! Packed line layout (one 64-byte `Limb` holding 128 digits): digit `p` of
//! the line lives in byte `p` low nibble for `p < 64`, byte `p - 64` high
//! nibble for `p >= 64`. Lines are LSD-first like the byte-per-digit form,
//! and a packed line is exactly `Limb::pack` of two adjacent unpacked limbs.

use crate::impossible;
use crate::integer_limb::{Integer, LV_LEN, Limb, LimbVec};
use std::alloc::Allocator;
use std::hint::{cold_path, likely};
use std::simd::prelude::*;

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
/// the carry out of the top digit, and whether any digit carried.
struct LineSum {
    packed: LimbVec,
    carry_out: bool,
    carried: bool,
}

/// Adds a packed line to an unpacked reversed operand (given as its two
/// 64-digit planes) and resolves every decimal carry inside the line, with
/// `carry` into the line's lowest digit. The carry chain runs over the
/// whole 128-digit line at once: the generate/propagate words of the two
/// halves concatenate into a u128 (see `resolve_digits` for the
/// derivation), so the line costs one wide add instead of two chained
/// 64-digit resolutions.
#[inline(always)]
fn add_resolve_line(a: LimbVec, r_lo: LimbVec, r_hi: LimbVec, carry: bool) -> LineSum {
    const NINES: LimbVec = LimbVec::splat(9);
    const TOP_BIT: u32 = (2 * LV_LEN - 1) as u32;

    let (a_lo, a_hi) = unpack_line(a);
    let sum_lo = a_lo + r_lo;
    let sum_hi = a_hi + r_hi;

    for half in [sum_lo, sum_hi] {
        for digit in half.as_array() {
            if *digit > 18 {
                impossible!("Got impossible addition result");
            }
        }
    }

    let generate = (u128::from(sum_hi.simd_gt(NINES).to_bitmask()) << LV_LEN)
        | u128::from(sum_lo.simd_gt(NINES).to_bitmask());
    let propagate = (u128::from(sum_hi.simd_eq(NINES).to_bitmask()) << LV_LEN)
        | u128::from(sum_lo.simd_eq(NINES).to_bitmask());
    let gp = generate | propagate;
    let (sum, adder_carry) = generate.carrying_add(gp, carry);

    let carry_in = sum ^ generate ^ gp;
    let carry_out = if 2 * LV_LEN == 128 {
        adder_carry
    } else {
        (generate | (propagate & carry_in)) & (1 << TOP_BIT) != 0
    };
    let emit = (carry_in >> 1) | (u128::from(carry_out) << TOP_BIT);

    #[inline(always)]
    fn fix(sums: LimbVec, receive: u64, emit: u64) -> LimbVec {
        type M = std::simd::Mask<i8, LV_LEN>;
        let out = M::from_bitmask(emit).select(sums - LimbVec::splat(10), sums);
        let out = M::from_bitmask(receive).select(out + LimbVec::splat(1), out);
        for digit in out.as_array() {
            if *digit > 9 {
                impossible!("Got impossible carry propagation result");
            }
        }
        out
    }

    let lo = fix(sum_lo, carry_in as u64, emit as u64);
    let hi = fix(sum_hi, (carry_in >> LV_LEN) as u64, (emit >> LV_LEN) as u64);

    LineSum {
        packed: pack_line(lo, hi),
        carry_out,
        carried: carry_in != 0 || carry_out,
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

/// A number held as one packed copy, ping-ponged between two buffers:
/// `a[cur]` is the value LSD-first, and `a[1 - cur]` is the write target of
/// the next iteration.
pub struct PackedInt<T: Allocator + Clone + Copy> {
    a: [Vec<Limb, T>; 2],
    cur: usize,
    pub(crate) digits: usize,
}

impl<T: Allocator + Clone + Copy> PackedInt<T> {
    /// Builds the packed representation from a byte-per-digit integer.
    pub fn from_integer(integer: &Integer<T>, allocator: T) -> Self {
        if integer.0.is_empty() {
            impossible!("Tried to pack an empty integer");
        }
        let digits = integer.len() as usize;
        let mut buf = Vec::with_capacity_in(integer.0.len().div_ceil(2), allocator);
        for pair in integer.0.chunks(2) {
            let hi = if pair.len() == 2 { pair[1].0 } else { LimbVec::splat(0) };
            buf.push(Limb(pack_line(pair[0].0, hi)));
        }
        Self {
            a: [buf, Vec::new_in(allocator)],
            cur: 0,
            digits,
        }
    }

    /// The value as a byte-per-digit integer (for reports and checkpoints).
    pub fn to_integer<G: Allocator + Clone + Copy>(&self, allocator: G) -> Integer<G> {
        let a = &self.a[self.cur];
        let mut out = Vec::with_capacity_in(a.len() * 2, allocator);
        let limbs = self.digits.div_ceil(LV_LEN);
        for line in a {
            let (lo, hi) = unpack_line(line.0);
            out.push(Limb(lo));
            out.push(Limb(hi));
        }
        out.truncate(limbs);
        Integer(out)
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn a_cur(&self) -> &[Limb] {
        &self.a[self.cur]
    }

    /// Whether the value is a palindrome: every digit equals its mirror.
    /// Called only on iterations where nothing carried, which is rare, so a
    /// scalar scan suffices.
    #[inline]
    pub fn is_palindrome(&self) -> bool {
        let a = &self.a[self.cur];
        (0..self.digits / 2).all(|d| digit_at(a, d) == digit_at(a, self.digits - 1 - d))
    }

    /// Whether the next reverse-and-add gains a digit, decided exactly before
    /// the pass: descending from the top, skip digit sums equal to nine (they
    /// propagate whatever comes from below); the first other sum decides.
    /// All-nines sums generate no carry at all and do not grow.
    pub(crate) fn prescan_grow(&self) -> bool {
        let a = &self.a[self.cur];
        let l = self.digits;
        for d in (0..l).rev() {
            let s = digit_at(a, d) + digit_at(a, l - 1 - d);
            if s != 9 {
                return s > 9;
            }
        }
        false
    }

    /// One reverse-and-add step, single-threaded: an ascending fused add with
    /// an exact running carry, the reversed operand gathered digit by digit.
    /// The scalar gather keeps this path independent of the funnel machinery
    /// the engine uses, so their agreement pins the funnel down.
    #[cfg(test)]
    pub fn step(&mut self) -> bool {
        let l = self.digits;
        let grew = self.prescan_grow();
        let lp = l + grew as usize;
        let lines = l.div_ceil(DPL);
        let lines_out = lp.div_ceil(DPL);

        let [b0, b1] = &mut self.a;
        let (src, dst) = if self.cur == 0 { (&*b0, b1) } else { (&*b1, b0) };
        dst.resize(lines_out, Limb::new());

        let mut carry = false;
        let mut any_carried = false;
        for k in 0..lines {
            let mut r = [0u8; DPL];
            for (p, byte) in r.iter_mut().enumerate() {
                let d = DPL * k + p;
                if d < l {
                    *byte = digit_at(src, l - 1 - d);
                }
            }
            let r_lo = LimbVec::from_slice(&r[..LV_LEN]);
            let r_hi = LimbVec::from_slice(&r[LV_LEN..]);
            let sum = add_resolve_line(src[k].0, r_lo, r_hi, carry);
            dst[k] = Limb(sum.packed);
            carry = sum.carry_out;
            any_carried |= sum.carried;
        }
        if carry {
            // only reachable when the top line was full to the brim
            debug_assert!(grew && l % DPL == 0);
            set_digit(dst, lp - 1, 1);
        }

        debug_assert!(digit_at(dst, lp - 1) != 0, "prescan missed growth");
        if lp < dst.len() * DPL {
            debug_assert_eq!(
                (lp..dst.len() * DPL).map(|d| digit_at(dst, d)).max().unwrap_or(0),
                0,
                "dirty padding above the top digit"
            );
        }

        self.digits = lp;
        self.cur = 1 - self.cur;

        likely(any_carried)
    }
}

use crate::parallel::{Padded, SpinBarrier, allowed_cpus, pin_participant};
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
    a_src: AtomicPtr<Limb>,
    a_dst: AtomicPtr<Limb>,
    /// Input lines: ceil(digits / DPL).
    lines: AtomicUsize,
    /// The input digit count; the backward stream's line indices and
    /// intra-line phase derive from it.
    digits: AtomicUsize,
    stop: AtomicBool,
    ever_carried: AtomicBool,
    /// 2 * num_threads + 1 entries: block boundaries in lines.
    bounds: Box<[AtomicUsize]>,
    /// 2 * num_threads entries: each block's speculative carry-out.
    block_carry: Box<[Padded<AtomicBool>]>,
}

// The destination pointer partitions by block, the source is read-only
// during the pass, and the barrier orders every access.
unsafe impl Send for SharedPacked {}
unsafe impl Sync for SharedPacked {}

/// The funnel index for `rev_operand`: byte `j` of an assembled plane is a
/// DESCENDING walk over the concatenation of two source planes, so `idx[j]
/// = 63 - s - j` with `s` chosen per phase branch. `s` is a per-iteration
/// constant, so the index vector is built once per block. Because the
/// indices descend, no lowering of the permutation can become a contiguous
/// load; the reversal is what keeps the funnel in registers.
#[inline]
fn funnel_index(s: usize) -> LimbVec {
    const IOTA: LimbVec = {
        let mut lanes = [0u8; LV_LEN];
        let mut i = 0;
        while i < LV_LEN {
            lanes[i] = i as u8;
            i += 1;
        }
        LimbVec::from_array(lanes)
    };
    LimbVec::splat((LV_LEN - 1).wrapping_sub(s) as u8) - IOTA
}

/// Byte `j` of the result is `concat(a, b)[idx[j] mod 128]`, with `idx` from
/// `funnel_index`.
#[inline(always)]
fn funnel(a: LimbVec, idx: LimbVec, b: LimbVec) -> LimbVec {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi",
        not(feature = "no-avx")
    ))]
    unsafe {
        use std::arch::x86_64::{__m512i, _mm512_permutex2var_epi8};
        LimbVec::from(_mm512_permutex2var_epi8(
            __m512i::from(a),
            __m512i::from(idx),
            __m512i::from(b),
        ))
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi",
        not(feature = "no-avx")
    )))]
    {
        let mut out = [0u8; LV_LEN];
        for (j, lane) in out.iter_mut().enumerate() {
            let i = (idx[j] as usize) % (2 * LV_LEN);
            *lane = if i < LV_LEN { a[i] } else { b[i - LV_LEN] };
        }
        LimbVec::from_array(out)
    }
}

/// The funnel index the `rev_operand` branch for `phi` uses.
#[inline]
fn rev_index(phi: usize) -> LimbVec {
    let s = if phi > LV_LEN {
        3 * LV_LEN - phi
    } else {
        2 * LV_LEN - phi
    };
    funnel_index(s)
}

/// The reversed operand for one output line as its two 64-digit planes
/// (low plane first). Output line k covers slots [DPL*k, DPL*(k+1)), which
/// read source digits (L-1-DPL*k) down to (L-DPL*(k+1)): a 128-digit window
/// at intra-line phase `phi = L mod DPL`, spanning source lines `lower` (the
/// window's low end) and `upper` (its high end), each given unpacked. The
/// funnel's descending indices perform the digit reversal; `idx` comes from
/// `rev_index(phi)`.
#[inline(always)]
fn rev_operand(
    phi: usize,
    idx: LimbVec,
    lower: (LimbVec, LimbVec),
    upper: (LimbVec, LimbVec),
) -> (LimbVec, LimbVec) {
    if phi == 0 {
        (lower.1.reverse(), lower.0.reverse())
    } else if phi <= LV_LEN {
        (
            funnel(lower.1, idx, upper.0),
            funnel(lower.0, idx, lower.1),
        )
    } else {
        (
            funnel(upper.0, idx, upper.1),
            funnel(lower.1, idx, upper.0),
        )
    }
}

/// Buffer size in lines above which the pass's destination stores go around
/// the cache: the two buffers no longer fit in an L3, and a fresh-write
/// stream pays a read-for-ownership per line unless stored non-temporally.
const STREAM_MIN_LINES: usize = 1 << 19;

/// Stores one resolved output line, non-temporally when the pass streams.
#[inline(always)]
unsafe fn store_line<const STREAM: bool>(dst: *mut Limb, k: usize, line: LimbVec) {
    #[cfg(all(target_feature = "avx512f", not(feature = "no-avx")))]
    if STREAM {
        unsafe {
            use std::arch::x86_64::{__m512i, _mm512_stream_si512};
            _mm512_stream_si512(dst.add(k).cast(), __m512i::from(line));
        }
        return;
    }

    unsafe {
        *dst.add(k) = Limb(line);
    }
}

impl SharedPacked {
    fn new(num_threads: usize) -> Self {
        Self {
            barrier: SpinBarrier::new(num_threads),
            num_threads,
            a_src: AtomicPtr::new(std::ptr::null_mut()),
            a_dst: AtomicPtr::new(std::ptr::null_mut()),
            lines: AtomicUsize::new(0),
            digits: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            ever_carried: AtomicBool::new(false),
            bounds: (0..=num_threads * 2).map(|_| AtomicUsize::new(0)).collect(),
            block_carry: (0..num_threads * 2)
                .map(|_| Padded(AtomicBool::new(false)))
                .collect(),
        }
    }

    /// The fused pass over block `j`: for each output line, gather the
    /// reversed operand from the backward stream's rolling line pair, add it
    /// to the forward stream's line with a speculative carry-in of zero, and
    /// store the sum slot-aligned into the destination buffer. Lines outside
    /// the source read as zeros (the virtual padding below digit 0 and above
    /// the top line), which also zeros the output's top-line padding: sums
    /// there are 0 + 0, and a carry out of the top digit lands in the first
    /// padding slot as the grown number's leading 1.
    fn run_block(&self, j: usize) {
        if self.lines.load(Relaxed) >= STREAM_MIN_LINES {
            self.run_block_inner::<true>(j);
        } else {
            self.run_block_inner::<false>(j);
        }
    }

    fn run_block_inner<const STREAM: bool>(&self, j: usize) {
        let src = self.a_src.load(Relaxed);
        let dst = self.a_dst.load(Relaxed);
        let lines = self.lines.load(Relaxed);
        let digits = self.digits.load(Relaxed);

        let start = self.bounds[j].load(Relaxed);
        let end = self.bounds[j + 1].load(Relaxed);
        if start >= end {
            self.block_carry[j].0.store(false, Relaxed);
            return;
        }

        let phi = digits % DPL;
        let idx = rev_index(phi);
        let q = (digits / DPL) as isize;

        let load = |m: isize| -> (LimbVec, LimbVec) {
            if m >= 0 && (m as usize) < lines {
                unpack_line(unsafe { (*src.offset(m)).0 })
            } else {
                (LimbVec::splat(0), LimbVec::splat(0))
            }
        };

        let mut upper = load(q - start as isize);
        let mut carry = false;
        let mut any_carried = false;

        for k in start..end {
            let m = q - 1 - k as isize;

            #[cfg(all(target_arch = "x86_64", not(feature = "no-prefetch")))]
            unsafe {
                use std::arch::x86_64::{_MM_HINT_ET0, _MM_HINT_T0, _mm_prefetch};
                _mm_prefetch::<_MM_HINT_T0>(src.wrapping_add(k + 16).cast());
                _mm_prefetch::<_MM_HINT_T0>(src.wrapping_offset(m - 16).cast());
                if !STREAM {
                    _mm_prefetch::<_MM_HINT_ET0>(dst.wrapping_add(k + 16).cast());
                }
            }

            let lower = load(m);
            let (r_lo, r_hi) = rev_operand(phi, idx, lower, upper);
            let sum = add_resolve_line(unsafe { (*src.add(k)).0 }, r_lo, r_hi, carry);
            unsafe { store_line::<STREAM>(dst, k, sum.packed) };
            carry = sum.carry_out;
            any_carried |= sum.carried;
            upper = lower;
        }

        #[cfg(all(target_feature = "avx512f", not(feature = "no-avx")))]
        if STREAM {
            // non-temporal stores are weakly ordered; drain them before the
            // end barrier publishes the buffer
            unsafe { std::arch::x86_64::_mm_sfence() };
        }

        self.block_carry[j].0.store(carry, Relaxed);
        if likely(any_carried) {
            self.ever_carried.store(true, Relaxed);
        }
    }

    /// Both blocks of participant `t`, mirror-owned like the byte engine:
    /// a block's backward stream reads the mirror of its own line range,
    /// which is the participant's other block, so below the streaming
    /// threshold the second block's reads hit the lines the first block
    /// already pulled in.
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

        // Growth past the top line is decided by the prescan; the resize
        // zeroes the appended line, whose single digit is set after the
        // carry scan confirms it. The destination never shrinks, and its
        // stale lines all sit below `lines`, which the pass overwrites.
        let next = 1 - x.cur;
        x.a[next].resize(lines_out, Limb::new());

        shared.a_src.store(x.a[x.cur].as_ptr().cast_mut(), Relaxed);
        shared.a_dst.store(x.a[next].as_mut_ptr(), Relaxed);
        shared.lines.store(lines, Relaxed);
        shared.digits.store(digits, Relaxed);
        shared.ever_carried.store(false, Relaxed);
        for k in 0..=num_blocks {
            shared.bounds[k].store(k * lines / num_blocks, Relaxed);
        }

        shared.barrier.wait();
        shared.run_blocks(0);

        // Serial carry resolution across blocks: a block whose true carry-in
        // turned out to be one gets a decimal increment at its base.
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
                let (_, escaped) =
                    increment_digits(&mut x.a[next], start * DPL, (end * DPL).min(out_digits));
                carry_out |= escaped;
            }
            carry = carry_out;
        }

        if unlikely(carry) {
            // only reachable when the input's top line was full to the brim
            cold_path();
            debug_assert!(grew && digits % DPL == 0);
            set_digit(&mut x.a[next], out_digits - 1, 1);
        }

        x.digits = out_digits;
        x.cur = next;

        debug_assert!(
            digit_at(&x.a[next], out_digits - 1) != 0,
            "prescan missed growth"
        );
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

/// The iteration loop over the packed representation. The engine runs with
/// one participant while the number is small, and widens at the same
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
                "Packed engine: {target_threads} thread{} at iteration {i} ({num_limbs} limbs, {:.3} s elapsed)",
                if target_threads == 1 {""} else {"s"},
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
