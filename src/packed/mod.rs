//! The single-copy nibble-packed representation: only the number itself is
//! kept packed (two digits per byte, the checkpoint format), ping-ponged
//! between two buffers. The reverse-and-add pass reads the current buffer
//! with two streams -- forward for a[d], backward from the top for
//! a[L-1-d], the reversed operand assembled in registers by a descending
//! funnel permutation -- and writes the sum slot-aligned into the other
//! buffer.
//!
//! All passes walk mirror block pairs as interleaved chunks, so the
//! backward stream reads lines its round's partner chunk just pulled into
//! cache and each source line costs one memory fetch. Above the streaming
//! threshold the engine additionally fuses iterations per pass (`step2`,
//! `step3`): each round materializes the intermediate steps' output for
//! its chunk pair in per-thread scratch, one level per fused step, and
//! computes the final step from the last level, so the number crosses
//! DRAM once -- one read and one write -- per two or three iterations,
//! with cross-iteration carry misspeculation repaired after the pass from
//! scalar ground truth (a few dozen lines per pass, independent of size).
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
pub const DPL: usize = 2 * LV_LEN;

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
pub fn digit_at(lines: &[Limb], d: usize) -> u8 {
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
pub fn set_digit(lines: &mut [Limb], d: usize, digit: u8) {
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
            let hi = if pair.len() == 2 {
                pair[1].0
            } else {
                LimbVec::splat(0)
            };
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
    pub(crate) const fn a_cur(&self) -> &[Limb] {
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
        let lp = l + usize::from(grew);
        let lines = l.div_ceil(DPL);
        let lines_out = lp.div_ceil(DPL);

        let [b0, b1] = &mut self.a;
        let (src, dst) = if self.cur == 0 {
            (&*b0, b1)
        } else {
            (&*b1, b0)
        };
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
            debug_assert!(grew && l.is_multiple_of(DPL));
            set_digit(dst, lp - 1, 1);
        }

        debug_assert!(digit_at(dst, lp - 1) != 0, "prescan missed growth");
        if lp < dst.len() * DPL {
            debug_assert_eq!(
                (lp..dst.len() * DPL)
                    .map(|d| digit_at(dst, d))
                    .max()
                    .unwrap_or(0),
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
    /// Whether the pass fuses two iterations (`run_pair2`) or runs one
    /// (`run_pair`).
    fused: AtomicBool,
    /// Whether the pass fuses three iterations (`run_pair3`); takes
    /// precedence over `fused`.
    fused3: AtomicBool,
    /// Fused passes only: the first step's exact digit count.
    digits1: AtomicUsize,
    /// Triple pass only: the second step's exact digit count.
    digits2: AtomicUsize,
    /// Fused passes only: whether any digit carried in the first step.
    carried1: AtomicBool,
    /// Triple pass only: whether any digit carried in the second step, as
    /// the pass saw it (repair recomputes it exactly on misspeculation).
    carried2: AtomicBool,
    /// Fused passes only: whether every first-step digit equaled its mirror,
    /// AND-accumulated across threads during the second step's add.
    pal1: AtomicBool,
    /// Triple pass only: `pal1` for the second step's digits, accumulated
    /// during the third step's add.
    pal2: AtomicBool,
    /// Triple pass only: whether any worker's repair found a first-step
    /// range whose speculative carry-in was wrong.
    rip1_seen: AtomicBool,
    /// Triple pass only: whether any worker's repair found a wrong
    /// second-step digit.
    wrong2_seen: AtomicBool,
    /// `2 * num_threads + 1` entries: block boundaries in lines.
    bounds: Box<[AtomicUsize]>,
    /// `2 * num_threads` entries: each block's speculative carry-out. Only
    /// the low blocks (below `num_threads`) use theirs; high blocks carry
    /// per chunk instead.
    block_carry: Box<[Padded<AtomicBool>]>,
    /// Carry-outs of the high blocks' chunks, published by the coordinator
    /// each step. The chunk starting at line `c` of block `j` owns slot
    /// `j + c / CHUNK_LINES`: block indices break ties between the two
    /// chunks a non-aligned block boundary splits a grid cell into.
    chunk_carry: AtomicPtr<AtomicBool>,
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
        (funnel(lower.1, idx, upper.0), funnel(lower.0, idx, lower.1))
    } else {
        (funnel(upper.0, idx, upper.1), funnel(lower.1, idx, upper.0))
    }
}

/// Buffer size in lines above which the pass's destination stores go around
/// the cache: the two buffers no longer fit in an L3, and a fresh-write
/// stream pays a read-for-ownership per line unless stored non-temporally.
const STREAM_MIN_LINES: usize = 1 << 19;

/// Chunk granularity of the mirror-interleaved pair walk, in lines. A round
/// touches two chunk-sized source ranges (128KB at 1024 lines), which must
/// stay resident in one core's L2 between the low chunk's pass and the high
/// chunk's, where the same two ranges are read with the streams swapped.
/// Tests shrink it so their small fixtures cross chunk boundaries.
const CHUNK_LINES: usize = if cfg!(test) { 4 } else { 1 << 13 };

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
            fused: AtomicBool::new(false),
            fused3: AtomicBool::new(false),
            digits1: AtomicUsize::new(0),
            digits2: AtomicUsize::new(0),
            carried1: AtomicBool::new(false),
            carried2: AtomicBool::new(false),
            pal1: AtomicBool::new(true),
            pal2: AtomicBool::new(true),
            rip1_seen: AtomicBool::new(false),
            wrong2_seen: AtomicBool::new(false),
            bounds: (0..=num_threads * 2).map(|_| AtomicUsize::new(0)).collect(),
            block_carry: (0..num_threads * 2)
                .map(|_| Padded(AtomicBool::new(false)))
                .collect(),
            chunk_carry: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// The fused pass over participant `t`'s mirror block pair, walked as
    /// interleaved chunks: the low block ascends while the high block
    /// descends, one chunk each per round. A chunk's backward stream reads
    /// the mirror of its own line range, which is (to within boundary
    /// rounding) the other chunk of the same round, so each round pulls two
    /// chunk-sized source ranges from memory and the second chunk's reads
    /// hit them in cache. For each output line, the reversed operand is
    /// gathered from the backward stream's rolling line pair and added to
    /// the forward stream's line with a speculative carry-in of zero at
    /// every chain break -- the low block's chain runs unbroken through its
    /// ascending chunks, while each descending high chunk starts its own.
    /// Lines outside the source read as zeros (the virtual padding below
    /// digit 0 and above the top line), which also zeros the output's
    /// top-line padding: sums there are 0 + 0, and a carry out of the top
    /// digit lands in the first padding slot as the grown number's leading
    /// 1.
    fn run_pair(&self, t: usize) {
        if self.lines.load(Relaxed) >= STREAM_MIN_LINES {
            self.run_pair_inner::<true>(t);
        } else {
            self.run_pair_inner::<false>(t);
        }
    }

    fn run_pair_inner<const STREAM: bool>(&self, t: usize) {
        let src = self.a_src.load(Relaxed);
        let dst = self.a_dst.load(Relaxed);
        let lines = self.lines.load(Relaxed);
        let digits = self.digits.load(Relaxed);
        let chunk_carry = self.chunk_carry.load(Relaxed);

        let lo = t;
        let hi = self.num_threads * 2 - 1 - t;
        let lo_end = self.bounds[lo + 1].load(Relaxed);
        let hi_start = self.bounds[hi].load(Relaxed);
        let mut lo_c = self.bounds[lo].load(Relaxed);
        let mut hi_c = self.bounds[hi + 1].load(Relaxed);

        let phi = digits % DPL;
        let idx = rev_index(phi);
        let q = (digits / DPL).cast_signed();

        let load = |m: isize| -> (LimbVec, LimbVec) {
            if m >= 0 && m.cast_unsigned() < lines {
                unpack_line(unsafe { (*src.offset(m)).0 })
            } else {
                (LimbVec::splat(0), LimbVec::splat(0))
            }
        };

        let mut any_carried = false;

        let mut run_range = |start: usize, end: usize, carry_in: bool| -> bool {
            let mut upper = load(q - start.cast_signed());
            let mut carry = carry_in;
            for k in start..end {
                let m = q - 1 - k.cast_signed();

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
            carry
        };

        let mut lo_carry = false;
        while lo_c < lo_end || hi_c > hi_start {
            if lo_c < lo_end {
                let next = ((lo_c / CHUNK_LINES + 1) * CHUNK_LINES).min(lo_end);
                lo_carry = run_range(lo_c, next, lo_carry);
                lo_c = next;
            }
            if hi_c > hi_start {
                let prev = ((hi_c - 1) / CHUNK_LINES * CHUNK_LINES).max(hi_start);
                let carry = run_range(prev, hi_c, false);
                unsafe { (*chunk_carry.add(chunk_slot(hi, prev))).store(carry, Relaxed) };
                hi_c = prev;
            }
        }

        #[cfg(all(target_feature = "avx512f", not(feature = "no-avx")))]
        if STREAM {
            // non-temporal stores are weakly ordered; drain them before the
            // end barrier publishes the buffer
            unsafe { std::arch::x86_64::_mm_sfence() };
        }

        self.block_carry[lo].0.store(lo_carry, Relaxed);
        if likely(any_carried) {
            self.ever_carried.store(true, Relaxed);
        }
    }

    #[inline]
    fn run_blocks(&self, t: usize, scratch: &mut FusedScratch) {
        if self.fused3.load(Relaxed) {
            self.run_pair3(t, scratch);
        } else if self.fused.load(Relaxed) {
            self.run_pair2(t, scratch);
        } else {
            self.run_pair(t);
        }
        self.barrier.wait();
    }

    fn run_pair3(&self, t: usize, scratch: &mut FusedScratch) {
        if self.lines.load(Relaxed) >= STREAM_MIN_LINES {
            self.run_pair3_inner::<true>(t, scratch);
        } else {
            self.run_pair3_inner::<false>(t, scratch);
        }
        self.repair_pair3(t);
    }

    fn run_pair2(&self, t: usize, scratch: &mut FusedScratch) {
        if self.lines.load(Relaxed) >= STREAM_MIN_LINES {
            self.run_pair2_inner::<true>(t, scratch);
        } else {
            self.run_pair2_inner::<false>(t, scratch);
        }
    }

    /// The fused two-iteration pass over participant `t`'s mirror block
    /// pair. Each round first materializes the two first-step ranges the
    /// round consumes into per-thread scratch (phase A: the same
    /// funnel-and-add as the single-step pass, reading the input buffer),
    /// then computes the round's two second-step output chunks from scratch
    /// (phase B) and stores them slot-aligned into the destination. The
    /// input buffer is read once and the destination written once per two
    /// iterations; everything in between lives in cache.
    ///
    /// Every scratch range and every high output chunk speculates a
    /// carry-in of zero; the low output chain runs unbroken through its
    /// ascending chunks. Scratch misspeculation leaks wrong digits into
    /// consumed operands and is repaired after the pass by the
    /// coordinator, which replays `for_each_round`; second-step chunk
    /// carries resolve exactly like the single-step pass's.
    fn run_pair2_inner<const STREAM: bool>(&self, t: usize, scratch: &mut FusedScratch) {
        let src = self.a_src.load(Relaxed);
        let dst = self.a_dst.load(Relaxed);
        let lines = self.lines.load(Relaxed);
        let digits = self.digits.load(Relaxed);
        let digits1 = self.digits1.load(Relaxed);
        let chunk_carry = self.chunk_carry.load(Relaxed);

        let lo = t;
        let hi = self.num_threads * 2 - 1 - t;
        let lo_bounds = (
            self.bounds[lo].load(Relaxed),
            self.bounds[lo + 1].load(Relaxed),
        );
        let hi_bounds = (
            self.bounds[hi].load(Relaxed),
            self.bounds[hi + 1].load(Relaxed),
        );

        let phi0 = digits % DPL;
        let idx0 = rev_index(phi0);
        let q0 = (digits / DPL).cast_signed();
        let phi1 = digits1 % DPL;
        let idx1 = rev_index(phi1);
        let q1 = digits1 / DPL;
        let q1_lines = digits1.div_ceil(DPL);

        let load_a = |m: isize| -> (LimbVec, LimbVec) {
            if m >= 0 && m.cast_unsigned() < lines {
                unpack_line(unsafe { (*src.offset(m)).0 })
            } else {
                (LimbVec::splat(0), LimbVec::splat(0))
            }
        };

        let FusedScratch {
            lo: scratch_lo,
            hi: scratch_hi,
            ..
        } = scratch;

        let mut carried1 = false;
        let mut carried2 = false;
        let mut all_eq = true;
        let mut lo_carry = false;
        let hi_stride = fused_hi_stride(q1_lines, self.num_threads * 2);
        let mut hi_ord = 0usize;

        for_each_round(lo_bounds, hi_bounds, q1, q1_lines, |round| {
            // phase A: first-step lines for both ranges, into scratch
            for (range, scratch) in [
                (round.r_lo, &mut *scratch_lo),
                (round.r_hi, &mut *scratch_hi),
            ] {
                let (s, e) = range;
                if s >= e {
                    continue;
                }
                assert!(e - s <= SCRATCH_LINES, "fused scratch range overflow");
                let mut upper = load_a(q0 - s.cast_signed());
                let mut carry = false;
                for j in s..e {
                    let m = q0 - 1 - j.cast_signed();

                    #[cfg(all(target_arch = "x86_64", not(feature = "no-prefetch")))]
                    unsafe {
                        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                        _mm_prefetch::<_MM_HINT_T0>(src.wrapping_add(j + 16).cast());
                        _mm_prefetch::<_MM_HINT_T0>(src.wrapping_offset(m - 16).cast());
                    }

                    let lower = load_a(m);
                    let (r_lo, r_hi) = rev_operand(phi0, idx0, lower, upper);
                    let fwd = if j < lines {
                        unsafe { (*src.add(j)).0 }
                    } else {
                        LimbVec::splat(0)
                    };
                    let sum = add_resolve_line(fwd, r_lo, r_hi, carry);
                    scratch[j - s] = Limb(sum.packed);
                    carry = sum.carry_out;
                    carried1 |= sum.carried;
                    upper = lower;
                }
            }

            // phase B: the round's two output chunks, from scratch. The low
            // chunk reads forward from r_lo and backward from r_hi; the
            // high chunk the reverse.
            for (chunk, fwd_range, fwd_scr, rev_range, rev_scr, is_hi) in [
                (
                    round.lo,
                    round.r_lo,
                    &scratch_lo,
                    round.r_hi,
                    &scratch_hi,
                    false,
                ),
                (
                    round.hi,
                    round.r_hi,
                    &scratch_hi,
                    round.r_lo,
                    &scratch_lo,
                    true,
                ),
            ] {
                let (x0, x1) = chunk;
                if x0 >= x1 {
                    continue;
                }
                let load_s = |m: isize| -> (LimbVec, LimbVec) {
                    if m >= 0 && m.cast_unsigned() >= rev_range.0 && m.cast_unsigned() < rev_range.1
                    {
                        unpack_line(rev_scr[m.cast_unsigned() - rev_range.0].0)
                    } else {
                        debug_assert!(
                            m < 0 || m.cast_unsigned() >= q1_lines,
                            "fused pass read outside its scratch ranges"
                        );
                        (LimbVec::splat(0), LimbVec::splat(0))
                    }
                };
                let mut upper = load_s(q1.cast_signed() - x0.cast_signed());
                let mut carry = if is_hi { false } else { lo_carry };
                for k in x0..x1 {
                    let m = q1.cast_signed() - 1 - k.cast_signed();

                    #[cfg(all(target_arch = "x86_64", not(feature = "no-prefetch")))]
                    if !STREAM {
                        unsafe {
                            use std::arch::x86_64::{_MM_HINT_ET0, _mm_prefetch};
                            _mm_prefetch::<_MM_HINT_ET0>(dst.wrapping_add(k + 16).cast());
                        }
                    }

                    let lower = load_s(m);
                    let (r_lo, r_hi) = rev_operand(phi1, idx1, lower, upper);
                    let fwd = fwd_scr[k - fwd_range.0].0;
                    let (f_lo, f_hi) = unpack_line(fwd);
                    all_eq &= f_lo.simd_eq(r_lo).all() && f_hi.simd_eq(r_hi).all();
                    let sum = add_resolve_line(fwd, r_lo, r_hi, carry);
                    unsafe { store_line::<STREAM>(dst, k, sum.packed) };
                    carry = sum.carry_out;
                    carried2 |= sum.carried;
                    upper = lower;
                }
                if is_hi {
                    assert!(hi_ord < hi_stride, "fused chunk ordinal overflow");
                    let slot = (hi - self.num_threads) * hi_stride + hi_ord;
                    unsafe { (*chunk_carry.add(slot)).store(carry, Relaxed) };
                    hi_ord += 1;
                } else {
                    lo_carry = carry;
                }
            }
        });

        #[cfg(all(target_feature = "avx512f", not(feature = "no-avx")))]
        if STREAM {
            // non-temporal stores are weakly ordered; drain them before the
            // end barrier publishes the buffer
            unsafe { std::arch::x86_64::_mm_sfence() };
        }

        self.block_carry[lo].0.store(lo_carry, Relaxed);
        if likely(carried1) {
            self.carried1.store(true, Relaxed);
        }
        if likely(carried2) {
            self.ever_carried.store(true, Relaxed);
        }
        if !all_eq {
            self.pal1.store(false, Relaxed);
        }
    }

    /// The fused three-iteration pass over participant `t`'s mirror block
    /// pair: `run_pair2_inner` with one more scratch level. Per round,
    /// phase A materializes the first-step ranges the round consumes into
    /// scratch (reading the input buffer), phase B computes the
    /// second-step ranges from them into the second scratch level, and
    /// phase C computes the round's two third-step output chunks from
    /// those and stores them slot-aligned into the destination -- one read
    /// and one write of the number per THREE iterations.
    ///
    /// Every scratch range at both levels and every high output chunk
    /// speculates a carry-in of zero; the low output chain runs unbroken.
    /// Scratch misspeculation at either level leaks wrong digits into
    /// consumed operands and is repaired after the pass by the
    /// coordinator (`repair_fused3`), which replays `for_each_round3`.
    fn run_pair3_inner<const STREAM: bool>(&self, t: usize, scratch: &mut FusedScratch) {
        let src = self.a_src.load(Relaxed);
        let dst = self.a_dst.load(Relaxed);
        let lines = self.lines.load(Relaxed);
        let digits = self.digits.load(Relaxed);
        let digits1 = self.digits1.load(Relaxed);
        let digits2 = self.digits2.load(Relaxed);
        let chunk_carry = self.chunk_carry.load(Relaxed);

        let lo = t;
        let hi = self.num_threads * 2 - 1 - t;
        let lo_bounds = (
            self.bounds[lo].load(Relaxed),
            self.bounds[lo + 1].load(Relaxed),
        );
        let hi_bounds = (
            self.bounds[hi].load(Relaxed),
            self.bounds[hi + 1].load(Relaxed),
        );

        let phi0 = digits % DPL;
        let idx0 = rev_index(phi0);
        let q0 = (digits / DPL).cast_signed();
        let phi1 = digits1 % DPL;
        let idx1 = rev_index(phi1);
        let q1 = digits1 / DPL;
        let q1_lines = digits1.div_ceil(DPL);
        let phi2 = digits2 % DPL;
        let idx2 = rev_index(phi2);
        let q2 = digits2 / DPL;
        let q2_lines = digits2.div_ceil(DPL);

        let load_a = |m: isize| -> (LimbVec, LimbVec) {
            if m >= 0 && m.cast_unsigned() < lines {
                unpack_line(unsafe { (*src.offset(m)).0 })
            } else {
                (LimbVec::splat(0), LimbVec::splat(0))
            }
        };

        let FusedScratch {
            lo: scratch_lo,
            hi: scratch_hi,
            lo2: scratch_lo2,
            hi2: scratch_hi2,
        } = scratch;

        let mut carried1 = false;
        let mut carried2 = false;
        let mut carried3 = false;
        let mut all_eq1 = true;
        let mut all_eq2 = true;
        let mut lo_carry = false;
        let hi_stride = fused_hi_stride(q2_lines, self.num_threads * 2);
        let mut hi_ord = 0usize;

        for_each_round3(lo_bounds, hi_bounds, q2, q2_lines, q1, q1_lines, |round| {
            // phase A: first-step lines for both ranges, into scratch.
            // The two ranges carry independent speculated chains and are
            // walked interleaved so two carry chains stay in flight.
            let (s0, e0) = round.r1_lo;
            let (s1, e1) = round.r1_hi;
            let n0 = e0.saturating_sub(s0);
            let n1 = e1.saturating_sub(s1);
            assert!(
                n0 <= SCRATCH_LINES && n1 <= SCRATCH_LINES,
                "fused scratch range overflow"
            );
            let mut up0 = load_a(q0 - s0.cast_signed());
            let mut up1 = load_a(q0 - s1.cast_signed());
            let mut c0 = false;
            let mut c1 = false;
            let mut a_line =
                |j: usize, upper: &mut (LimbVec, LimbVec), carry: &mut bool, out: &mut Limb| {
                    let m = q0 - 1 - j.cast_signed();

                    #[cfg(all(target_arch = "x86_64", not(feature = "no-prefetch")))]
                    unsafe {
                        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                        _mm_prefetch::<_MM_HINT_T0>(src.wrapping_add(j + 16).cast());
                        _mm_prefetch::<_MM_HINT_T0>(src.wrapping_offset(m - 16).cast());
                    }

                    let lower = load_a(m);
                    let (r_lo, r_hi) = rev_operand(phi0, idx0, lower, *upper);
                    let fwd = if j < lines {
                        unsafe { (*src.add(j)).0 }
                    } else {
                        LimbVec::splat(0)
                    };
                    let sum = add_resolve_line(fwd, r_lo, r_hi, *carry);
                    *out = Limb(sum.packed);
                    *carry = sum.carry_out;
                    carried1 |= sum.carried;
                    *upper = lower;
                };
            let n = n0.min(n1);
            for i in 0..n {
                a_line(s0 + i, &mut up0, &mut c0, &mut scratch_lo[i]);
                a_line(s1 + i, &mut up1, &mut c1, &mut scratch_hi[i]);
            }
            for i in n..n0 {
                a_line(s0 + i, &mut up0, &mut c0, &mut scratch_lo[i]);
            }
            for i in n..n1 {
                a_line(s1 + i, &mut up1, &mut c1, &mut scratch_hi[i]);
            }
            drop(a_line);

            // phase B: second-step lines for both ranges, from the first
            // scratch level into the second, the two independent chains
            // interleaved. A low-range line reads forward from the low
            // production and backward from the high one; a high-range line
            // the reverse.
            let load_s1 = |m: isize, rev_scr: &[Limb], rev_range: (usize, usize)| {
                if m >= 0 && m.cast_unsigned() >= rev_range.0 && m.cast_unsigned() < rev_range.1 {
                    unpack_line(rev_scr[m.cast_unsigned() - rev_range.0].0)
                } else {
                    debug_assert!(
                        m < 0 || m.cast_unsigned() >= q1_lines,
                        "fused pass read outside its scratch ranges"
                    );
                    (LimbVec::splat(0), LimbVec::splat(0))
                }
            };
            let (s0, e0) = round.r2_lo;
            let (s1, e1) = round.r2_hi;
            let n0 = e0.saturating_sub(s0);
            let n1 = e1.saturating_sub(s1);
            assert!(
                n0 <= SCRATCH_LINES && n1 <= SCRATCH_LINES,
                "fused scratch range overflow"
            );
            let zero = (LimbVec::splat(0), LimbVec::splat(0));
            let mut up0 = if n0 > 0 {
                load_s1(q1.cast_signed() - s0.cast_signed(), scratch_hi, round.r1_hi)
            } else {
                zero
            };
            let mut up1 = if n1 > 0 {
                load_s1(q1.cast_signed() - s1.cast_signed(), scratch_lo, round.r1_lo)
            } else {
                zero
            };
            let mut c0 = false;
            let mut c1 = false;
            let mut b_line = |j: usize,
                              upper: &mut (LimbVec, LimbVec),
                              carry: &mut bool,
                              fwd_scr: &[Limb],
                              fwd_base: usize,
                              rev_scr: &[Limb],
                              rev_range: (usize, usize),
                              out: &mut Limb| {
                let m = q1.cast_signed() - 1 - j.cast_signed();
                let lower = load_s1(m, rev_scr, rev_range);
                let (r_lo, r_hi) = rev_operand(phi1, idx1, lower, *upper);
                let fwd = fwd_scr[j - fwd_base].0;
                let (f_lo, f_hi) = unpack_line(fwd);
                all_eq1 &= f_lo.simd_eq(r_lo).all() && f_hi.simd_eq(r_hi).all();
                let sum = add_resolve_line(fwd, r_lo, r_hi, *carry);
                *out = Limb(sum.packed);
                *carry = sum.carry_out;
                carried2 |= sum.carried;
                *upper = lower;
            };
            let n = n0.min(n1);
            for i in 0..n {
                b_line(
                    s0 + i,
                    &mut up0,
                    &mut c0,
                    scratch_lo,
                    round.r1_lo.0,
                    scratch_hi,
                    round.r1_hi,
                    &mut scratch_lo2[i],
                );
                b_line(
                    s1 + i,
                    &mut up1,
                    &mut c1,
                    scratch_hi,
                    round.r1_hi.0,
                    scratch_lo,
                    round.r1_lo,
                    &mut scratch_hi2[i],
                );
            }
            for i in n..n0 {
                b_line(
                    s0 + i,
                    &mut up0,
                    &mut c0,
                    scratch_lo,
                    round.r1_lo.0,
                    scratch_hi,
                    round.r1_hi,
                    &mut scratch_lo2[i],
                );
            }
            for i in n..n1 {
                b_line(
                    s1 + i,
                    &mut up1,
                    &mut c1,
                    scratch_hi,
                    round.r1_hi.0,
                    scratch_lo,
                    round.r1_lo,
                    &mut scratch_hi2[i],
                );
            }
            drop(b_line);

            // phase C: the round's two output chunks, from the second
            // scratch level, the low chunk's chained carry and the high
            // chunk's speculated one interleaved.
            let load_s2 = |m: isize, rev_scr: &[Limb], rev_range: (usize, usize)| {
                if m >= 0 && m.cast_unsigned() >= rev_range.0 && m.cast_unsigned() < rev_range.1 {
                    unpack_line(rev_scr[m.cast_unsigned() - rev_range.0].0)
                } else {
                    debug_assert!(
                        m < 0 || m.cast_unsigned() >= q2_lines,
                        "fused pass read outside its scratch ranges"
                    );
                    (LimbVec::splat(0), LimbVec::splat(0))
                }
            };
            let (x0, x1) = round.lo;
            let (y0, y1) = round.hi;
            let n0 = x1.saturating_sub(x0);
            let n1 = y1.saturating_sub(y0);
            let mut up0 = if n0 > 0 {
                load_s2(
                    q2.cast_signed() - x0.cast_signed(),
                    scratch_hi2,
                    round.r2_hi,
                )
            } else {
                zero
            };
            let mut up1 = if n1 > 0 {
                load_s2(
                    q2.cast_signed() - y0.cast_signed(),
                    scratch_lo2,
                    round.r2_lo,
                )
            } else {
                zero
            };
            let mut c0 = lo_carry;
            let mut c1 = false;
            let mut c_line = |k: usize,
                              upper: &mut (LimbVec, LimbVec),
                              carry: &mut bool,
                              fwd_scr: &[Limb],
                              fwd_base: usize,
                              rev_scr: &[Limb],
                              rev_range: (usize, usize)| {
                let m = q2.cast_signed() - 1 - k.cast_signed();

                #[cfg(all(target_arch = "x86_64", not(feature = "no-prefetch")))]
                if !STREAM {
                    unsafe {
                        use std::arch::x86_64::{_MM_HINT_ET0, _mm_prefetch};
                        _mm_prefetch::<_MM_HINT_ET0>(dst.wrapping_add(k + 16).cast());
                    }
                }

                let lower = load_s2(m, rev_scr, rev_range);
                let (r_lo, r_hi) = rev_operand(phi2, idx2, lower, *upper);
                let fwd = fwd_scr[k - fwd_base].0;
                let (f_lo, f_hi) = unpack_line(fwd);
                all_eq2 &= f_lo.simd_eq(r_lo).all() && f_hi.simd_eq(r_hi).all();
                let sum = add_resolve_line(fwd, r_lo, r_hi, *carry);
                unsafe { store_line::<STREAM>(dst, k, sum.packed) };
                *carry = sum.carry_out;
                carried3 |= sum.carried;
                *upper = lower;
            };
            let n = n0.min(n1);
            for i in 0..n {
                c_line(
                    x0 + i,
                    &mut up0,
                    &mut c0,
                    scratch_lo2,
                    round.r2_lo.0,
                    scratch_hi2,
                    round.r2_hi,
                );
                c_line(
                    y0 + i,
                    &mut up1,
                    &mut c1,
                    scratch_hi2,
                    round.r2_hi.0,
                    scratch_lo2,
                    round.r2_lo,
                );
            }
            for i in n..n0 {
                c_line(
                    x0 + i,
                    &mut up0,
                    &mut c0,
                    scratch_lo2,
                    round.r2_lo.0,
                    scratch_hi2,
                    round.r2_hi,
                );
            }
            for i in n..n1 {
                c_line(
                    y0 + i,
                    &mut up1,
                    &mut c1,
                    scratch_hi2,
                    round.r2_hi.0,
                    scratch_lo2,
                    round.r2_lo,
                );
            }
            drop(c_line);
            lo_carry = c0;
            if n1 > 0 {
                assert!(hi_ord < hi_stride, "fused chunk ordinal overflow");
                let slot = (hi - self.num_threads) * hi_stride + hi_ord;
                unsafe { (*chunk_carry.add(slot)).store(c1, Relaxed) };
                hi_ord += 1;
            }
        });

        #[cfg(all(target_feature = "avx512f", not(feature = "no-avx")))]
        if STREAM {
            // non-temporal stores are weakly ordered; drain them before the
            // end barrier publishes the buffer
            unsafe { std::arch::x86_64::_mm_sfence() };
        }

        self.block_carry[lo].0.store(lo_carry, Relaxed);
        if likely(carried1) {
            self.carried1.store(true, Relaxed);
        }
        if likely(carried2) {
            self.carried2.store(true, Relaxed);
        }
        if likely(carried3) {
            self.ever_carried.store(true, Relaxed);
        }
        if !all_eq1 {
            self.pal1.store(false, Relaxed);
        }
        if !all_eq2 {
            self.pal2.store(false, Relaxed);
        }
    }

    /// Repairs the triple pass's misspeculation for participant `t`'s own
    /// block pair, on the worker, between the pass and the end barrier:
    /// every affected line, every convergence cap, and every carry slot a
    /// walk can correct is pair-local, and ground truth reads only the
    /// immutable source, so the repairs never touch another pair's lines.
    ///
    /// Level-1 range bases resolve exactly as in `repair_fused`; their
    /// ripples, together with wrongly speculated level-2 range bases, make
    /// second-step scratch digits wrong, which a dual chain walk (the
    /// pass's values against ground truth) enumerates per range. Every
    /// wrong second-step digit's consumers -- its own line through the
    /// forward stream and its mirror's through the backward -- are then
    /// recomputed from exact third-step ground truth, walking upward until
    /// the recomputed line matches what the pass stored; a walk reaching
    /// its chunk's speculation boundary corrects the recorded carry-out
    /// instead. Misspeculation sightings are reported through `rip1_seen`
    /// and `wrong2_seen` for the coordinator's exact carry flags.
    fn repair_pair3(&self, worker: usize) {
        let src_ptr = self.a_src.load(Relaxed);
        let dst = self.a_dst.load(Relaxed);
        let lines = self.lines.load(Relaxed);
        let l = self.digits.load(Relaxed);
        let l1 = self.digits1.load(Relaxed);
        let l2 = self.digits2.load(Relaxed);
        let chunk_carry = self.chunk_carry.load(Relaxed);
        let q1 = l1 / DPL;
        let q1_lines = l1.div_ceil(DPL);
        let q2 = l2 / DPL;
        let q2_lines = l2.div_ceil(DPL);
        let num_threads = self.num_threads;
        let num_blocks = num_threads * 2;
        let bound = |j: usize| self.bounds[j].load(Relaxed);

        let src = unsafe { std::slice::from_raw_parts(src_ptr, lines) };

        // level-1 ripples, exactly as in `repair_fused`
        let ripple1 = |(base_line, end_line): (usize, usize)| -> Vec<(usize, u8)> {
            let base = base_line * DPL;
            if base_line == 0 || base_line >= end_line || base >= l1 {
                return Vec::new();
            }
            if likely(!s1_carry_into(src, l, base)) {
                return Vec::new();
            }
            cold_path();
            let top = (end_line * DPL).min(l1);
            let mut out = Vec::new();
            let mut carry = 0u8;
            for p in base..top {
                let sum = s1_sum(src, l, p) + carry;
                let spec = sum % 10;
                carry = u8::from(sum >= 10);
                out.push((p, spec));
                if spec != 9 {
                    break;
                }
            }
            out
        };

        // the first-step digit the pass consumed at position `p` through a
        // stream with the given ripple list
        let p1c = |p: usize, rips: &[(usize, u8)]| -> u8 {
            for &(rp, spec) in rips {
                if rp == p {
                    return spec;
                }
            }
            s1_digit(src, l, p)
        };

        struct Consumer3 {
            /// Wrong second-step digits, with the values the pass consumed,
            /// of the production this chunk's forward stream reads.
            fwd_w2: Vec<(usize, u8)>,
            /// The same for the backward stream's production.
            bwd_w2: Vec<(usize, u8)>,
            spec_base: usize,
            cap: usize,
            slot: usize,
            hi_side: bool,
            affected: Vec<usize>,
        }
        let mut consumers: Vec<Consumer3> = Vec::new();
        let mut any_rip1 = false;
        let mut any_wrong2 = false;

        let hi_stride = fused_hi_stride(q2_lines, num_blocks);
        let lo_j = worker;
        let hi_j = num_blocks - 1 - worker;
        let lo_b = (bound(lo_j), bound(lo_j + 1));
        let hi_b = (bound(hi_j), bound(hi_j + 1));
        let mut hi_ord = 0usize;
        for_each_round3(lo_b, hi_b, q2, q2_lines, q1, q1_lines, |round| {
            let ord = hi_ord;
            if round.hi.0 < round.hi.1 {
                hi_ord += 1;
            }
            let rip1_lo = ripple1(round.r1_lo);
            let rip1_hi = ripple1(round.r1_hi);
            any_rip1 |= !(rip1_lo.is_empty() && rip1_hi.is_empty());

            // wrong second-step digits of one production, by a dual chain
            // walk (pass values against ground truth) from each seed: the
            // range base if its true carry-in was one, and every position
            // whose consumed first-step digit a ripple changed. Truncated
            // at the range top: consumers beyond it read other
            // productions.
            let wrong2 = |r2: (usize, usize),
                          own: &[(usize, u8)],
                          other: &[(usize, u8)]|
             -> Vec<(usize, u8)> {
                if r2.0 >= r2.1 {
                    return Vec::new();
                }
                let base = r2.0 * DPL;
                let top = r2.1 * DPL;
                let mut seeds: Vec<usize> = Vec::new();
                if base > 0 && unlikely(s2_carry_into(src, l, l1, base)) {
                    seeds.push(base);
                }
                for &(p, _) in own {
                    if p >= base && p < top {
                        seeds.push(p);
                    }
                }
                for &(p, _) in other {
                    let d = l1 - 1 - p;
                    if d >= base && d < top {
                        seeds.push(d);
                    }
                }
                if likely(seeds.is_empty()) {
                    return Vec::new();
                }
                cold_path();
                seeds.sort_unstable();
                seeds.dedup();
                let psum = |d: usize| -> u8 {
                    if d < l1 {
                        p1c(d, own) + p1c(l1 - 1 - d, other)
                    } else {
                        0
                    }
                };
                let mut out = Vec::new();
                let mut done_until = base;
                for &s0 in &seeds {
                    if s0 < done_until {
                        continue;
                    }
                    let mut pc = false;
                    for pos in (base..s0).rev() {
                        let sum = psum(pos);
                        if sum != 9 {
                            pc = sum > 9;
                            break;
                        }
                    }
                    let mut tc = s2_carry_into(src, l, l1, s0);
                    let mut d = s0;
                    loop {
                        let ps = psum(d) + u8::from(pc);
                        let ts = s2_sum(src, l, l1, d) + u8::from(tc);
                        let pd = ps % 10;
                        let td = ts % 10;
                        pc = ps >= 10;
                        tc = ts >= 10;
                        if pd != td {
                            out.push((d, pd));
                        }
                        d += 1;
                        if d >= top || (pd == td && pc == tc) {
                            break;
                        }
                    }
                    done_until = d;
                }
                out
            };
            let w2_lo = wrong2(round.r2_lo, &rip1_lo, &rip1_hi);
            let w2_hi = wrong2(round.r2_hi, &rip1_hi, &rip1_lo);
            if likely(w2_lo.is_empty() && w2_hi.is_empty()) {
                return;
            }
            cold_path();
            any_wrong2 = true;

            // wrong position d reaches output slot d through the forward
            // stream and slot l2 - 1 - d through the backward
            for (chunk, fwd, bwd, hi_side) in [
                (round.lo, &w2_lo, &w2_hi, false),
                (round.hi, &w2_hi, &w2_lo, true),
            ] {
                if chunk.0 >= chunk.1 {
                    continue;
                }
                let mut affected = Vec::new();
                let mut mark = |d: usize| {
                    let line = d / DPL;
                    if line >= chunk.0 && line < chunk.1 {
                        affected.push(line);
                    }
                };
                for &(d, _) in fwd {
                    mark(d);
                }
                for &(d, _) in bwd {
                    if d < l2 {
                        mark(l2 - 1 - d);
                    }
                }
                if affected.is_empty() {
                    continue;
                }
                affected.sort_unstable();
                affected.dedup();
                let (spec_base, cap, slot) = if hi_side {
                    (chunk.0, chunk.1, (hi_j - num_threads) * hi_stride + ord)
                } else {
                    (lo_b.0, lo_b.1, lo_j)
                };
                consumers.push(Consumer3 {
                    fwd_w2: fwd.clone(),
                    bwd_w2: bwd.clone(),
                    spec_base,
                    cap,
                    slot,
                    hi_side,
                    affected,
                });
            }
        });

        if unlikely(any_rip1) {
            self.rip1_seen.store(true, Relaxed);
        }
        if unlikely(any_wrong2) {
            self.wrong2_seen.store(true, Relaxed);
        }
        if likely(consumers.is_empty()) {
            return;
        }
        cold_path();

        // Pass 2: recompute each affected line and its convergence tail.
        for con in &consumers {
            // What the pass consumed at position `p` through the given
            // stream: the production's wrong value where the walk found
            // one, ground truth elsewhere.
            let consumed = |p: usize, w2: &[(usize, u8)]| -> u8 {
                for &(wp, val) in w2 {
                    if wp == p {
                        return val;
                    }
                }
                s2_digit(src, l, l1, p)
            };
            let consumed_sum = |s: usize| -> u8 {
                let fwd = consumed(s, &con.fwd_w2);
                if s < l2 {
                    fwd + consumed(l2 - 1 - s, &con.bwd_w2)
                } else {
                    fwd
                }
            };

            // True digits are evaluated a line-sized window at a time with
            // one carry-into walk and a chained carry, instead of a fresh
            // backward walk per digit: a naive per-digit chain costs the
            // repair over a millisecond per pass, which is most of the
            // triple pass's non-DRAM budget.
            //
            // True first-step digits for positions base..base+DPL.
            let s1_window = |base: isize| -> [u8; DPL] {
                let mut out = [0u8; DPL];
                let mut carry = base > 0 && s1_carry_into(src, l, base.cast_unsigned());
                for (p, slot) in out.iter_mut().enumerate() {
                    let pos = base + p.cast_signed();
                    if pos < 0 {
                        continue;
                    }
                    let total = s1_sum(src, l, pos.cast_unsigned()) + u8::from(carry);
                    *slot = total % 10;
                    carry = total >= 10;
                }
                out
            };
            // True second-step digits for positions base..base+DPL: the
            // forward window and the ascending window of its mirrors.
            let s2_window = |base: isize| -> [u8; DPL] {
                let fwd_win = s1_window(base);
                let bwd_win = s1_window(l1.cast_signed() - base - DPL.cast_signed());
                let mut out = [0u8; DPL];
                let mut carry = base > 0 && s2_carry_into(src, l, l1, base.cast_unsigned());
                for (p, slot) in out.iter_mut().enumerate() {
                    let pos = base + p.cast_signed();
                    if pos < 0 {
                        continue;
                    }
                    let sum = if pos.cast_unsigned() < l1 {
                        fwd_win[p] + bwd_win[DPL - 1 - p]
                    } else {
                        0
                    };
                    let total = sum + u8::from(carry);
                    *slot = total % 10;
                    carry = total >= 10;
                }
                out
            };

            for &first_line in &con.affected {
                // the chain carry the pass had entering the first affected
                // line: walk the consumed sums down to the chunk's
                // speculation base
                let mut carry = false;
                for d in (con.spec_base * DPL..first_line * DPL).rev() {
                    let sum = consumed_sum(d);
                    if sum != 9 {
                        carry = sum > 9;
                        break;
                    }
                }

                let mut line_idx = first_line;
                loop {
                    let fwd_win = s2_window((line_idx * DPL).cast_signed());
                    let bwd_win =
                        s2_window(l2.cast_signed() - ((line_idx + 1) * DPL).cast_signed());
                    let mut lo_plane = [0u8; LV_LEN];
                    let mut hi_plane = [0u8; LV_LEN];
                    for p in 0..DPL {
                        let sum = if line_idx * DPL + p < l2 {
                            fwd_win[p] + bwd_win[DPL - 1 - p]
                        } else {
                            0
                        };
                        let total = sum + u8::from(carry);
                        let digit = total % 10;
                        carry = total >= 10;
                        if p < LV_LEN {
                            lo_plane[p] = digit;
                        } else {
                            hi_plane[p - LV_LEN] = digit;
                        }
                    }
                    let line =
                        pack_line(LimbVec::from_array(lo_plane), LimbVec::from_array(hi_plane));
                    if line == unsafe { (*dst.add(line_idx)).0 } {
                        break; // chain and inputs agree with the pass again
                    }
                    unsafe { *dst.add(line_idx) = Limb(line) };
                    line_idx += 1;
                    if line_idx >= con.cap {
                        // the correction changed the chunk's carry-out;
                        // resolve_carries propagates it from here
                        if con.hi_side {
                            unsafe { (*chunk_carry.add(con.slot)).store(carry, Relaxed) };
                        } else {
                            self.block_carry[con.slot].0.store(carry, Relaxed);
                        }
                        break;
                    }
                }
            }
        }
    }
}

/// Scalar ground truth for the fused two-iteration step's repair path. All
/// of these read only the immutable input buffer `a` (`l` digits), so they
/// are exact regardless of what the speculative pass produced. Positions at
/// and above the top digit read as zero sums, so indices may run into the
/// padding.
///
/// The digit sum feeding slot `d` of the first step's output.
#[inline]
fn s1_sum(a: &[Limb], l: usize, d: usize) -> u8 {
    if d < l {
        digit_at(a, d) + digit_at(a, l - 1 - d)
    } else {
        0
    }
}

/// The true carry into slot `d` of the first step's output: walking down
/// from `d`, sums of nine propagate whatever comes from below and the first
/// other sum decides. The walk is expected O(1) and bounded by slot 0,
/// whose carry-in is zero.
fn s1_carry_into(a: &[Limb], l: usize, d: usize) -> bool {
    for j in (0..d).rev() {
        let s = s1_sum(a, l, j);
        if s != 9 {
            return s > 9;
        }
    }
    false
}

/// The true digit at slot `d` of the first step's output, `d <= l`.
#[inline]
fn s1_digit(a: &[Limb], l: usize, d: usize) -> u8 {
    (s1_sum(a, l, d) + u8::from(s1_carry_into(a, l, d))) % 10
}

/// The digit sum feeding slot `d` of the second step's output, from true
/// first-step digits. `l1` is the first step's exact digit count.
#[inline]
fn s2_sum(a: &[Limb], l: usize, l1: usize, d: usize) -> u8 {
    if d < l1 {
        s1_digit(a, l, d) + s1_digit(a, l, l1 - 1 - d)
    } else {
        0
    }
}

/// The true carry into slot `d` of the second step's output, by the same
/// backward walk as `s1_carry_into` one level up.
fn s2_carry_into(a: &[Limb], l: usize, l1: usize, d: usize) -> bool {
    for j in (0..d).rev() {
        let s = s2_sum(a, l, l1, j);
        if s != 9 {
            return s > 9;
        }
    }
    false
}

/// The true digit at slot `d` of the second step's output, `d <= l1`.
#[inline]
fn s2_digit(a: &[Limb], l: usize, l1: usize, d: usize) -> u8 {
    (s2_sum(a, l, l1, d) + u8::from(s2_carry_into(a, l, l1, d))) % 10
}

/// The digit sum feeding slot `d` of the third step's output, from true
/// second-step digits. `l2` is the second step's exact digit count.
#[inline]
fn s3_sum(a: &[Limb], l: usize, l1: usize, l2: usize, d: usize) -> u8 {
    if d < l2 {
        s2_digit(a, l, l1, d) + s2_digit(a, l, l1, l2 - 1 - d)
    } else {
        0
    }
}

/// Whether any digit of the true second step carries: some digit sum
/// reaches ten (propagation alone cannot start a carry). The sums are
/// mirror-symmetric, so half the range decides; the scan exits at the
/// first generator, which sits within a few digits of the bottom for
/// anything but a near-palindromic step.
fn s2_generates(a: &[Limb], l: usize, l1: usize) -> bool {
    (0..l1.div_ceil(2)).any(|d| s2_sum(a, l, l1, d) >= 10)
}

/// `s2_generates` one level up: whether any digit of the true third step
/// carries.
fn s3_generates(a: &[Limb], l: usize, l1: usize, l2: usize) -> bool {
    (0..l2.div_ceil(2)).any(|d| s3_sum(a, l, l1, l2, d) >= 10)
}

/// Whether the second reverse-and-add of a fused pass gains a digit,
/// decided exactly before the pass from the input buffer alone: the same
/// descending scan as `prescan_grow`, over true first-step digits.
fn prescan_grow2(a: &[Limb], l: usize, l1: usize) -> bool {
    for d in (0..l1).rev() {
        let s = s2_sum(a, l, l1, d);
        if s != 9 {
            return s > 9;
        }
    }
    false
}

/// One chunk pair of the interleaved walk: the two output chunks of the
/// round and, for the fused pass, the two first-step ranges that feed them.
/// `r_lo` covers the low chunk's forward reads and the high chunk's
/// backward window; `r_hi` the reverse. All ranges are in output line
/// space and may be empty.
#[derive(Clone, Copy)]
struct Round {
    lo: (usize, usize),
    hi: (usize, usize),
    r_lo: (usize, usize),
    r_hi: (usize, usize),
}

/// Lines per fused scratch buffer: a round's ranges exceed a chunk only by
/// the block-bound rounding and, for the triple pass's first-step ranges,
/// one more level of mirror slack.
const SCRATCH_LINES: usize = CHUNK_LINES + 16;

/// Per-participant scratch of the fused passes: the two first-step ranges a
/// round consumes, and for the triple pass the two second-step ranges
/// derived from them. Heap-allocated once per worker so the chunk size is
/// not limited by thread stacks.
struct FusedScratch {
    lo: Box<[Limb]>,
    hi: Box<[Limb]>,
    lo2: Box<[Limb]>,
    hi2: Box<[Limb]>,
}

impl FusedScratch {
    fn new() -> Self {
        Self {
            lo: vec![Limb::new(); SCRATCH_LINES].into_boxed_slice(),
            hi: vec![Limb::new(); SCRATCH_LINES].into_boxed_slice(),
            lo2: vec![Limb::new(); SCRATCH_LINES].into_boxed_slice(),
            hi2: vec![Limb::new(); SCRATCH_LINES].into_boxed_slice(),
        }
    }
}

/// Slot capacity per high block for the fused pass's ordinal chunk-carry
/// indexing: a block's contiguous chunk cover has at most this many
/// chunks. The fused pass's high chunks sit at mirrored, non-grid bases,
/// so their carries are indexed by order of generation instead of by
/// position.
#[inline]
const fn fused_hi_stride(q1_lines: usize, num_blocks: usize) -> usize {
    q1_lines / num_blocks / CHUNK_LINES + 3
}

/// The union of two intervals, either possibly empty; any gap between them
/// is included.
#[inline]
const fn interval_union(a: (usize, usize), b: (usize, usize)) -> (usize, usize) {
    if a.0 >= a.1 {
        b
    } else if b.0 >= b.1 {
        a
    } else {
        (a.0.min(b.0), a.1.max(b.1))
    }
}

/// The chunk-carry slot of the high chunk starting at line `x` of block
/// `j` in the single-step pass, whose high chunks sit on the absolute
/// line grid. Grid starts land in distinct cells, a block's bottom partial
/// chunk lands one cell below its neighbor, and the block index breaks
/// ties across block boundaries.
#[inline]
const fn chunk_slot(j: usize, x: usize) -> usize {
    j + x / CHUNK_LINES + 1
}

/// Replays the chunk walk of one block pair, calling `f` once per round.
/// The engine pass, the carry resolution, and the repair scan all derive
/// the rounds from this, so their views of the speculative ranges agree
/// exactly. `q1` is the second step's mirror line (`l1 / DPL`) and
/// `q1_lines` the first step's output line count; the backward window of
/// output line `k` spans lines `q1 - 1 - k ..= q1 - k`.
///
/// Low chunks walk the absolute line grid; each round's high chunk is the
/// mirror image of its low chunk (base `q1 - lo.1`, clipped to the block
/// and to what remains), so the pair's two scratch ranges -- each the
/// union of a chunk and the mirror of its partner -- coincide to within
/// the block-bound rounding rather than drifting by up to a whole chunk.
/// A high block that outlives its partner falls back to the plain grid.
fn for_each_round(
    (lo_s, lo_e): (usize, usize),
    (hi_s, hi_e): (usize, usize),
    q1: usize,
    q1_lines: usize,
    mut f: impl FnMut(Round),
) {
    let bwd_of = |(s, e): (usize, usize)| -> (usize, usize) {
        if s >= e {
            return (0, 0);
        }
        let lo = (q1.cast_signed() - e.cast_signed()).max(0).cast_unsigned();
        let hi = (q1.cast_signed() - s.cast_signed() + 1)
            .max(0)
            .cast_unsigned()
            .min(q1_lines);
        (lo, hi)
    };

    let mut lo_c = lo_s;
    let mut hi_c = hi_e;
    while lo_c < lo_e || hi_c > hi_s {
        let lo = if lo_c < lo_e {
            (lo_c, ((lo_c / CHUNK_LINES + 1) * CHUNK_LINES).min(lo_e))
        } else {
            (lo_c, lo_c)
        };
        let hi = if hi_c > hi_s {
            let base = if lo.0 < lo.1 {
                let m = (q1.cast_signed() - lo.1.cast_signed())
                    .max(hi_s.cast_signed())
                    .cast_unsigned();
                m.min(hi_c)
            } else {
                (hi_c.saturating_sub(1) / CHUNK_LINES * CHUNK_LINES).max(hi_s)
            };
            (base, hi_c)
        } else {
            (hi_c, hi_c)
        };
        f(Round {
            lo,
            hi,
            r_lo: interval_union((lo.0, lo.1), bwd_of(hi)),
            r_hi: interval_union((hi.0, hi.1), bwd_of(lo)),
        });
        lo_c = lo.1;
        hi_c = hi.0;
    }
}

/// One chunk pair of the triple pass's walk: the round's two output
/// chunks, the two second-step ranges that feed them, and the two
/// first-step ranges that feed those. All in output line space of their
/// respective levels; any range may be empty.
#[derive(Clone, Copy)]
struct Round3 {
    lo: (usize, usize),
    hi: (usize, usize),
    r2_lo: (usize, usize),
    r2_hi: (usize, usize),
    r1_lo: (usize, usize),
    r1_hi: (usize, usize),
}

/// `for_each_round` for the triple pass: the chunk pairing and the
/// second-step ranges mirror through `q2` (the third step's mirror line),
/// and each first-step range is its second-step range extended by the
/// mirror through `q1` of the partner's -- the lines phase B's backward
/// stream reads from the other scratch buffer.
fn for_each_round3(
    lo_b: (usize, usize),
    hi_b: (usize, usize),
    q2: usize,
    q2_lines: usize,
    q1: usize,
    q1_lines: usize,
    mut f: impl FnMut(Round3),
) {
    let bwd1_of = |(s, e): (usize, usize)| -> (usize, usize) {
        if s >= e {
            return (0, 0);
        }
        let lo = (q1.cast_signed() - e.cast_signed()).max(0).cast_unsigned();
        let hi = (q1.cast_signed() - s.cast_signed() + 1)
            .max(0)
            .cast_unsigned()
            .min(q1_lines);
        (lo, hi)
    };
    for_each_round(lo_b, hi_b, q2, q2_lines, |round| {
        f(Round3 {
            lo: round.lo,
            hi: round.hi,
            r2_lo: round.r_lo,
            r2_hi: round.r_hi,
            r1_lo: interval_union(round.r_lo, bwd1_of(round.r_hi)),
            r1_hi: interval_union(round.r_hi, bwd1_of(round.r_lo)),
        });
    });
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
    /// Backing store of `SharedPacked::chunk_carry`, regrown by the
    /// coordinator between passes when the number outgrows it.
    chunk_carries: Box<[AtomicBool]>,
    /// Participant 0's fused-pass scratch.
    scratch: FusedScratch,
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
                    let mut scratch = FusedScratch::new();
                    loop {
                        shared.barrier.wait();
                        if unlikely(shared.stop.load(Relaxed)) {
                            break;
                        }
                        shared.run_blocks(t, &mut scratch);
                    }
                })
            })
            .collect();

        Self {
            shared,
            chunk_carries: Box::new([]),
            scratch: FusedScratch::new(),
            handles,
        }
    }

    /// One packed reverse-and-add step. Returns whether any digit carried.
    pub fn step<T: Allocator + Clone + Copy>(&mut self, x: &mut PackedInt<T>) -> bool {
        let shared = &*self.shared;
        let num_blocks = shared.num_threads * 2;

        let digits = x.digits;
        let grew = x.prescan_grow();
        let out_digits = digits + usize::from(grew);
        let lines = digits.div_ceil(DPL);
        let lines_out = out_digits.div_ceil(DPL);

        // every chunk-carry slot a high block can address this pass
        let slots = num_blocks + lines / CHUNK_LINES + 2;
        if unlikely(self.chunk_carries.len() < slots) {
            cold_path();
            self.chunk_carries = (0..slots.next_power_of_two())
                .map(|_| AtomicBool::new(false))
                .collect();
        }
        shared
            .chunk_carry
            .store(self.chunk_carries.as_ptr().cast_mut(), Relaxed);

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
        shared.run_blocks(0, &mut self.scratch);

        let carry = self.resolve_carries(&mut x.a[next], out_digits);

        if unlikely(carry) {
            // only reachable when the input's top line was full to the brim
            cold_path();
            debug_assert!(grew && digits.is_multiple_of(DPL));
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

    /// Serial carry resolution across the single-step pass's speculative
    /// ranges in ascending line order -- the low blocks whole, the high
    /// blocks chunk by chunk on the line grid -- where a range whose true
    /// carry-in turned out to be one gets a decimal increment at its base,
    /// capped at `out_digits`. Returns whether a carry escaped the top
    /// range.
    fn resolve_carries(&self, dst: &mut [Limb], out_digits: usize) -> bool {
        let shared = &*self.shared;
        let num_blocks = shared.num_threads * 2;
        let mut resolve = |carry: bool, start: usize, end: usize, carry_out: bool| {
            let mut carry_out = carry_out;
            if unlikely(carry) {
                cold_path();
                let (_, escaped) = increment_digits(dst, start * DPL, (end * DPL).min(out_digits));
                carry_out |= escaped;
            }
            carry_out
        };
        let mut carry = false;
        for j in 0..num_blocks {
            let start = shared.bounds[j].load(Relaxed);
            let end = shared.bounds[j + 1].load(Relaxed);
            if start >= end {
                continue; // an empty block passes the carry through
            }
            if j < shared.num_threads {
                let carry_out = shared.block_carry[j].0.load(Relaxed);
                carry = resolve(carry, start, end, carry_out);
            } else {
                let mut c = start;
                while c < end {
                    let c_end = ((c / CHUNK_LINES + 1) * CHUNK_LINES).min(end);
                    let carry_out = self.chunk_carries[chunk_slot(j, c)].load(Relaxed);
                    carry = resolve(carry, c, c_end, carry_out);
                    c = c_end;
                }
            }
        }
        carry
    }

    /// Carry resolution for the fused pass. Low blocks resolve whole; a
    /// high block's chunk partition sits at mirrored, non-grid bases, so
    /// its thread's round walk is replayed to recover the chunks and their
    /// ordinal carry slots, which then resolve in ascending line order.
    fn resolve_carries_fused(
        &self,
        dst: &mut [Limb],
        out_digits: usize,
        q1: usize,
        q1_lines: usize,
    ) -> bool {
        let shared = &*self.shared;
        let num_threads = shared.num_threads;
        let num_blocks = num_threads * 2;
        let hi_stride = fused_hi_stride(q1_lines, num_blocks);
        let bound = |j: usize| shared.bounds[j].load(Relaxed);
        let mut resolve = |carry: bool, start: usize, end: usize, carry_out: bool| {
            let mut carry_out = carry_out;
            if unlikely(carry) {
                cold_path();
                let (_, escaped) = increment_digits(dst, start * DPL, (end * DPL).min(out_digits));
                carry_out |= escaped;
            }
            carry_out
        };
        let mut carry = false;
        for j in 0..num_blocks {
            let start = bound(j);
            let end = bound(j + 1);
            if start >= end {
                continue; // an empty block passes the carry through
            }
            if j < num_threads {
                carry = resolve(carry, start, end, shared.block_carry[j].0.load(Relaxed));
            } else {
                let t = num_blocks - 1 - j;
                let mut chunks: Vec<(usize, usize, usize)> = Vec::new();
                let mut ord = 0usize;
                for_each_round(
                    (bound(t), bound(t + 1)),
                    (start, end),
                    q1,
                    q1_lines,
                    |round| {
                        if round.hi.0 < round.hi.1 {
                            chunks.push((round.hi.0, round.hi.1, ord));
                            ord += 1;
                        }
                    },
                );
                for &(s, e, o) in chunks.iter().rev() {
                    let slot = (j - num_threads) * hi_stride + o;
                    carry = resolve(carry, s, e, self.chunk_carries[slot].load(Relaxed));
                }
            }
        }
        carry
    }

    /// One fused double step: two reverse-and-add iterations with one read
    /// of the current buffer and one write of the other, the intermediate
    /// value living only in per-thread scratch. Returns whether any digit
    /// carried in the second iteration (which gates the caller's palindrome
    /// check of the materialized result) and whether the unmaterialized
    /// intermediate value was a palindrome.
    pub fn step2<T: Allocator + Clone + Copy>(&mut self, x: &mut PackedInt<T>) -> Step2Result {
        let shared = &*self.shared;
        let num_blocks = shared.num_threads * 2;

        let l = x.digits;
        let grew = x.prescan_grow();
        let l1 = l + usize::from(grew);
        let lines = l.div_ceil(DPL);
        let q1 = l1 / DPL;
        let q1_lines = l1.div_ceil(DPL);
        // covers slot l1, where the second step's growth lands
        let lines_out = (l1 + 1).div_ceil(DPL);

        let next = 1 - x.cur;
        x.a[next].resize(lines_out, Limb::new());

        let slots = shared.num_threads * fused_hi_stride(q1_lines, num_blocks);
        if unlikely(self.chunk_carries.len() < slots) {
            cold_path();
            self.chunk_carries = (0..slots.next_power_of_two())
                .map(|_| AtomicBool::new(false))
                .collect();
        }
        shared
            .chunk_carry
            .store(self.chunk_carries.as_ptr().cast_mut(), Relaxed);

        shared.a_src.store(x.a[x.cur].as_ptr().cast_mut(), Relaxed);
        shared.a_dst.store(x.a[next].as_mut_ptr(), Relaxed);
        shared.lines.store(lines, Relaxed);
        shared.digits.store(l, Relaxed);
        shared.digits1.store(l1, Relaxed);
        shared.fused.store(true, Relaxed);
        shared.ever_carried.store(false, Relaxed);
        shared.carried1.store(false, Relaxed);
        shared.pal1.store(true, Relaxed);
        for k in 0..=num_blocks {
            shared.bounds[k].store(k * q1_lines / num_blocks, Relaxed);
        }

        shared.barrier.wait();
        shared.run_blocks(0, &mut self.scratch);
        shared.fused.store(false, Relaxed);

        self.repair_fused(x, l, l1, q1, q1_lines);

        let carry = self.resolve_carries_fused(&mut x.a[next], l1 + 1, q1, q1_lines);
        if unlikely(carry) {
            // only reachable when the first step's top line was full to the
            // brim; the escaping carry is the second step's growth digit
            cold_path();
            debug_assert!(l1.is_multiple_of(DPL));
            set_digit(&mut x.a[next], l1, 1);
        }

        let l2 = l1 + usize::from(digit_at(&x.a[next], l1) != 0);
        x.digits = l2;
        x.cur = next;

        debug_assert!(digit_at(&x.a[next], l2 - 1) != 0, "fused step lost growth");

        let carried1 = shared.carried1.load(Relaxed);
        Step2Result {
            carried: shared.ever_carried.load(Relaxed),
            palindrome_mid: !carried1 && shared.pal1.load(Relaxed),
        }
    }

    /// Repairs the fused pass's first-step misspeculation: replays the
    /// round walk, finds every scratch range whose speculative carry-in of
    /// zero was wrong, enumerates the digits its increment would have
    /// changed (the consumed values the pass fed into the second step), and
    /// recomputes the second-step output lines that consumed them -- the
    /// line of each changed position and the line of its mirror -- from
    /// exact first-step ground truth, walking upward until the recomputed
    /// line matches what the pass stored. A walk that reaches its chunk's
    /// speculation boundary without converging instead corrects the
    /// recorded carry-out, and `resolve_carries` propagates from there.
    fn repair_fused<T: Allocator + Clone + Copy>(
        &self,
        x: &mut PackedInt<T>,
        l: usize,
        l1: usize,
        q1: usize,
        q1_lines: usize,
    ) {
        let shared = &*self.shared;
        let num_threads = shared.num_threads;
        let num_blocks = num_threads * 2;
        let bound = |j: usize| shared.bounds[j].load(Relaxed);

        let [b0, b1] = &mut x.a;
        let (src, dst) = if x.cur == 0 { (&*b0, b1) } else { (&*b1, b0) };

        // The digits a wrong carry-in at `base_line` changed, with the
        // (wrong) values the pass consumed: the run of nines the increment
        // turns to zeros plus the digit it lands on, all under the range's
        // speculative chain. A ripple that reaches the range's top is
        // truncated there: consumers beyond it read other ranges.
        let ripple_of = |(base_line, end_line): (usize, usize)| -> Vec<(usize, u8)> {
            let base = base_line * DPL;
            if base_line == 0 || base_line >= end_line || base >= l1 {
                return Vec::new();
            }
            if likely(!s1_carry_into(src, l, base)) {
                return Vec::new();
            }
            cold_path();
            let top = (end_line * DPL).min(l1);
            let mut out = Vec::new();
            let mut carry = 0u8;
            for p in base..top {
                let sum = s1_sum(src, l, p) + carry;
                let spec = sum % 10;
                carry = u8::from(sum >= 10);
                out.push((p, spec));
                if spec != 9 {
                    break;
                }
            }
            out
        };

        // Pass 1: ripples per round, kept for consumed-value lookups, and
        // the affected second-step output lines with their consumer chunks.
        struct Consumer {
            fwd_rips: Vec<(usize, u8)>,
            bwd_rips: Vec<(usize, u8)>,
            /// Speculation base of the chunk's carry chain: the chunk base
            /// for high chunks, the block base for low chunks.
            spec_base: usize,
            /// The chain's speculation boundary above: the chunk end for
            /// high chunks, the block end for low chunks.
            cap: usize,
            /// Recorded-carry slot to correct if a walk reaches `cap`:
            /// chunk slot for high chunks, block index for low blocks.
            slot: usize,
            hi_side: bool,
            affected: Vec<usize>,
        }
        let mut consumers: Vec<Consumer> = Vec::new();
        let mut any_rip = false;

        let hi_stride = fused_hi_stride(q1_lines, num_blocks);
        for t in 0..num_threads {
            let lo_j = t;
            let hi_j = num_blocks - 1 - t;
            let lo_b = (bound(lo_j), bound(lo_j + 1));
            let hi_b = (bound(hi_j), bound(hi_j + 1));
            let mut hi_ord = 0usize;
            for_each_round(lo_b, hi_b, q1, q1_lines, |round| {
                let ord = hi_ord;
                if round.hi.0 < round.hi.1 {
                    hi_ord += 1;
                }
                let rip_lo = ripple_of(round.r_lo);
                let rip_hi = ripple_of(round.r_hi);
                if likely(rip_lo.is_empty() && rip_hi.is_empty()) {
                    return;
                }
                cold_path();
                any_rip = true;
                // consumed position p reaches output slot p through the
                // forward stream and slot l1 - 1 - p through the backward
                for (chunk, fwd, bwd, hi_side) in [
                    (round.lo, &rip_lo, &rip_hi, false),
                    (round.hi, &rip_hi, &rip_lo, true),
                ] {
                    if chunk.0 >= chunk.1 {
                        continue;
                    }
                    let mut affected = Vec::new();
                    let mut mark = |d: usize| {
                        let line = d / DPL;
                        if line >= chunk.0 && line < chunk.1 {
                            affected.push(line);
                        }
                    };
                    for &(p, _) in fwd {
                        mark(p);
                    }
                    for &(p, _) in bwd {
                        debug_assert!(p < l1, "first-step ripple escaped the exact length");
                        mark(l1 - 1 - p);
                    }
                    if affected.is_empty() {
                        continue;
                    }
                    affected.sort_unstable();
                    affected.dedup();
                    let (spec_base, cap, slot) = if hi_side {
                        (chunk.0, chunk.1, (hi_j - num_threads) * hi_stride + ord)
                    } else {
                        (lo_b.0, lo_b.1, lo_j)
                    };
                    consumers.push(Consumer {
                        fwd_rips: fwd.clone(),
                        bwd_rips: bwd.clone(),
                        spec_base,
                        cap,
                        slot,
                        hi_side,
                        affected,
                    });
                }
            });
        }

        if unlikely(any_rip) {
            // Wrong consumed digits make the pass's second-step sums -- and
            // with them the accumulated carry flag -- untrustworthy in both
            // directions, and a spuriously TRUE flag would silence the
            // caller's palindrome check on the repaired value. The exact
            // flag is whether any true sum generates, which the scan below
            // decides within a few digits unless the step really is
            // carry-free.
            shared.ever_carried.store(s2_generates(src, l, l1), Relaxed);
        }

        if likely(consumers.is_empty()) {
            return;
        }
        cold_path();

        // Pass 2: recompute each affected line and its convergence tail.
        for con in &consumers {
            // What the pass consumed at position `p` through the given
            // stream: the range's speculative value where the ripple
            // touched it, ground truth elsewhere.
            let consumed = |p: usize, rips: &[(usize, u8)]| -> u8 {
                for &(rp, spec) in rips {
                    if rp == p {
                        return spec;
                    }
                }
                s1_digit(src, l, p)
            };
            let consumed_sum = |d: usize| -> u8 {
                if d < l1 {
                    consumed(d, &con.fwd_rips) + consumed(l1 - 1 - d, &con.bwd_rips)
                } else {
                    0
                }
            };

            for &k0 in &con.affected {
                // the chain carry the pass had entering line k0: walk the
                // consumed sums down to the chunk's speculation base
                let mut carry = false;
                for d in (con.spec_base * DPL..k0 * DPL).rev() {
                    let s = consumed_sum(d);
                    if s != 9 {
                        carry = s > 9;
                        break;
                    }
                }

                let mut k = k0;
                loop {
                    let mut lo_plane = [0u8; LV_LEN];
                    let mut hi_plane = [0u8; LV_LEN];
                    for p in 0..DPL {
                        let s = s2_sum(src, l, l1, k * DPL + p) + u8::from(carry);
                        let digit = s % 10;
                        carry = s >= 10;
                        if p < LV_LEN {
                            lo_plane[p] = digit;
                        } else {
                            hi_plane[p - LV_LEN] = digit;
                        }
                    }
                    let line =
                        pack_line(LimbVec::from_array(lo_plane), LimbVec::from_array(hi_plane));
                    if line == dst[k].0 {
                        break; // chain and inputs agree with the pass again
                    }
                    dst[k] = Limb(line);
                    k += 1;
                    if k >= con.cap {
                        // the correction changed the chunk's carry-out;
                        // resolve_carries propagates it from here
                        if con.hi_side {
                            self.chunk_carries[con.slot].store(carry, Relaxed);
                        } else {
                            shared.block_carry[con.slot].0.store(carry, Relaxed);
                        }
                        break;
                    }
                }
            }
        }
    }

    /// One fused triple step: three reverse-and-add iterations with one
    /// read of the current buffer and one write of the other, both
    /// intermediate values living only in per-thread scratch. Returns
    /// whether any digit carried in the third iteration and whether either
    /// unmaterialized intermediate value was a palindrome.
    pub fn step3<T: Allocator + Clone + Copy>(&mut self, x: &mut PackedInt<T>) -> Step3Result {
        let shared = &*self.shared;
        let num_blocks = shared.num_threads * 2;

        let l = x.digits;
        let grew = x.prescan_grow();
        let l1 = l + usize::from(grew);
        let grew2 = prescan_grow2(&x.a[x.cur], l, l1);
        let l2 = l1 + usize::from(grew2);
        let lines = l.div_ceil(DPL);
        let q2 = l2 / DPL;
        let q2_lines = l2.div_ceil(DPL);
        // covers slot l2, where the third step's growth lands
        let lines_out = (l2 + 1).div_ceil(DPL);

        let next = 1 - x.cur;
        x.a[next].resize(lines_out, Limb::new());

        let slots = shared.num_threads * fused_hi_stride(q2_lines, num_blocks);
        if unlikely(self.chunk_carries.len() < slots) {
            cold_path();
            self.chunk_carries = (0..slots.next_power_of_two())
                .map(|_| AtomicBool::new(false))
                .collect();
        }
        shared
            .chunk_carry
            .store(self.chunk_carries.as_ptr().cast_mut(), Relaxed);

        shared.a_src.store(x.a[x.cur].as_ptr().cast_mut(), Relaxed);
        shared.a_dst.store(x.a[next].as_mut_ptr(), Relaxed);
        shared.lines.store(lines, Relaxed);
        shared.digits.store(l, Relaxed);
        shared.digits1.store(l1, Relaxed);
        shared.digits2.store(l2, Relaxed);
        shared.fused3.store(true, Relaxed);
        shared.ever_carried.store(false, Relaxed);
        shared.carried1.store(false, Relaxed);
        shared.carried2.store(false, Relaxed);
        shared.pal1.store(true, Relaxed);
        shared.pal2.store(true, Relaxed);
        shared.rip1_seen.store(false, Relaxed);
        shared.wrong2_seen.store(false, Relaxed);
        for k in 0..=num_blocks {
            shared.bounds[k].store(k * q2_lines / num_blocks, Relaxed);
        }

        shared.barrier.wait();
        shared.run_blocks(0, &mut self.scratch);
        shared.fused3.store(false, Relaxed);

        let repaired = self.exact_flags3(x, l, l1, l2);

        let carry = self.resolve_carries_fused(&mut x.a[next], l2 + 1, q2, q2_lines);
        if unlikely(carry) {
            // only reachable when the second step's top line was full to the
            // brim; the escaping carry is the third step's growth digit
            cold_path();
            debug_assert!(l2.is_multiple_of(DPL));
            set_digit(&mut x.a[next], l2, 1);
        }

        let l3 = l2 + usize::from(digit_at(&x.a[next], l2) != 0);
        x.digits = l3;
        x.cur = next;

        debug_assert!(digit_at(&x.a[next], l3 - 1) != 0, "fused step lost growth");

        let carried1 = shared.carried1.load(Relaxed);
        Step3Result {
            carried: repaired.carried3,
            palindrome_mid1: !carried1 && shared.pal1.load(Relaxed),
            palindrome_mid2: !repaired.carried2
                && (!repaired.pal2_trusted || shared.pal2.load(Relaxed)),
        }
    }

    /// Exact post-pass flags of the triple step. The repair itself runs
    /// inside each worker (`SharedPacked::repair_pair3`); only the carry
    /// flags need the whole picture. Misspeculation leaks wrong digits
    /// into the pass's later sums, so the accumulated carry flags are not
    /// trustworthy in either direction once a worker saw it -- and a
    /// spuriously true flag would silence a palindrome check. The exact
    /// flag is whether any true sum generates, which the scans decide
    /// within a few digits unless the step really is carry-free.
    fn exact_flags3<T: Allocator + Clone + Copy>(
        &self,
        x: &PackedInt<T>,
        l: usize,
        l1: usize,
        l2: usize,
    ) -> Repair3 {
        let shared = &*self.shared;
        let src = &x.a[x.cur];
        let any_rip1 = shared.rip1_seen.load(Relaxed);
        let any_wrong2 = shared.wrong2_seen.load(Relaxed);
        Repair3 {
            carried2: if likely(!any_rip1) {
                shared.carried2.load(Relaxed)
            } else {
                s2_generates(src, l, l1)
            },
            carried3: if likely(!any_wrong2) {
                shared.ever_carried.load(Relaxed)
            } else {
                s3_generates(src, l, l1, l2)
            },
            pal2_trusted: !any_rip1,
        }
    }
}

/// The result of a fused double step.
pub struct Step2Result {
    /// Whether any digit carried in the second iteration.
    pub carried: bool,
    /// Whether the intermediate value (the first iteration's result, never
    /// materialized) was a palindrome.
    pub palindrome_mid: bool,
}

/// The result of a fused triple step.
pub struct Step3Result {
    /// Whether any digit carried in the third iteration.
    pub carried: bool,
    /// Whether the first iteration's unmaterialized result was a palindrome.
    pub palindrome_mid1: bool,
    /// Whether the second iteration's unmaterialized result was a
    /// palindrome.
    pub palindrome_mid2: bool,
}

/// Exact flags out of `repair_fused3`. A reverse-and-add result is a
/// palindrome exactly when no digit of the step carried, so the exact
/// carry flags decide the two intermediate palindrome checks; `pal2` is
/// only a cross-check and holds pass values, so it is consulted only when
/// no first-step misspeculation could have corrupted them.
struct Repair3 {
    carried2: bool,
    carried3: bool,
    pal2_trusted: bool,
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
                if target_threads == 1 { "" } else { "s" },
                start_time.elapsed().as_secs_f64()
            );
        }

        let engine = unsafe { engine.as_mut().unwrap_unchecked() };
        // A fused step materializes only its last iteration, so a fused
        // span must not step over a report iteration -- a multiple of
        // LOG_MASK, where the value has to exist (and, at 2^18 multiples,
        // be checkpointed). Triples run whenever the two skipped
        // iterations are both plain; the double and single steps realign
        // the cadence at report boundaries and serve the range tail.
        // Below the streaming threshold both buffers fit in L3, reads are
        // cache hits either way, and the fused passes' scratch round trips
        // only cost, so the single-step pass runs instead.
        let fused_ok = current.digits >= STREAM_MIN_LINES * DPL && !i.is_multiple_of(LOG_MASK);
        if likely(fused_ok && !(i + 1).is_multiple_of(LOG_MASK) && i + 3 <= range.end) {
            #[cfg(not(feature = "no-verify"))]
            let digits_in = current.digits;
            let fused = engine.step3(&mut current);
            #[cfg(not(feature = "no-verify"))]
            if unlikely(fused.palindrome_mid1 || fused.palindrome_mid2) {
                cold_path();
                // the palindrome is an unmaterialized intermediate value;
                // the input buffer is intact, so single steps rebuild it
                current.cur = 1 - current.cur;
                current.digits = digits_in;
                engine.step(&mut current);
                i += 1;
                if !fused.palindrome_mid1 {
                    engine.step(&mut current);
                    i += 1;
                }
                break;
            }
            carried = fused.carried;
            i += 2;
        } else if fused_ok && i + 2 <= range.end {
            #[cfg(not(feature = "no-verify"))]
            let digits_in = current.digits;
            let fused = engine.step2(&mut current);
            #[cfg(not(feature = "no-verify"))]
            if unlikely(fused.palindrome_mid) {
                cold_path();
                // the palindrome is the pair's unmaterialized intermediate
                // value; the input buffer is intact, so one single step
                // rebuilds it
                current.cur = 1 - current.cur;
                current.digits = digits_in;
                engine.step(&mut current);
                i += 1;
                break;
            }
            carried = fused.carried;
            i += 1;
        } else {
            carried = engine.step(&mut current);
        }

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
