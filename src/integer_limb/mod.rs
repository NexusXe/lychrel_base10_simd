#[cfg(target_arch = "x86")]
#[allow(unused_imports)]
use std::arch::x86::{__m512i, _MM_HINT_ET0, _mm_prefetch};
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::{__m512i, _MM_HINT_ET0, _mm_prefetch};

use std::alloc::{Allocator, Global as GlobalAllocator};
use std::fmt::Write;
#[cfg(not(debug_assertions))]
#[allow(unused_imports)]
use std::hint::unreachable_unchecked;
use std::hint::{cold_path, likely};
use std::intrinsics::const_eval_select;
use std::mem::transmute;

#[allow(unused_imports)]
use std::simd::{Select, prelude::*};

use zerocopy::{FromZeros, IntoBytes, KnownLayout, transmute};

mod values;
pub use values::*;

pub const LV_BYTES: usize = LV_LEN * (LimbVecScalar::BITS / 8) as usize;

pub type LimbVecScalar = u8;
pub type LimbVec = Simd<LimbVecScalar, { LV_LEN }>;

#[allow(dead_code)]
type LimbVecMask =
    <std::simd::Simd<LimbVecScalar, { LV_LEN }> as std::simd::cmp::SimdPartialEq>::Mask;

pub const WV_LEN: usize = LV_LEN / (WideVecScalar::BITS as usize / LimbVecScalar::BITS as usize);
pub(crate) type WideVec = Simd<WideVecScalar, WV_LEN>;
pub const WV_BYTES: usize = WV_LEN * (WideVecScalar::BITS / 8) as usize;

const fn assert_good_vec_sizes() {
    assert!(std::mem::size_of::<LimbVec>() == std::mem::size_of::<WideVec>());
}

const _: () = assert_good_vec_sizes();

#[cfg(all(
    not(feature = "global-alloc"),
    any(target_family = "windows", target_family = "unix")
))]
mod huge_page_alloc;

#[cfg(all(
    not(feature = "global-alloc"),
    any(target_family = "windows", target_family = "unix")
))]
#[allow(unused_imports)]
pub use huge_page_alloc::*;

#[macro_export]
macro_rules! impossible {
    () => {
        #[cfg(debug_assertions)]
        unreachable!();

        #[cfg(not(debug_assertions))]
        #[allow(unused_unsafe)]
        unsafe {
            std::hint::unreachable_unchecked()
        }
    };
    ($($arg:tt)+) => {
        #[cfg(debug_assertions)]
        unreachable!($($arg)+);

        #[cfg(not(debug_assertions))]
        #[allow(unused_unsafe)]
        unsafe {
            std::hint::unreachable_unchecked()
        }
    };
}

/// A 64-byte vector of u8, representing a single "limb" of a large integer.
///
/// Each byte represents a single digit in base 10, with the least significant digit at index 0.
/// Thus, the digits are stored in reverse order.
#[derive(Clone, Copy, FromZeros, IntoBytes, KnownLayout)]
pub struct Limb(pub LimbVec);

const impl std::cmp::PartialEq for Limb {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        #[inline]
        const fn eq_const(lhs: LimbVec, rhs: LimbVec) -> bool {
            let arr1 = lhs.to_array();
            let arr2 = rhs.to_array();
            let arr1_64b: [WideVecScalar; WV_LEN] = transmute!(arr1);
            let arr2_64b: [WideVecScalar; WV_LEN] = transmute!(arr2);
            let mut i: usize = WV_LEN;
            while i > 0 {
                if arr1_64b[i - 1] == arr2_64b[i - 1] {
                    i -= 1;
                } else {
                    return false;
                }
            }
            true
        }

        #[inline]
        fn eq_rt(lhs: LimbVec, rhs: LimbVec) -> bool {
            #[cfg(debug_assertions)]
            {
                lhs == rhs
            }

            #[cfg(not(debug_assertions))]
            unsafe {
                transmute::<LimbVec, WideVec>(lhs) == transmute::<LimbVec, WideVec>(rhs)
            }
        }

        const_eval_select((self.0, other.0), eq_const, eq_rt)
    }
}

impl std::cmp::Eq for Limb {}

#[cfg(all(target_feature = "avx512f", not(feature = "no-avx"),))]
impl From<Limb> for __m512i {
    #[inline]
    fn from(val: Limb) -> Self {
        val.0.into()
    }
}

const impl From<Limb> for LimbVec {
    #[inline]
    fn from(val: Limb) -> Self {
        val.0
    }
}

#[cfg(all(target_feature = "avx512f", not(feature = "no-avx"),))]
impl From<__m512i> for Limb {
    #[inline]
    fn from(val: __m512i) -> Self {
        Self(val.into())
    }
}

const impl From<LimbVec> for Limb {
    #[inline]
    fn from(val: LimbVec) -> Self {
        Self(val)
    }
}

impl Limb {
    #[inline]
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self(LimbVec::splat(0))
    }

    #[must_use]
    pub fn new_from_value(value: u128) -> Self {
        let input_digits = value.to_string();
        let mut digits = LimbVec::splat(0);
        for (i, c) in input_digits.chars().rev().enumerate() {
            if let Some(digit) = c.to_digit(10) {
                digits[i] = digit as LimbVecScalar;
            } else {
                impossible!("Invalid digit in input value: {c}");
            }
        }
        Self(digits)
    }

    #[inline]
    fn has_carries(&self) -> bool {
        self.0.simd_ge(LimbVec::splat(10)).any()
    }

    #[inline]
    pub(crate) fn len(&self) -> u8 {
        const ZEROS: LimbVec = LimbVec::splat(0);
        if self.0 == ZEROS {
            impossible!("Tried to get the length of an empty limb");
        }
        let eq_mask = self.0.simd_ne(ZEROS);
        let bitmask = eq_mask.to_bitmask();
        let output = unsafe { bitmask.highest_one().unwrap_unchecked() } + 1;
        output as u8
    }

    #[inline(always)]
    /// ## Safety
    /// N must be <= 8
    const unsafe fn shl_wide<const N: u8>(&self) -> Self {
        assert!(N <= 8, "Limb::shr_wide() must not be used with N > 8");

        #[inline(always)]
        fn shl_wide_rt<const N: u8>(input: LimbVec) -> LimbVec {
            unsafe {
                transmute::<WideVec, LimbVec>(
                    transmute::<LimbVec, WideVec>(input) << WideVecScalar::from(N),
                )
            }
        }

        #[inline(always)]
        const fn shl_wide_const<const N: u8>(input: LimbVec) -> LimbVec {
            let mut i: usize = 0;
            let mut output = unsafe { transmute::<LimbVec, WideVec>(input) }.to_array();
            while i < WV_LEN {
                output[i] <<= N;
                i += 1;
            }
            unsafe { transmute(WideVec::from_array(output)) }
        }

        Self(const_eval_select(
            (self.0,),
            shl_wide_const::<N>,
            shl_wide_rt::<N>,
        ))
    }

    #[allow(dead_code)]
    #[inline(always)]
    /// ## Safety
    /// N must be <= 8
    const unsafe fn shr_wide<const N: u8>(&self) -> Self {
        #[inline(always)]
        fn shr_wide_rt<const N: u8>(input: LimbVec) -> LimbVec {
            unsafe {
                transmute::<WideVec, LimbVec>(
                    transmute::<LimbVec, WideVec>(input) >> WideVecScalar::from(N),
                )
            }
        }

        #[inline(always)]
        const fn shr_wide_const<const N: u8>(input: LimbVec) -> LimbVec {
            let mut i: usize = 0;
            let mut output = unsafe { transmute::<LimbVec, WideVec>(input) }.to_array();
            while i < WV_LEN {
                output[i] >>= N;
                i += 1;
            }
            unsafe { transmute(WideVec::from_array(output)) }
        }

        assert!(N <= 8, "Limb::shr_wide() must not be used with N > 8");

        Self(const_eval_select(
            (self.0,),
            shr_wide_const::<N>,
            shr_wide_rt::<N>,
        ))
    }

    #[inline]
    fn pack(self, other: Self) -> Self {
        debug_assert!(!self.has_carries());
        debug_assert!(!other.has_carries());

        debug_assert_eq!(LimbVec::splat(0), self.0 & LimbVec::splat(0xF0));
        debug_assert_eq!(LimbVec::splat(0), other.0 & LimbVec::splat(0xF0));

        unsafe {
            let other_shifted: LimbVec = other.shl_wide::<4>().0;
            Self(self.0 ^ other_shifted)
        }
    }

    #[inline]
    fn unpack(&self) -> (Self, Self) {
        (Self((self.0 << 4) >> 4), Self(self.0 >> 4))
    }

    #[inline]
    const fn into_bytes(self) -> [LimbVecScalar; LV_LEN] {
        self.0.to_array()
    }

    #[inline]
    const fn from_bytes(input: [LimbVecScalar; LV_LEN]) -> Self {
        Self(LimbVec::from_array(input))
    }

    #[inline]
    const fn is_empty(&self) -> bool {
        self == &Self::new()
    }

    fn display_raw(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in self.0.reverse().as_array() {
            write!(f, "{i}")?;
        }
        Ok(())
    }

    #[inline(always)]
    pub unsafe fn zipper(limb_ptr: *mut LimbVec, rev_ptr: *mut LimbVec, lb: usize, ub: usize) {
        // instead of reversing into a seperate vector, reverse and pack into the original limb
        // branch like this so the smaller-than-cache variant still gets unrolled
        if lb > ub || ub == 0 {
            impossible!("Incoherent zipper lb/ub");
        }

        for i in lb..ub {
            unsafe {
                let left_limb_ptr = limb_ptr.add(i);
                let right_limb_ptr = rev_ptr.sub(i);

                // shift these as qwords since byte-wise shifts use gfni
                let lhs_output =
                    *left_limb_ptr | Self((&mut *right_limb_ptr).reverse()).shl_wide::<4>().0;
                let rhs_output =
                    *right_limb_ptr | Self((&mut *left_limb_ptr).reverse()).shl_wide::<4>().0;
                // The fallback below is the exact negation of this condition,
                // so exactly one of the two arms is always active.
                #[cfg(all(target_feature = "avx512f", feature = "stream"))]
                {
                    _mm512_stream_si512(left_limb_ptr.cast(), lhs_output.into());
                    _mm512_stream_si512(right_limb_ptr.cast(), rhs_output.into());
                }

                #[cfg(not(all(target_feature = "avx512f", feature = "stream")))]
                {
                    *left_limb_ptr = lhs_output;
                    *right_limb_ptr = rhs_output;
                }
            }
        }
    }

    #[cfg(any(test, feature = "reference-impl"))]
    #[inline(always)]
    pub(crate) fn zip_halves(limbs_ptr: *mut LimbVec, total_limbs: usize) {
        let rev_ptr = unsafe { limbs_ptr.add(total_limbs - 1) };
        unsafe { Self::zipper(limbs_ptr, rev_ptr, 0, total_limbs.div_ceil(2)) };
    }
}

const impl std::default::Default for Limb {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Add for Limb {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self::Output {
        if self.0 & LimbVec::splat(0xF0) != LimbVec::splat(0)
            || other.0 & LimbVec::splat(0xF0) != LimbVec::splat(0)
        {
            impossible!("Tried to wide add Limbs with dirty uppers");
        }

        unsafe {
            let input_64: WideVec = transmute(self.0);
            let other_64: WideVec = transmute(other.0);
            let output_64: WideVec = input_64 + other_64;
            Self(transmute::<WideVec, LimbVec>(output_64))
        }
    }
}

impl std::fmt::Display for Limb {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in self.0.reverse().to_array() {
            if i > 9 as LimbVecScalar {
                write!(f, "\x1b[31m{i}\x1b[0m")?;
            } else {
                write!(f, "{i}")?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for Limb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let digits = self.0.to_array();
        write!(f, "[")?;
        for i in &digits {
            write!(f, "{i}")?;
        }
        write!(f, "]")
    }
}

/// Resolves every decimal carry in a vector of digit sums (each 0..=18, with
/// `forward_carry` into the lowest digit), returning the resolved digits, the
/// carry out of the top digit, and whether any digit carried.
#[cfg(any(test, feature = "reference-impl"))]
#[inline(always)]
pub(crate) fn resolve_digits(sums: LimbVec, forward_carry: bool) -> (LimbVec, bool, bool) {
    const TEN_VEC_BYTES: LimbVec = LimbVec::splat(10);
    const CARRY_NINE_CMP: LimbVec = LimbVec::splat(9);
    const TOP_LANE: u64 = 1 << (LV_LEN - 1);

    // Exact carry propagation through the hardware adder.
    //
    // With generate g = (digit > 9) and propagate p = (digit == 9),
    // the carry recurrence c_i = g_i | (p_i & c_(i-1)) is the carry
    // chain of the binary sum g + (g | p): g & (g | p) == g
    // reproduces generate, and g ^ (g | p) == p reproduces
    // propagate, since a digit cannot be both greater than and equal
    // to 9. Adding those two words with the previous limb's carry as
    // carry-in therefore resolves every carry in the limb at once,
    // however long the run of nines, and the adder's carry-out is
    // the carry out of the limb. The whole limb-to-limb dependency
    // is then one `adc`.
    let generate = sums.simd_gt(CARRY_NINE_CMP).to_bitmask();
    let propagate = sums.simd_eq(CARRY_NINE_CMP).to_bitmask();
    let gp = generate | propagate;
    let (sum, adder_carry) = generate.carrying_add(gp, forward_carry);

    // sum_i = g_i ^ gp_i ^ (carry into digit i), so undoing the two
    // operands leaves the carry-in of every digit. A digit that
    // receives a carry gains 1, and a digit that emits one loses 10;
    // the emitting digits are the receiving ones shifted down a lane,
    // with the carry out of the limb occupying the top lane.
    let carry_in = sum ^ generate ^ gp;
    let carry_out = if LV_LEN == u64::BITS as usize {
        adder_carry
    } else {
        (generate | (propagate & carry_in)) & TOP_LANE != 0
    };
    let carry_out_mask = (carry_in >> 1) | (u64::from(carry_out) << (LV_LEN - 1));

    let mut output = LimbVecMask::from_bitmask(carry_out_mask).select(sums - TEN_VEC_BYTES, sums);
    output = LimbVecMask::from_bitmask(carry_in).select(output + LimbVec::splat(1), output);

    for result in output.as_array() {
        if *result > 9 {
            impossible!("Got impossible carry propagation result");
        }
    }

    (output, carry_out, carry_in != 0 || carry_out)
}

/// Adds `reversed_limb` into `limb` and resolves every decimal carry inside
/// the limb, returning the carry out of the limb. `forward_carry` is the carry
/// into the limb's lowest digit. Both operands must hold clean digits (0..=9).
#[inline(always)]
#[cfg(any(test, feature = "reference-impl"))]
pub(crate) unsafe fn add_resolve_limb(
    limb: &mut Limb,
    reversed_limb: LimbVec,
    forward_carry: bool,
    ever_carried: &mut bool,
) -> bool {
    {
        limb.0 = (limb.0 << 4) >> 4;

        if reversed_limb.simd_gt(LimbVec::splat(9)).any() {
            impossible!("Invalid digit in reversed_limb");
        }
        if limb.0.simd_gt(LimbVec::splat(9)).any() {
            impossible!("Invalid digit in limb");
        }

        // actual add done here, within the Limb struct to force quadword addition
        *limb = *limb + Limb(reversed_limb);

        for result in limb.0.as_array() {
            if *result > 18 {
                impossible!("Got impossible addition result");
            }
        }

        let (output, carry_out, carried) = resolve_digits(limb.0, forward_carry);
        if likely(carried) {
            *ever_carried = true;
        }

        // The fallback below is the exact negation of this condition,
        // so exactly one of the two arms is always active.
        #[cfg(all(target_feature = "avx512f", feature = "stream"))]
        unsafe {
            _mm512_stream_si512((&raw mut limb.0).cast::<__m512i>(), output.into());
        }

        #[cfg(not(all(target_feature = "avx512f", feature = "stream")))]
        {
            limb.0 = output;
        }

        carry_out
    }
}

/// The add pass of the fused reverse-and-add over limbs `start..end` of a
/// zipped limb vector, with a speculative carry-in of zero. Returns the carry
/// out of the block's top limb and whether any digit in the block carried.
///
/// `boundary_limb` must hold the zipped value of limb `end`: the last limb's
/// unaligned reload straddles into it, and in a threaded pass its owner may
/// have already replaced it with its own add result.
#[cfg(any(test, feature = "reference-impl"))]
pub(crate) unsafe fn add_block(
    limbs_ptr: *mut LimbVec,
    start: usize,
    end: usize,
    skip_len: usize,
    boundary_limb: LimbVec,
) -> (bool, bool) {
    use std::ptr::read_unaligned;

    if start >= end || skip_len >= LV_BYTES {
        impossible!("Incoherent add_block bounds");
    }

    let mut overflowed = false;
    let mut ever_carried = false;

    for i in start..end {
        unsafe {
            let limb = &mut *limbs_ptr.add(i).cast::<Limb>();
            let limb_vec_ptr = &raw const limb.0;

            // Write-intent prefetch 16 limbs (1KiB) ahead; the line is read,
            // resolved and stored back below. Past-the-block addresses are
            // harmless. Measured on the 822M checkpoint: 2.6% faster than no
            // prefetch; a distance of 32 or prefetching in the zipper too
            // measured slower.
            #[cfg(all(target_arch = "x86_64", not(feature = "no-prefetch")))]
            _mm_prefetch::<_MM_HINT_ET0>(limbs_ptr.add(i + 16).cast());

            let reversed_limb: LimbVec = if likely(i + 1 < end) {
                read_unaligned(limb_vec_ptr.byte_add(skip_len))
            } else {
                let pair: [LimbVec; 2] = [limb.0, boundary_limb];
                read_unaligned((&raw const pair).cast::<LimbVec>().byte_add(skip_len))
            } >> 4;

            overflowed = add_resolve_limb(limb, reversed_limb, overflowed, &mut ever_carried);
        }
    }

    (overflowed, ever_carried)
}

/// Adds one at the lowest digit of limb `start` and propagates the decimal
/// carry upward through limbs `start..end`. Returns true when the carry
/// propagates out of the whole range, which requires every digit in it to be
/// nine. Digits must already be resolved (0..=9).
#[cfg(any(test, feature = "reference-impl"))]
pub(crate) unsafe fn increment_block(limbs_ptr: *mut LimbVec, start: usize, end: usize) -> bool {
    const FULL_MASK: u64 = if LV_LEN == u64::BITS as usize {
        u64::MAX
    } else {
        (1 << LV_LEN) - 1
    };

    for i in start..end {
        let limb = unsafe { &mut *limbs_ptr.add(i).cast::<Limb>() };

        if limb.0.simd_gt(LimbVec::splat(9)).any() {
            impossible!("Unresolved digit in increment_block");
        }

        let nines = limb.0.simd_eq(LimbVec::splat(9)).to_bitmask();
        if nines == FULL_MASK {
            cold_path();
            limb.0 = LimbVec::splat(0);
            continue;
        }

        // The carry turns the run of nines at the bottom of the limb into
        // zeros and stops at the first lower digit, which gains one.
        let first_non_nine = (!nines).trailing_zeros();
        let cleared_lanes = (1u64 << first_non_nine) - 1;
        let mut output = LimbVecMask::from_bitmask(cleared_lanes).select(LimbVec::splat(0), limb.0);
        output.as_mut_array()[first_non_nine as usize] += 1;
        limb.0 = output;
        return false;
    }

    true
}

#[derive(Clone, KnownLayout)]
pub struct Integer<T: Allocator + Clone + Copy>(pub Vec<Limb, T>);

#[derive(Debug, PartialEq, Eq)]
pub struct Checkpoint {
    iteration: usize,
    pub integer: Vec<u8>,
}

impl Checkpoint {
    #[must_use]
    #[inline(always)]
    pub const fn new(iteration: usize, integer: Vec<u8>) -> Self {
        Self { iteration, integer }
    }

    #[must_use]
    #[inline(always)]
    pub fn data(self) -> (usize, Vec<u8>) {
        (self.iteration, self.integer)
    }
}

impl<T: Allocator + Clone + Copy> Integer<T> {
    /// Writes the digit-reversed value into `output`, reusing its buffer.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `self` is empty or the realignment
    /// bookkeeping is inconsistent; release builds treat both as
    /// unreachable.
    #[inline]
    pub fn reverse_into_integer(&self, output: &mut Integer<GlobalAllocator>) {
        cold_path();
        if self.0.is_empty() {
            impossible!("Tried to reverse an empty integer");
        }

        let output_vec: &mut Vec<Limb, GlobalAllocator> = &mut output.0;
        output_vec.clear();

        for limb in self.0.iter().rev() {
            output_vec.push(Limb(limb.0.reverse()));
        }
        // at this point, the contents of the limbs and the order of the limbs are reversed
        // however, the digits are misaligned

        // safe because of the check at the top
        let skip_len = LV_LEN as u8 - unsafe { self.0.last().unwrap_unchecked() }.len();

        output_vec.push(Limb::new());

        let vec_beginning_ptr = output_vec.as_mut_ptr().cast::<u8>();
        let output_len_bytes = output_vec.len() * LV_LEN;

        let output_slice =
            unsafe { std::slice::from_raw_parts_mut(vec_beginning_ptr, output_len_bytes) };

        debug_assert_eq!(
            output_slice[(output_slice.len() - LV_LEN)..output_slice.len()],
            [0; LV_LEN]
        );

        let right_bound = output_slice.len() - usize::from(LV_LEN as u8 - skip_len);
        if !(right_bound - usize::from(skip_len)).is_multiple_of(LV_LEN) {
            impossible!("Reversal memory copy is not a multiple of {LV_LEN:} bytes");
        }
        output_slice.copy_within(usize::from(skip_len)..right_bound, 0);

        let discarded = output_vec.pop();
        debug_assert_eq!(Limb::new(), discarded.unwrap());
    }

    #[cfg(any(test, feature = "reference-impl"))]
    #[inline]
    pub(crate) fn num_limbs(&self) -> usize {
        if self.0.is_empty() {
            impossible!("Tried to get limb count for an empty integer");
        }

        let num_limbs = self.0.len();
        if num_limbs > 2usize.pow(26) {
            impossible!("Tried to get limb count for an integer with more than 2^26 limbs");
        }

        num_limbs
    }

    #[inline]
    #[cfg(any(debug_assertions, test))]
    pub fn show_differences(&self, rhs: &Self) -> String {
        if self.0.is_empty() {
            impossible!("Tried to show differences between empty integers");
        }

        if self.0.len() != rhs.0.len() {
            impossible!(
                "Tried to show differences between integers of different lengths, {:} vs {:}:
                {self:?}\n{rhs:?}",
                self.0.len(),
                rhs.0.len()
            );
        }

        let mut self_string: String = String::new();
        let mut other_string: String = String::new();

        writeln!(self_string, "Left Integer:").unwrap();
        writeln!(other_string, "Right Integer:").unwrap();

        for (self_limb, other_limb) in self.0.iter().zip(rhs.0.iter()) {
            write!(self_string, "[").unwrap();
            write!(other_string, "[").unwrap();
            for (self_digit, other_digit) in self_limb
                .0
                .as_array()
                .iter()
                .zip(other_limb.0.as_array().iter())
            {
                if likely(self_digit == other_digit) {
                    write!(self_string, "{self_digit}").unwrap();
                    write!(other_string, "{other_digit}").unwrap();
                } else {
                    write!(self_string, "\x1b[31m{self_digit}\x1b[0m").unwrap();
                    write!(other_string, "\x1b[31m{other_digit}\x1b[0m").unwrap();
                }
            }
            writeln!(self_string, "]").unwrap();
            writeln!(other_string, "]").unwrap();
        }

        let mut output_string = String::with_capacity(self_string.len() + other_string.len());
        write!(output_string, "{self_string}\n{other_string}").unwrap();
        output_string
    }

    #[must_use]
    #[inline]
    pub fn has_carries(&self) -> bool {
        if self.0.is_empty() {
            impossible!("Tried to check if empty integer has carries");
        }

        for limb in &self.0 {
            if limb.has_carries() {
                return true;
            }
        }

        false
    }

    #[inline]
    pub fn len(&self) -> u32 {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            impossible!("Tried to get the length of an empty integer");
        }

        if self.0.len() >= 2usize.pow(26) {
            impossible!("Tried to get the length of an integer with more than 2^26 limbs");
        }

        unsafe {
            ((self.0.len() - 1) as u32 * LV_LEN as u32)
                + u32::from(self.0.last().unwrap_unchecked().len())
        }
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        if self.0.is_empty() {
            impossible!();
        }

        #[cfg(debug_assertions)]
        {
            let mut i: usize = 0;
            while i < self.0.len() {
                if self.0.as_slice()[i].is_empty() {
                    i += 1;
                } else {
                    return false;
                }
            }
            true
        }

        #[cfg(not(debug_assertions))]
        {
            false
        }
    }

    #[inline]
    pub fn pack(self) -> Integer<GlobalAllocator> {
        if self.0.is_empty() {
            impossible!("Tried to pack an empty integer");
        }

        // take Limbs in pairs and pack them together

        let mut output_vec: Vec<Limb> = Vec::with_capacity(self.0.len() / 2);

        for limb_pair in self.0.chunks(2) {
            match limb_pair.len() {
                2 => {
                    output_vec.push(limb_pair[0].pack(limb_pair[1]));
                }
                1 => {
                    output_vec.push(limb_pair[0]);
                }
                _ => {
                    unreachable!();
                }
            }
        }

        Integer::<GlobalAllocator>(output_vec)
    }

    #[must_use]
    pub fn unpack(self, allocator: T) -> Self {
        if self.0.is_empty() {
            impossible!("Tried to unpack an empty integer");
        }

        let mut output = Vec::with_capacity_in(self.0.len() * 2, allocator);

        for limb in &self.0 {
            let (low, high) = limb.unpack();
            if !low.is_empty() {
                output.push(low);
            }
            if !high.is_empty() {
                output.push(high);
            }
        }

        Self(output)
    }

    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        let mut output: Vec<LimbVecScalar> = Vec::with_capacity(self.0.len() * LV_LEN);
        for limb in &self.0 {
            output.extend_from_slice(&limb.into_bytes());
        }
        output
    }

    #[must_use]
    #[inline]
    pub fn from_bytes(input: &[[LimbVecScalar; LV_LEN]], allocator: T) -> Self {
        let mut output = Vec::with_capacity_in(input.len(), allocator);
        for limb in input {
            output.push(Limb::from_bytes(*limb));
        }
        Self(output)
    }

    #[inline]
    pub fn into_checkpoint(self, iteration: usize) -> Checkpoint {
        Checkpoint {
            iteration,
            integer: self.pack().into_bytes(),
        }
    }

    /// Rebuilds the integer and its iteration number from a checkpoint.
    ///
    /// # Panics
    ///
    /// Panics if the checkpoint's byte length is not a whole number of
    /// limbs.
    #[must_use]
    #[inline]
    pub fn from_checkpoint(input: &Checkpoint, allocator: T) -> (Self, usize) {
        let chopped_data = Self::chop(&input.integer).unwrap();
        let packed_integer = Self::from_bytes(&chopped_data, allocator);
        let integer = packed_integer.unpack(allocator);
        (integer, input.iteration)
    }

    #[must_use]
    #[inline]
    pub fn chop(data: &[u8]) -> Option<Vec<[LimbVecScalar; LV_LEN]>> {
        data.chunks(LV_LEN)
            .map(|chunk| chunk.try_into().ok())
            .collect()
    }

    /// Formats the raw limb contents, one limb per line.
    ///
    /// # Errors
    ///
    /// Propagates any error from the underlying formatter.
    pub fn display_raw(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct LimbRawDisplay<'a>(&'a Limb);

        impl std::fmt::Display for LimbRawDisplay<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                // delegate the formatting call to `Limb::display_raw`
                self.0.display_raw(f)
            }
        }

        let mut output_string = String::new();

        for limb in self.0.iter().rev() {
            write!(output_string, "{}", LimbRawDisplay(limb))?;
        }

        write!(f, "{}", output_string.trim_start_matches('0'))
    }
}

impl<T: Allocator + Clone + Copy> std::fmt::Debug for Integer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Integer(")?;
        for (i, limb) in self.0.iter().enumerate() {
            write!(f, "\n{i:}: {limb:#?}")?;
        }
        write!(f, "\n)")
    }
}

impl<T: Allocator + Clone + Copy> std::fmt::Display for Integer<T> {
    #[inline(never)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output_string = String::new();
        for limb in self.0.iter().rev() {
            write!(output_string, "{limb}")?;
        }
        write!(f, "{}", output_string.trim_start_matches('0'))
    }
}

impl<T: Allocator + Clone + Copy> std::cmp::PartialEq for Integer<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.0.is_empty() {
            impossible!("Tried to compare an empty integer");
        }

        if self.0.len() != other.0.len() {
            impossible!(
                "Tried to compare two integers of different lengths, {:} vs {:}:
                {self:?}\n{other:?}",
                self.0.len(),
                other.0.len()
            );
        }

        for (a, b) in self.0.iter().zip(other.0.iter()) {
            if likely(a != b) {
                return false;
            }
        }
        cold_path();
        true
    }
}

impl<T: Allocator + Clone + Copy> std::cmp::Eq for Integer<T> {}

/// A base-10 integer. The limbs grow left-to-right, so the most significant limb is the last one in the vector
#[macro_export]
macro_rules! integer {
    ($value:expr) => {{
        let value_str: &str = $value;
        let mut limbs: Vec<Limb> =
            Vec::with_capacity(value_str.len() / $crate::integer_limb::LV_LEN + 1);
        let mut current_limb_digits: Vec<$crate::integer_limb::LimbVecScalar> = Vec::new();

        for digit in value_str.chars().rev() {
            if !digit.is_digit(10) {
                unreachable!("Invalid digit: {}", digit);
            }
            current_limb_digits
                .push(digit.to_digit(10).unwrap() as $crate::integer_limb::LimbVecScalar);

            if current_limb_digits.len() == $crate::integer_limb::LV_LEN {
                let mut limb_bytes: [$crate::integer_limb::LimbVecScalar;
                    $crate::integer_limb::LV_LEN] = [0; $crate::integer_limb::LV_LEN];
                for (i, &digit) in current_limb_digits.iter().enumerate() {
                    limb_bytes[i] = digit;
                }
                limbs.push(Limb($crate::integer_limb::LimbVec::from(limb_bytes)));
                current_limb_digits.clear();
            }
        }

        if !current_limb_digits.is_empty() {
            let mut limb_bytes: [$crate::integer_limb::LimbVecScalar;
                $crate::integer_limb::LV_LEN] = [0; $crate::integer_limb::LV_LEN];
            for (i, &digit) in current_limb_digits.iter().enumerate() {
                limb_bytes[i] = digit;
            }
            limbs.push(Limb($crate::integer_limb::LimbVec::from(limb_bytes)));
        }

        Integer(limbs)
    }};
}

#[cfg(test)]
mod tests;
