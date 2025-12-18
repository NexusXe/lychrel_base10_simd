#[cfg(target_arch = "x86")]
#[allow(unused_imports)]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

use std::alloc::{Allocator, Global as GlobalAllocator};
use std::fmt::Write;
#[cfg(not(debug_assertions))]
use std::hint::unreachable_unchecked;
use std::hint::{cold_path, likely};
use std::intrinsics::const_eval_select;
use std::mem::transmute;

use std::simd::prelude::*;

use zerocopy::{FromZeros, IntoBytes, KnownLayout, transmute};

#[cfg(any(
    target_feature = "avx512f",
    target_arch = "powerpc64",
    feature = "64-byte-limbs"
))] // 512-bit vectors
mod values {
    pub const LV_LEN: usize = 64;
    pub type WideVecScalar = u64;
}

#[cfg(all(
    not(any(target_feature = "avx512f", target_feature = "sve")),
    target_feature = "avx2",
    not(feature = "64-byte-limbs")
))] // 256-bit vectors
mod values {
    pub const LV_LEN: usize = 32;
    pub type WideVecScalar = u64;
}

#[cfg(any(
    all(
        not(any(
            target_feature = "avx512f",
            target_feature = "avx2",
            feature = "64-byte-limbs"
        )),
        target_feature = "sse",
        target_feature = "neon"
    ),
    target_feature = "sve",
    target_feature = "simd128"
))] // 128-bit vectors
mod values {
    pub const LV_LEN: usize = 16;
    pub type WideVecScalar = u64;
}

#[cfg(all(
    not(any(
        target_feature = "avx512f",
        target_feature = "avx2",
        target_feature = "sve",
        feature = "64-byte-limbs",
        target_feature = "sse"
    )),
    target_feature = "fxsr",
    target_pointer_width = "32"
))] // 64-bit vectors, 32-bit pointer
mod values {
    pub const LV_LEN: usize = 8;
    pub type WideVecScalar = u64;
}
#[cfg(all(
    not(any(
        target_feature = "avx512f",
        target_feature = "avx2",
        target_feature = "sve",
        feature = "64-byte-limbs",
        target_feature = "sse",
        target_feature = "fxsr"
    )),
    target_pointer_width = "32"
))] // 32-bit vectors, 32-bit pointer
mod values {
    pub const LV_LEN: usize = 4;
    pub type WideVecScalar = u32;
}

#[cfg(all(target_pointer_width = "16", not(feature = "64-byte-limbs")))]
mod values {
    pub const LV_LEN: usize = 2;
    pub type WideVecScalar = u16;
}

// reasonable fallback for zerocopy. TODO: will this work on non-AVX512 builds?
#[cfg(not(any(
    target_feature = "avx512f",
    target_feature = "avx2",
    target_feature = "sve",
    feature = "64-byte-limbs",
    target_feature = "sse",
    target_feature = "fxsr",
    target_pointer_width = "64",
    target_pointer_width = "32",
    target_pointer_width = "16",
)))]
mod values {
    pub const LV_LEN: usize = 64;
    pub type WideVecScalar = u64;
}

pub use values::*;
pub const LV_BYTES: usize = LV_LEN * (LimbVecScalar::BITS / 8) as usize;

pub type LimbVecScalar = u8;
pub type LimbVec = Simd<LimbVecScalar, { LV_LEN }>;

pub const WV_LEN: usize = LV_LEN / (WideVecScalar::BITS as usize / LimbVecScalar::BITS as usize);
type WideVec = Simd<WideVecScalar, WV_LEN>;
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
    ($message:expr) => {
        #[cfg(debug_assertions)]
        unreachable!($message);

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

impl const std::cmp::PartialEq for Limb {
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

impl const From<Limb> for LimbVec {
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

impl const From<LimbVec> for Limb {
    #[inline]
    fn from(val: LimbVec) -> Self {
        Self(val)
    }
}

impl Limb {
    #[inline]
    #[must_use]
    const fn new() -> Self {
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
    fn len(&self) -> u8 {
        const ZEROS: LimbVec = LimbVec::splat(0);

        #[allow(unused)]
        #[inline(always)]
        fn len_portable(input: &Limb) -> u8 {
            let eq_mask = input.0.simd_ne(ZEROS);
            let bitmask = eq_mask.to_bitmask();
            64 - bitmask.leading_zeros() as u8
        }

        #[cfg(all(target_feature = "avx512bw", not(feature = "no-avx")))]
        #[inline(always)]
        fn len_avx512bw(input: &Limb) -> u8 {
            unsafe {
                let bitmask = _mm512_cmpneq_epu8_mask(input.0.into(), ZEROS.into());
                64 - bitmask.leading_zeros() as u8
            }
        }

        #[cfg(all(target_feature = "avx512bw", not(feature = "no-avx")))]
        {
            debug_assert_eq!(len_portable(self), len_avx512bw(self));
            len_avx512bw(self)
        }

        #[cfg(not(all(target_feature = "avx512bw", not(feature = "no-avx"))))]
        len_portable(self)
    }

    #[inline(always)]
    /// ## Safety
    /// N must be <= 8
    const unsafe fn shl_wide<const N: u8>(&self) -> Self {
        if N > 8 {
            panic!("Limb::shr_wide() must not be used with N > 8");
        }

        #[inline(always)]
        fn shl_wide_rt<const N: u8>(input: LimbVec) -> LimbVec {
            unsafe {
                transmute::<WideVec, LimbVec>(
                    transmute::<LimbVec, WideVec>(input) << N as WideVecScalar,
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
        if N > 8 {
            panic!("Limb::shr_wide() must not be used with N > 8");
        }

        #[inline(always)]
        fn shr_wide_rt<const N: u8>(input: LimbVec) -> LimbVec {
            unsafe {
                transmute::<WideVec, LimbVec>(
                    transmute::<LimbVec, WideVec>(input) >> N as WideVecScalar,
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

    #[inline]
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
                    *left_limb_ptr | Limb((&mut *right_limb_ptr).reverse()).shl_wide::<4>().0;
                let rhs_output =
                    *right_limb_ptr | Limb((&mut *left_limb_ptr).reverse()).shl_wide::<4>().0;
                #[cfg(all(
                    target_feature = "avx512f",
                    target_os = "windows",
                    not(feature = "no-stream")
                ))]
                {
                    _mm512_stream_si512(left_limb_ptr.cast(), lhs_output.into());
                    _mm512_stream_si512(right_limb_ptr.cast(), rhs_output.into());
                }

                #[cfg(any(
                    not(target_feature = "avx512f"),
                    target_os = "linux",
                    feature = "no-stream"
                ))]
                {
                    *left_limb_ptr = lhs_output;
                    *right_limb_ptr = rhs_output;
                }
            }
        }
    }

    #[inline(always)]
    fn zip_halves(limbs_ptr: *mut LimbVec, total_limbs: usize) {
        let rev_ptr = unsafe { limbs_ptr.add(total_limbs - 1) };
        unsafe { Self::zipper(limbs_ptr, rev_ptr, 0, total_limbs.div_ceil(2)) };
    }
}

impl const std::default::Default for Limb {
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

        let right_bound = output_slice.len() as u32 - u32::from(LV_LEN as u8 - skip_len);
        if !(right_bound - u32::from(skip_len)).is_multiple_of(LV_LEN as u32) {
            impossible!("Reversal memory copy is not a multiple of 64 bytes");
        }
        output_slice.copy_within(skip_len as usize..right_bound as usize, 0);

        let discarded = output_vec.pop();
        debug_assert_eq!(Limb::new(), discarded.unwrap());
    }

    #[inline]
    fn num_limbs(&self) -> usize {
        if self.0.is_empty() {
            impossible!("Tried to get limb count for an empty integer");
        }

        let num_limbs = self.0.len();
        if num_limbs > 2usize.pow(26) {
            impossible!("Tried to get limb count for an integer with more than 2^26 limbs");
        }

        num_limbs
    }

    #[inline(always)]
    pub fn fused_reverse_add_asm_interleave(&mut self) -> bool {
        use std::ptr::read_unaligned;

        if self.0.is_empty() {
            impossible!("Tried to reverse and add empty integer");
        }

        let total_limbs = self.num_limbs();
        if total_limbs > 2usize.pow(26) {
            impossible!("Tried to iterate over an integer with more than 2^26 limbs");
        }

        self.0.push(Limb::new()); // padding

        let skip_len = LV_LEN as u8 - unsafe { self.0.get_unchecked(total_limbs - 1).len() };

        let limbs_ptr = self.0.as_mut_ptr().cast::<LimbVec>();
        let rev_ptr = unsafe { limbs_ptr.add(total_limbs - 1) };
        if !std::ptr::eq(rev_ptr, unsafe {
            &mut self.0.get_unchecked_mut(total_limbs.unchecked_sub(1)).0
        }) {
            impossible!("Incoherent rev_ptr");
        }

        Limb::zip_halves(limbs_ptr, total_limbs);

        #[cfg(all(target_arch = "x86_64", not(feature = "no-prefetch")))]
        #[allow(clippy::pointers_in_nomem_asm_block)] // ptr is being used for prefetch
        unsafe {
            std::arch::asm!(r#"
            # implicit xor {i:e}, {i:e}; 2 bytes, 1 uop
            2:
            prefetchw byte ptr [{limbs_ptr:r} + {i:r} * 8]      # 4 bytes, 1 uop
            add {i:l}, 8                                        # 3 or 4 bytes; fuses with conditional jump for 1 uop for both
            jns 2b                                              # 2 bytes; shares uop with add instruction
            "#,
            limbs_ptr = in(reg) limbs_ptr,
            i = inout(reg) 0 => _,
            options(nostack, nomem));
            // in addition to the first limbs, the last ones are also accessed first
            // however, they are likely still in cache
        }

        let mut overflowed = false;
        let mut ever_carried = false;

        // addition process
        // the reversed data is offset, but is guaranteed to be fully contained between the current and next cache line
        // conveniently, the next cache line is going to be needed soon anyway so this is fine
        for (_, limb) in self
            .0
            .iter_mut()
            .enumerate()
            .take_while(|(idx, _)| idx < &total_limbs)
        {
            unsafe {
                let limb_ptr = &raw const limb.0;

                let reversed_limb: LimbVec =
                    read_unaligned(limb_ptr.byte_add(skip_len as usize)) >> 4;

                limb.0 = (limb.0 << 4) >> 4;

                let forward_carry = overflowed;

                *limb = *limb + Limb(reversed_limb);
                for result in limb.0.as_array() {
                    if *result > 18 {
                        impossible!("Got impossible addition result");
                    }
                }

                const TEN_VEC_BYTES: LimbVec = LimbVec::splat(10);
                const CARRY_NINE_CMP: LimbVec = LimbVec::splat(9);

                #[cfg(all(target_feature = "avx512bw", not(feature = "no-avx")))]
                {
                    // incorporate previous limb carry into carry propogation
                    // do the loop once by hand, with some tweaks
                    // doing it like this instead of adding one to the lowest digit separately is ~34% faster
                    let mut carry_mask =
                        _mm512_cmpgt_epu8_mask(limb.0.into(), CARRY_NINE_CMP.into());

                    let ng_carry_mask =
                        _mm512_cmpeq_epu8_mask(limb.0.into(), CARRY_NINE_CMP.into());

                    // 2: 26.2
                    // 3: 24.5
                    // 4: 25.3
                    for _ in 0..3 {
                        carry_mask |= ng_carry_mask & (carry_mask << 1);
                    }

                    if likely(carry_mask != 0) || forward_carry {
                        ever_carried = true;
                        overflowed = carry_mask & 0x8000_0000_0000_0000_u64 != 0; // not a branch, just shifts bits right

                        let mut output = _mm512_mask_sub_epi8(
                            limb.0.into(),
                            carry_mask,
                            limb.0.into(),
                            TEN_VEC_BYTES.into(),
                        );

                        output = _mm512_mask_add_epi8(
                            output,
                            (carry_mask << 1) | __mmask64::from(forward_carry), // do a round of carry propogation AND deal with a forward carry. absolute cinema
                            output,
                            _mm512_set1_epi8(1),
                        );

                        carry_mask = _mm512_cmpgt_epu8_mask(output, CARRY_NINE_CMP.into());
                        while likely(carry_mask != 0) {
                            if likely(carry_mask & 0x8000_0000_0000_0000_u64 != 0) {
                                overflowed = true;
                            }

                            output = _mm512_mask_sub_epi8(
                                output,
                                carry_mask,
                                output,
                                TEN_VEC_BYTES.into(),
                            );

                            output = _mm512_mask_add_epi8(
                                output,
                                carry_mask << 1,
                                output,
                                _mm512_set1_epi8(1),
                            );

                            carry_mask = _mm512_cmpgt_epu8_mask(output, CARRY_NINE_CMP.into());
                            // at this point, three rounds of carry propogation have been done. chances are, no more will be needed
                        }

                        #[cfg(all(
                            target_feature = "avx512f",
                            target_os = "windows",
                            not(feature = "no-stream")
                        ))]
                        {
                            _mm512_stream_si512(limb_ptr as *mut __m512i, output);
                        }

                        #[cfg(any(
                            not(target_feature = "avx512f"),
                            target_os = "linux",
                            feature = "no-stream"
                        ))]
                        {
                            limb.0 = output.into();
                        }
                    } else {
                        cold_path();
                    }
                }

                #[cfg(any(not(target_feature = "avx512bw"), feature = "no-avx"))]
                {
                    let mut carry_mask = limb.0.simd_gt(CARRY_NINE_CMP);
                    let ng_carry_mask = limb.0.simd_eq(CARRY_NINE_CMP);

                    for _ in 0..3 {
                        carry_mask |= ng_carry_mask & (carry_mask.shift_elements_right::<1>(false));
                    }

                    if likely(carry_mask.any()) || forward_carry {
                        ever_carried = true;
                        overflowed = carry_mask.test_unchecked(LV_LEN - 1);

                        let mut output = carry_mask.select(limb.0 - TEN_VEC_BYTES, limb.0);

                        output = (carry_mask.shift_elements_right::<1>(forward_carry))
                            .select(output + LimbVec::splat(1), output);

                        carry_mask = output.simd_gt(CARRY_NINE_CMP);

                        while likely(carry_mask.any()) {
                            if likely(carry_mask.test_unchecked(LV_LEN - 1)) {
                                overflowed = true;
                            }

                            let subtracted_limb = output - TEN_VEC_BYTES;
                            output = carry_mask.select(subtracted_limb, output);

                            let added_limb = output + LimbVec::splat(1);
                            output = carry_mask
                                .shift_elements_right::<1>(false)
                                .select(added_limb, output);

                            carry_mask = output.simd_gt(CARRY_NINE_CMP);
                        }
                        limb.0 = output;
                    }
                }
            }
        }

        let pad_ptr = unsafe { rev_ptr.add(1) as *mut WideVec };

        if unsafe { *pad_ptr != std::mem::zeroed() } {
            impossible!("Dirty padding data!");
        }

        if likely(overflowed) {
            #[cfg(all(target_feature = "avx512f", not(feature = "no-stream")))]
            unsafe {
                // by writing the entire 64-byte cache line again, this memory doesn't have to be read at all to set the overflow
                debug_assert_eq!(
                    LimbVec::from_array([
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                    std::mem::transmute::<WideVec, LimbVec>(WideVec::from_array([
                        1, 0, 0, 0, 0, 0, 0, 0
                    ]))
                );
                *pad_ptr = WideVec::from_array([1, 0, 0, 0, 0, 0, 0, 0]);
            }

            #[cfg(not(all(target_feature = "avx512f", not(feature = "no-stream"))))]
            unsafe {
                *((rev_ptr as usize).unchecked_add(LV_LEN) as *mut u8) = 1;
            }
        } else {
            self.0.pop();
        }

        #[cfg(all(target_arch = "x86_64", not(feature = "no-prefetch")))]
        #[allow(clippy::pointers_in_nomem_asm_block)] // ptr is being used for prefetch
        unsafe {
            // prefetch the first 16 limbs since, for integers larger than L3$, they've probably been evicted by now
            // prefetching in asm because I don't want this loop unrolled
            // while it probably doesn't matter, less pollution in the L1i$ and the L1$ overall is good
            // asm version uses 12 bytes overall, whereas unrolled version was 7 * 16 = 112 bytes
            std::arch::asm!(r#"
            # implicit xor {i:e}, {i:e}; 2 bytes, 1 uop
            2:
            prefetchw byte ptr [{limbs_ptr:r} + {i:r} * 8]      # 4 bytes, 1 uop
            neg {i:r}                                           # 3 bytes, 1 uop
            prefetchw byte ptr [{rev_ptr:r} + 0 + {i:r} * 8]    # 4 bytes, 1 uop
            neg {i:r}                                           # 3 bytes, 1 uop
            add {i:l}, 8                                        # 3 or 4 bytes; fuses with conditional jump for 1 uop for both
            jns 2b                                              # 2 bytes; shares uop with add instruction
            "#,
            limbs_ptr = in(reg) limbs_ptr,
            rev_ptr = in(reg) rev_ptr,
            i = inout(reg) 0 => _,
            options(nostack, nomem));
            // in addition to the first limbs, the last ones are also accessed first
            // however, they are likely still in cache
        }

        likely(ever_carried)
    }

    #[inline]
    #[cfg(debug_assertions)]
    pub fn show_differences(&self, rhs: &Self) -> String {
        if self.0.is_empty() {
            impossible!("Tried to show differences between empty integers");
        }

        if self.0.len() != rhs.0.len() {
            #[cfg(debug_assertions)]
            unreachable!(
                "Tried to show differences between integers of different lengths, {:} vs {:}:
                {self:?}\n{rhs:?}",
                self.0.len(),
                rhs.0.len()
            );

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
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
            #[cfg(debug_assertions)]
            unreachable!(); // Any operations with Integers that contain no data are #UB

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked(); // give the compiler a chance to refuse to run with an unsafe precondition check
            }
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
            #[cfg(debug_assertions)]
            unreachable!(
                "Tried to compare two integers of different lengths, {:} vs {:}:
                {self:?}\n{other:?}",
                self.0.len(),
                other.0.len()
            );

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
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
