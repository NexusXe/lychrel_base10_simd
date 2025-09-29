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

#[cfg(any(
    target_feature = "avx512f",
    target_feature = "sve",
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
            target_feature = "sve",
            feature = "64-byte-limbs"
        )),
        target_feature = "sse"
    ),
    target_feature = "neon",
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

pub use values::*;
pub const LV_BYTES: usize = LV_LEN * (LimbVecScalar::BITS / 8) as usize;

pub type LimbVecScalar = u8;
pub type LimbVec = Simd<LimbVecScalar, LV_LEN>;

pub const WV_LEN: usize = LV_LEN / (WideVecScalar::BITS as usize / LimbVecScalar::BITS as usize);
type WideVec = Simd<WideVecScalar, WV_LEN>;
pub const WV_BYTES: usize = WV_LEN * (WideVecScalar::BITS / 8) as usize;

const fn assert_good_vec_sizes() {
    assert!(std::mem::size_of::<LimbVec>() == std::mem::size_of::<WideVec>());
}

const _: () = assert_good_vec_sizes();

#[cfg(any(not(target_family = "wasm"), not(feature = "global-alloc")))]
mod huge_page_alloc;

#[cfg(any(not(target_family = "wasm"), not(feature = "global-alloc")))]
#[allow(unused_imports)]
pub use huge_page_alloc::*;

macro_rules! impossible {
    ($message:expr) => {
        #[cfg(debug_assertions)]
        unreachable!($message);

        #[cfg(not(debug_assertions))]
        unsafe {
            unreachable_unchecked()
        }
    };
}

/// A 64-byte vector of u8, representing a single "limb" of a large integer.
///
/// Each byte represents a single digit in base 10, with the least significant digit at index 0.
/// Thus, the digits are stored in reverse order.
#[derive(Clone, Copy)]
pub struct Limb(pub LimbVec);

impl const std::cmp::PartialEq for Limb {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        #[inline]
        const fn eq_const(lhs: LimbVec, rhs: LimbVec) -> bool {
            let arr1 = lhs.to_array();
            let arr2 = rhs.to_array();
            let arr1_64b: [WideVecScalar; WV_LEN] = unsafe { transmute(arr1) };
            let arr2_64b: [WideVecScalar; WV_LEN] = unsafe { transmute(arr2) };
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
            lhs == rhs
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
    pub const fn new() -> Self {
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
                panic!("Invalid digit in input value: {c}");
            }
        }
        Self(digits)
    }

    #[inline]
    fn has_carries(&self) -> bool {
        for byte in self.0.as_array() {
            if *byte >= 10 {
                return true;
            }
        }
        false
    }

    #[inline]
    fn reverse(self) -> Self {
        Self(self.0.reverse())
    }

    #[inline]
    fn len(&self) -> usize {
        let zeros = LimbVec::splat(0);
        let eq_mask = self.0.simd_ne(zeros);
        let bitmask = eq_mask.to_bitmask();
        LV_LEN - (bitmask.leading_zeros() as usize - (64 - LV_LEN))
    }

    #[inline(always)]
    unsafe fn shl_quad<const N: u64>(&self) -> Self {
        Self(unsafe { transmute::<WideVec, LimbVec>(transmute::<LimbVec, WideVec>(self.0) << N) })
    }

    #[allow(dead_code)]
    #[inline(always)]
    unsafe fn shr_quad<const N: u64>(&self) -> Self {
        Self(unsafe { transmute::<WideVec, LimbVec>(transmute::<LimbVec, WideVec>(self.0) >> N) })
    }

    fn pack(self, other: Self) -> Self {
        debug_assert!(!self.has_carries());
        debug_assert!(!other.has_carries());

        debug_assert_eq!(LimbVec::splat(0), self.0 & LimbVec::splat(0xF0));
        debug_assert_eq!(LimbVec::splat(0), other.0 & LimbVec::splat(0xF0));

        unsafe {
            let other_u64: WideVec = transmute(other.0);
            let other_shifted: LimbVec = transmute(other_u64 << 4);
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
        let digits = self.0.as_array();
        for i in digits.iter().rev() {
            write!(f, "{i}")?;
        }
        Ok(())
    }
}

impl const std::default::Default for Limb {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Add for Limb {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        unsafe {
            let input_64: WideVec = transmute(self.0);
            let other_64: WideVec = transmute(other.0);
            let output_64: WideVec = input_64 + other_64;
            Self(transmute::<WideVec, LimbVec>(output_64))
        }
    }
}

impl std::fmt::Display for Limb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let digits = self.0.to_array();
        for i in digits.iter().rev() {
            if *i > 9 as LimbVecScalar {
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

#[derive(Clone)]
pub struct Integer<T: Allocator + Clone + Copy>(pub Vec<Limb, T>);

#[derive(Debug, PartialEq, Eq)]
pub struct Checkpoint {
    iteration: usize,
    pub integer: Vec<u8>,
}

impl Checkpoint {
    #[must_use]
    pub const fn new(iteration: usize, integer: Vec<u8>) -> Self {
        Self { iteration, integer }
    }

    #[must_use]
    pub fn data(self) -> (usize, Vec<u8>) {
        (self.iteration, self.integer)
    }
}

impl<T: Allocator + Clone + Copy> Integer<T> {
    #[inline]
    pub fn reverse_into_integer(&self, output: &mut Integer<GlobalAllocator>) {
        if self.0.is_empty() {
            impossible!("Tried to reverse an empty integer");
        }

        let output_vec: &mut Vec<Limb, GlobalAllocator> = &mut output.0;
        output_vec.clear();

        for limb in self.0.iter().rev() {
            output_vec.push(limb.reverse());
        }
        // at this point, the contents of the limbs and the order of the limbs are reversed
        // however, the digits are misaligned

        // safe because of the check at the top
        let skip_len: usize = LV_LEN - unsafe { self.0.last().unwrap_unchecked() }.len();

        output_vec.push(Limb::new());

        let vec_beginning_ptr = output_vec.as_mut_ptr().cast::<u8>();
        let output_len_bytes = output_vec.len() * LV_LEN;

        let output_slice =
            unsafe { std::slice::from_raw_parts_mut(vec_beginning_ptr, output_len_bytes) };

        debug_assert_eq!(
            output_slice[(output_slice.len() - LV_LEN)..output_slice.len()],
            [0; LV_LEN]
        );

        let right_bound = output_slice.len() - (LV_LEN - skip_len);
        if !(right_bound - skip_len).is_multiple_of(LV_LEN) {
            impossible!("Reversal memory copy is not a multiple of 64 bytes");
        }
        output_slice.copy_within(skip_len..right_bound, 0);

        let discarded = output_vec.pop();
        debug_assert_eq!(Limb::new(), discarded.unwrap());
    }

    pub fn fused_reverse_add_asm_interleave(&mut self) -> bool {
        use std::ptr::read_unaligned;
        // instead of reversing into a seperate vector, reverse and pack into the original limb

        if self.0.is_empty() {
            impossible!("Tried to reverse and add empty integer");
        }

        let total_limbs = self.0.len();

        self.0.push(Limb::new()); // padding

        let skip_len = LV_LEN - unsafe { self.0.get_unchecked(total_limbs - 1).len() };

        let limbs_ptr = self.0.as_mut_ptr().cast::<LimbVec>();
        let rev_ptr = unsafe { &mut self.0.get_unchecked_mut(total_limbs.unchecked_sub(1)).0 }
            as *mut LimbVec;

        for i in 0..total_limbs.div_ceil(2) {
            unsafe {
                let left_limb_ptr = limbs_ptr.add(i);
                let right_limb_ptr = rev_ptr.sub(i);

                let lhs = &mut *left_limb_ptr;
                let rhs = &mut *right_limb_ptr;

                // shift these as qwords since byte-wise shifts use gfni
                let lhs_output = *lhs | Limb(rhs.reverse()).shl_quad::<4>().0;
                let rhs_output = *rhs | Limb(lhs.reverse()).shl_quad::<4>().0;
                *lhs = lhs_output;
                *rhs = rhs_output;
            }
        }

        let mut overflowed = false;
        let mut ever_carried = false;

        for (_, limb) in self
            .0
            .iter_mut()
            .enumerate()
            .take_while(|(idx, _)| idx < &total_limbs)
        {
            // the `impossible!()` macro contains its own `unsafe{}` block, which causes a warning
            #[allow(unused_unsafe)]
            unsafe {
                let limb_ptr = &raw const limb.0;

                #[cfg(debug_assertions)]
                let reversed_limb: LimbVec = read_unaligned(limb_ptr.byte_add(skip_len)) >> 4;

                #[cfg(not(debug_assertions))]
                let reversed_limb: LimbVec =
                    read_unaligned((limb_ptr as usize + skip_len) as *const LimbVec) >> 4;

                limb.0 = (limb.0 << 4) >> 4;

                let forward_carry = overflowed;
                const CARRY_MASK_CMP: LimbVec = LimbVec::splat(10);

                #[cfg(all(target_feature = "avx512bw", not(feature = "no-avx")))]
                {
                    limb.0 = _mm512_add_epi64(limb.0.into(), reversed_limb.into()).into();

                    for result in limb.0.as_array() {
                        if *result > 18 {
                            impossible!("Got impossible addition result");
                        }
                    }

                    // incorporate previous limb carry into carry propogation
                    // do the loop once by hand, with some tweaks
                    // doing it like this instead of adding one to the lowest digit separately is ~34% faster
                    let carry_mask = _mm512_cmpge_epu8_mask(limb.0.into(), CARRY_MASK_CMP.into());

                    if likely((carry_mask != 0) || forward_carry) {
                        overflowed = carry_mask & 0x8000_0000_0000_0000_u64 != 0; // not a branch, just shifts bits right

                        ever_carried = true;

                        limb.0 = _mm512_mask_sub_epi8(
                            limb.0.into(),
                            carry_mask,
                            limb.0.into(),
                            CARRY_MASK_CMP.into(),
                        )
                        .into();

                        limb.0 = _mm512_mask_add_epi8(
                            limb.0.into(),
                            (carry_mask << 1) | __mmask64::from(forward_carry), // do a round of carry propogation AND deal with a forward carry. absolute cinema
                            limb.0.into(),
                            _mm512_set1_epi8(1),
                        )
                        .into();

                        loop {
                            let carry_mask =
                                _mm512_cmpge_epu8_mask(limb.0.into(), CARRY_MASK_CMP.into());

                            limb.0 = _mm512_mask_sub_epi8(
                                limb.0.into(),
                                carry_mask,
                                limb.0.into(),
                                CARRY_MASK_CMP.into(),
                            )
                            .into();

                            limb.0 = _mm512_mask_add_epi8(
                                limb.0.into(),
                                carry_mask << 1,
                                limb.0.into(),
                                _mm512_set1_epi8(1),
                            )
                            .into();

                            if carry_mask & 0x8000_0000_0000_0000_u64 != 0 {
                                overflowed = true;
                            } else if carry_mask == 0 {
                                break;
                            }
                        }
                    }
                }

                #[cfg(any(not(target_feature = "avx512bw"), feature = "no-avx"))]
                {
                    *limb = *limb + Limb(reversed_limb);

                    for result in limb.0.as_array() {
                        if *result > 18 {
                            impossible!("Got impossible addition result");
                        }
                    }

                    let carry_mask = limb.0.simd_ge(CARRY_MASK_CMP);
                    if likely(forward_carry || carry_mask.any()) {
                        overflowed = carry_mask.test(LV_LEN - 1);

                        ever_carried = true;

                        let subtracted_limb = limb.0 - CARRY_MASK_CMP;
                        limb.0 = carry_mask.select(subtracted_limb, limb.0);

                        let added_limb = limb.0 + LimbVec::splat(1);
                        limb.0 = (carry_mask.shift_elements_right::<1>(forward_carry))
                            .select(added_limb, limb.0);

                        loop {
                            let carry_mask = limb.0.simd_ge(CARRY_MASK_CMP);
                            if carry_mask.test(LV_LEN - 1) {
                                overflowed = true;
                            } else if !carry_mask.any() {
                                cold_path();
                                break;
                            }

                            ever_carried = true;

                            let subtracted_limb = limb.0 - CARRY_MASK_CMP;
                            limb.0 = carry_mask.select(subtracted_limb, limb.0);

                            let added_limb = limb.0 + LimbVec::splat(1);
                            limb.0 = carry_mask
                                .shift_elements_right::<1>(false)
                                .select(added_limb, limb.0);
                        }
                    }
                }
            }
        }

        if overflowed {
            #[cfg(debug_assertions)]
            unsafe {
                *rev_ptr.add(1).cast::<u8>() = 1;
            }; // this limb is already zeroed for padding, so just set one byte

            #[cfg(not(debug_assertions))]
            unsafe {
                // for some reason an overflow check is happening on this addition
                *((rev_ptr as usize).unchecked_add(LV_LEN) as *mut u8) = 1; // do the math manually with unchecked addition to remove an overflow check branch 
            }
        } else {
            self.0.pop();
        }
        likely(ever_carried)
    }

    #[inline]
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
    pub fn len(&self) -> usize {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            impossible!("Tried to get the length of an empty integer");
        }

        unsafe { ((self.0.len() - 1) * LV_LEN) + self.0.last().unwrap_unchecked().len() }
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
        return {
            let mut i: usize = 0;
            while i < self.0.len() {
                if self.0.as_slice()[i].is_empty() {
                    i += 1;
                } else {
                    return false;
                }
            }
            true
        };

        #[cfg(not(debug_assertions))]
        {
            false
        }
    }

    pub fn pack(self) -> Integer<GlobalAllocator> {
        if self.0.is_empty() {
            impossible!("Tried pack an empty integer");
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
    pub fn unpack(self, allocator: T) -> Integer<T> {
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

        Integer::<T>(output)
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
    pub fn from_bytes(input: &[[LimbVecScalar; LV_LEN]], allocator: T) -> Integer<T> {
        let mut output = Vec::with_capacity_in(input.len(), allocator);
        for limb in input {
            output.push(Limb::from_bytes(*limb));
        }
        Integer(output)
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
    pub fn from_checkpoint(input: &Checkpoint, allocator: T) -> (Integer<T>, usize) {
        let chopped_data = Integer::<T>::chop(&input.integer).unwrap();
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
                panic!("Invalid digit: {}", digit);
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
