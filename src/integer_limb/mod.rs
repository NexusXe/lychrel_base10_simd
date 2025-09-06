use std::arch::x86_64::*;
use std::fmt::Write;
#[cfg(not(debug_assertions))]
use std::hint::unreachable_unchecked;
use std::hint::{cold_path, likely};
use std::intrinsics::const_eval_select;
use std::simd::prelude::*;

use windows::Win32::System::Memory::{
    GetLargePageMinimum, MEM_COMMIT, MEM_LARGE_PAGES, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc2,
};
use windows::Win32::System::Threading::GetCurrentProcess;


pub(crate) const HUGE_PAGE_SIZE_BYTES: usize = 2097152 * 512; // 1 GiB huge page
pub(crate) const FIXEDVEC_CAPACITY: usize = HUGE_PAGE_SIZE_BYTES / std::mem::size_of::<Limb>();

/// A 64-byte vector of u8, representing a single "limb" of a large integer.
/// Each byte represents a single digit in base 10, with the least significant digit at index 0.
/// Thus, the digits are stored in reverse order.
#[derive(Clone, Copy)]
pub(crate) struct Limb(pub(crate) u8x64);

impl const std::cmp::PartialEq for Limb {
    fn eq(&self, other: &Self) -> bool {
        const fn eq_const(lhs: u8x64, rhs: u8x64) -> bool {
            use std::mem::transmute;
            let arr1 = lhs.to_array();
            let arr2 = rhs.to_array();
            let arr1_64b: [u64; 8] = unsafe { transmute(arr1) };
            let arr2_64b: [u64; 8] = unsafe { transmute(arr2) };
            let mut i: usize = 8;
            while i > 0 {
                if arr1_64b[i] == arr2_64b[i] {
                    i -= 1;
                } else {
                    return false;
                }
            }
            false
        }

        fn eq_rt(lhs: u8x64, rhs: u8x64) -> bool {
            lhs == rhs
        }

        const_eval_select((self.0, other.0), eq_const, eq_rt)
    }
}

impl std::cmp::Eq for Limb {}

impl From<Limb> for __m512i {
    #[inline]
    fn from(val: Limb) -> Self {
        val.0.into()
    }
}

impl const From<Limb> for u8x64 {
    #[inline]
    fn from(val: Limb) -> Self {
        val.0
    }
}

impl From<__m512i> for Limb {
    #[inline]
    fn from(val: __m512i) -> Self {
        Self(val.into())
    }
}

impl const From<u8x64> for Limb {
    #[inline]
    fn from(val: u8x64) -> Self {
        Self(val)
    }
}

#[allow(dead_code)]
impl Limb {
    #[inline]
    pub(crate) const fn new() -> Self {
        Self(u8x64::splat(0))
    }

    fn new_from_value(value: u128) -> Self {
        let input_digits = value.to_string();
        let mut digits = u8x64::splat(0);
        for (i, c) in input_digits.chars().rev().enumerate() {
            if let Some(digit) = c.to_digit(10) {
                digits[i] = digit as u8;
            } else {
                panic!("Invalid digit in input value: {c}");
            }
        }
        Self(digits)
    }

    #[inline]
    fn has_carries(&self) -> bool {
        let self_vector: __m512i = (*self).into();
        let compare: __m512i = __m512i::from(u8x64::splat(10));
        let carries = unsafe { _mm512_cmpge_epu8_mask(self_vector, compare) };
        carries != 0
    }

    #[inline]
    fn process_carries(&self) -> (Self, bool) {
        const ONE_VEC: u8x64 = u8x64::splat(1);
        const COMPARE_VEC: u8x64 = u8x64::splat(10);

        if !self.has_carries() {
            return (*self, false);
        }

        let compare: __m512i = __m512i::from(COMPARE_VEC);
        let mut digits: __m512i = __m512i::from(self.0);
        let mut carries_past_last: bool = false;

        let mut carries: u64 = 0;

        for _ in 0..8 {
            for _ in 0..8 {
                carries = unsafe { _mm512_cmpge_epu8_mask(digits, compare) };

                // "unneccesarily" repeating this 64 times seems worth it to make this branchless,
                // but adding this makes this function comically faster for larger Integers

                digits = unsafe { _mm512_mask_sub_epi8(digits, carries, digits, compare) };
                carries_past_last |= carries & 0x8000_0000_0000_0000_u64 != 0; // the most we can ever carry is 1, so we track if this bit is ever set; it getting set multiple times is irrelevant to the final result
                // now add the carries to the next digit
                digits =
                    unsafe { _mm512_mask_add_epi8(digits, carries << 1, digits, ONE_VEC.into()) };
            }
            // the compiler automatically partially unrolls this carry propogation loop into chunks of 8 at a time
            // if the early exit carry check is done every loop, the result after unrolling is very branchy
            // instead, do it after every "run" of 8 loops
            // TODO: is this actually faster?
            if carries == 0 {
                break;
            }
        }

        (digits.into(), carries_past_last)
    }

    #[inline]
    fn reverse(self) -> Self {
        let self_u8x64: u8x64 = self.0;
        Limb(self_u8x64.reverse())
    }

    #[inline]
    fn len(&self) -> usize {
        let zero: __m512i = __m512i::from(u8x64::splat(0));
        let digit_mask = unsafe { _mm512_cmpeq_epu8_mask(self.0.into(), zero) };
        64 - digit_mask.leading_ones() as usize
    }

    fn pack(self, other: Self) -> Self {
        debug_assert!(!self.has_carries());
        debug_assert!(!other.has_carries());

        let self_vector: __m512i = self.into();
        let other_vector: __m512i = other.into();

        debug_assert_eq!(u8x64::splat(0), unsafe {
            _mm512_and_epi64(self_vector, u8x64::splat(0xF0).into()).into()
        });
        debug_assert_eq!(u8x64::splat(0), unsafe {
            _mm512_and_epi64(other_vector, u8x64::splat(0xF0).into()).into()
        });

        let other_shifted = unsafe { _mm512_slli_epi64(other_vector, 4) };
        unsafe { _mm512_or_si512(self_vector, other_shifted) }.into()
    }

    #[inline]
    fn unpack(&self) -> (Self, Self) {
        (Limb((self.0 << 4) >> 4), Limb(self.0 >> 4))
    }

    const fn into_bytes(self) -> [u8; 64] {
        self.0.to_array()
    }

    const fn from_bytes(input: [u8; 64]) -> Self {
        Self(u8x64::from_array(input))
    }

    #[inline]
    const fn is_empty(&self) -> bool {
        self == &Self::new()
    }

    fn display_raw(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let digits: [u8; 64] = self.0.into();
        for i in digits.iter().rev() {
            write!(f, "{i}")?;
        }
        Ok(())
    }

    /// A "Simple Example" for "byte-granularity bit rotate"
    /// As far as I'm concerned, this is a function that uses arcane magic to rotate each byte by 4 bits
    #[inline]
    pub(crate) fn ror4_galois(input: u8x64) -> u8x64 {
        let input_vec: __m512i = input.into();
        unsafe {
            let rorb4: __m512i = _mm512_set1_epi64(0x1020408001020408);
            _mm512_gf2p8affine_epi64_epi8(input_vec, rorb4, 0x0).into()
        }
    }
}

impl Default for Limb {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Add for Limb {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        //unsafe { _mm512_add_epi8(self.into(), other.into()) }.into()
        //Limb(self.0 + other.0) // should compile to be the same
        unsafe { _mm512_add_epi64(self.into(), other.into()) }.into() // use larger object size because each number will never overflow its byte boundary
    }
}

impl std::fmt::Display for Limb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let digits: [u8; 64] = self.0.into();
        for i in digits.iter().rev() {
            if i > &9u8 {
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
        let digits: [u8; 64] = self.0.into();
        write!(f, "[")?;
        for i in &digits {
            write!(f, "{i}")?;
        }
        write!(f, "]")
    }
}

#[derive(Clone)]
//#[repr(align(64))]
pub struct Integer(pub(crate) Vec<Limb>);

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Checkpoint {
    iteration: usize,
    integer: Vec<u8>,
}

#[allow(dead_code)]
impl Checkpoint {
    pub(crate) const fn new(iteration: usize, integer: Vec<u8>) -> Self {
        Self { iteration, integer }
    }

    pub(crate) fn data(self) -> (usize, Vec<u8>) {
        (self.iteration, self.integer)
    }
}

#[allow(dead_code)]
impl Integer {
    // #[inline]
    // pub fn iter(&self) -> std::slice::Iter<'_, Limb> {
    //     self.0.iter()
    // }

    // #[inline]
    // pub fn iter_rev(&self) -> std::iter::Rev<std::slice::Iter<'_, Limb>> {
    //     self.0.iter().rev()
    // }

    // #[inline]
    // pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Limb> {
    //     self.0.iter_mut()
    // }

    #[inline]
    pub(crate) fn reverse_into_integer(&self, output: &mut Self) {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to reverse an empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        let output_vec: &mut Vec<Limb> = &mut output.0;
        output_vec.clear();

        for limb in self.0.iter().rev() {
            output_vec.push(limb.reverse());
        }
        // at this point, the contents of the limbs and the order of the limbs are reversed
        // however, the digits are misaligned

        // safe because of the check at the top
        let skip_len: usize = 64 - unsafe { self.0.last().unwrap_unchecked() }.len();

        output_vec.push(Limb::new());

        let vec_beginning_ptr = output_vec.as_mut_ptr().cast::<u8>();
        let output_len_bytes = output_vec.len() * 64;

        let output_slice =
            unsafe { std::slice::from_raw_parts_mut(vec_beginning_ptr, output_len_bytes) };

        debug_assert_eq!(
            output_slice[(output_slice.len() - 64)..output_slice.len()],
            [0; 64]
        );

        let right_bound = output_slice.len() - (64 - skip_len);
        if !(right_bound - skip_len).is_multiple_of(64) {
            #[cfg(debug_assertions)]
            unreachable!("Reversal memory copy is not a multiple of 64 bytes");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }
        output_slice.copy_within(skip_len..right_bound, 0);

        let discarded = output_vec.pop();
        debug_assert_eq!(Limb::new(), discarded.unwrap());
    }

    fn reverse_interleave_x2(lhs: &mut u8x64, rhs: &mut u8x64) {
        // logically these are just bitwise ANDs; however, since the dst register
        // is the same as a src register, specifically VPXOR can have a much lower
        // latency and (somehow) doesn't use an FPU pipe
        // (this is based on data from Zen 4, but it is likely still true)
        let lhs_output = *lhs ^ rhs.reverse() << 4;
        let rhs_output = *rhs ^ lhs.reverse() << 4;

        *lhs = lhs_output;
        *rhs = rhs_output;
    }

    pub fn fused_reverse_add_asm_interleave(&mut self) -> bool {
        // instead of reversing into a seperate vector, reverse and pack into the original limb

        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to reverse and add empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        let total_limbs = self.0.len();

        self.0.push(Limb::new()); // padding

        let skip_len = 64 - self.0[total_limbs - 1].len();

        let limbs_ptr = self.0.as_mut_ptr() as *mut u8x64;
        let rev_ptr = &mut self.0[total_limbs - 1].0 as *mut u8x64;

        for i in 0..total_limbs.div_ceil(2) {
            unsafe {
                let left_limb_ptr = limbs_ptr.add(i);
                let right_limb_ptr = rev_ptr.sub(i);

                let lhs = &mut *left_limb_ptr;
                let rhs = &mut *right_limb_ptr;

                let lhs_output = *lhs ^ (rhs.reverse() << 4);
                let rhs_output = *rhs ^ (lhs.reverse() << 4);
                //let rhs_output =  Limb::ror4_galois(lhs_output).reverse();

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
            unsafe {
                let limb_ptr = &limb.0 as *const u8x64;

                let reversed_limb: u8x64 =
                    u8x64::from(_mm512_loadu_epi64(limb_ptr.byte_add(skip_len) as *const i64)) >> 4;
                limb.0 = (limb.0 << 4) >> 4;

                *limb = _mm512_add_epi64(limb.0.into(), reversed_limb.into()).into();

                *limb = _mm512_mask_add_epi8(
                    limb.0.into(),
                    overflowed as u64,
                    limb.0.into(),
                    _mm512_set1_epi64(1),
                )
                .into();
                overflowed = false;

                loop {
                    let carry_mask: __mmask64 =
                        _mm512_cmpge_epu8_mask(limb.0.into(), _mm512_set1_epi8(10));
                    if carry_mask & 0x8000_0000_0000_0000_u64 != 0 {
                        overflowed = true;
                    } else if carry_mask == 0 {
                        cold_path();
                        break;
                    }

                    ever_carried = true;

                    *limb = _mm512_mask_sub_epi8(
                        limb.0.into(),
                        carry_mask,
                        limb.0.into(),
                        _mm512_set1_epi8(10),
                    )
                    .into();
                    *limb = _mm512_mask_add_epi8(
                        limb.0.into(),
                        carry_mask << 1,
                        limb.0.into(),
                        _mm512_set1_epi8(1),
                    )
                    .into();
                }
            }
        }

        if overflowed {
            *unsafe { self.0.last_mut().unwrap_unchecked() } = Limb({
                let mut arr = [0u8; 64];
                arr[0] = 1;
                u8x64::from_array(arr)
            });
        } else {
            self.0.pop();
        }
        ever_carried
    }

    pub(crate) fn fused_reverse_add_asm_interleave_old(&mut self) -> bool {
        // instead of reversing into a seperate vector, reverse and pack into the original limb

        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to reverse and add empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        let total_limbs = self.0.len();

        let skip_len = 64 - unsafe { self.0.last().unwrap_unchecked() }.len();

        let limbs_ptr = self.0.as_mut_ptr() as *mut u8x64;
        let rev_ptr = &mut self.0.last_mut().unwrap().0 as *mut u8x64;

        for i in 0..total_limbs.div_ceil(2) {
            unsafe {
                let left_limb_ptr = limbs_ptr.add(i);
                let right_limb_ptr = rev_ptr.sub(i);

                Self::reverse_interleave_x2(&mut *left_limb_ptr, &mut *right_limb_ptr);
            }
        }

        let limb_count: usize = self.0.len();

        self.0.push(Limb::new()); // padding

        let mut ever_carried_byte: u8 = 0;
        let mut overflowed: u64 = 0;

        const ONE_VECTOR_B: u8x64 = u8x64::splat(1);
        const TEN_VECTOR_B: u8x64 = u8x64::splat(10);

        for (_, limb) in self
            .0
            .iter_mut()
            .enumerate()
            .take_while(|(idx, _)| idx < &limb_count)
        {
            // skip the last limb since it's padding

            unsafe {
                let limb_ptr = &limb.0 as *const u8x64;
                let carry_mask: u64;
                std::arch::asm!(
                    r#"
                    jmp 4f

                    5:
                    .byte   0
                    .byte   0
                    .byte   0
                    .byte   0
                    .byte   128
                    .byte   64
                    .byte   32
                    .byte   16

                    6:
                    .byte 15
                    .byte 15
                    .byte 15
                    .byte 15

                    4:
                    vmovdqu64 {rev}, [{limb_ptr} + {skip_len}]                                      # offset load the reversed limb, which is also interspersed with irrelevant limb data
                                                                                                    # AVX-512 lacks instructions for bytewise bit shifts... you're telling me that they
                                                                                                    # have the silicon space to implement this super niche multi-iteration 8x8 bit vector 
                                                                                                    # math operation that "computes an affine transformation in the Galois Field 2**8"
                                                                                                    # but they cant give me a bit shift that doesn't need to read from memory? 
                     vgf2p8affineqb{rev}, {rev}, qword ptr [rip + 5b]{{1to8}}, 0                    # shift the reversed limb left 4 bits using magic
                    
                    vpandd {limb}, {limb}, dword ptr [rip + 6b]{{1to16}}                            # mask off the high bits to only keep the low (useful) bits of the working limb
                                                                                                    # vpandq causes an illegal opcode exception? strange
                                                                                                    # use overflowed as a writemask so we can reuse one_zmm & not branch
                    vpaddb {limb}{{{overflowed}}}, {limb}, {one_b}                                  # add one if overflowed is set

                                                                                                    # using smaller mask sizes still clears the rest of the register
                    kxorb {overflowed}, {overflowed}, {overflowed}                                  # clear overflowed
                    kxorb {carry_mask_preserved}, {carry_mask_preserved}, {carry_mask_preserved}    # clear carry_mask_preserved
                    vpaddq {limb}, {limb}, {rev}                                                    # add the vectors together; use quadword variant because
                                                                                                    # the addition can't cross byte boundaries

                    2:                                                                              # carry processing loop
                    vpcmpub {carry_mask_kreg}, {limb}, {ten_b}, 5                                   # find the digits that are >= 10 and store them in carry_mask_kreg
                    korq {carry_mask_preserved}, {carry_mask_preserved}, {carry_mask_kreg}          # and carry_mask_kreg into carry_mask_preserved

                    ktestq {carry_mask_kreg}, {carry_mask_kreg}                                     # see if there are any carries
                    jz 3f                                                                           # if there are no carries, we are done
                    mov {ever_carried}, 1                                                           # there were carries because we didn't jump

                    vpsubb {limb}{{{carry_mask_kreg}}}, {limb}, {ten_b}                             # subtract 10 from those that triggered carries
                    kshiftlq {carry_mask_kreg}, {carry_mask_kreg}, 1                                # shift the mask left to use for carry propogation
                    vpaddb {limb}{{{carry_mask_kreg}}}, {limb}, {one_b}                             # propogate the carries
                    jmp 2b                                                                          # loop again because if there are no new carries it'll be caught earlier

                    3:                                                                              # done
                    "#,
                    limb = inout(zmm_reg) limb.0, // the limb that is getting modified
                    rev = out(zmm_reg) _, // scratch register for deinterleaving the reversed limb
                    limb_ptr = in(reg) limb_ptr, // pointer to the current limb
                    skip_len = in(reg) skip_len, // number of bytes to skip
                    overflowed = in(kreg) overflowed, // indicate if we need to add 1 to the next limb
                    one_b = in(zmm_reg) ONE_VECTOR_B, // a vector of 1s as 8-bit bytes
                    ten_b = in(zmm_reg) TEN_VECTOR_B, // a vector of 10s as 8-bit bytes
                    carry_mask_kreg = lateout(kreg) _, // tmp kreg for carry processing
                    carry_mask_preserved = lateout(kreg) carry_mask, // non_shifted kreg to determine if overflow needs to be set
                    ever_carried = inout(reg_byte) ever_carried_byte, // if the addition ever carried, this `Integer` cannot be a palindrome
                    options(nostack)
                );

                overflowed = (carry_mask & 0x8000_0000_0000_0000_u64 != 0).into();
            }
        }

        if overflowed == 0 {
            self.0.pop();
        } else {
            *unsafe { self.0.last_mut().unwrap_unchecked() } = Limb({
                let mut arr = [0u8; 64];
                arr[0] = 1;
                u8x64::from_array(arr)
            });
        }
        ever_carried_byte != 0
    }

    #[inline]
    pub(crate) fn show_differences(&self, rhs: &Self) -> String {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to show differences between empty integers");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
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
            #[cfg(debug_assertions)]
            unreachable!("Tried to check if empty integer has carries");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        for limb in &self.0 {
            if limb.has_carries() {
                return true;
            }
        }

        false
    }

    #[inline]
    fn process_carries(&mut self) -> bool {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to process carries in an empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        const ONE: Limb = {
            let mut array: [u8; 64] = [0u8; 64];
            array[0] = 1;
            Limb(u8x64::from_array(array))
        };

        let mut carry: bool = false;
        let mut ever_carried: bool = false;

        for limb in &mut self.0 {
            if carry {
                ever_carried = true;
                *limb = *limb + ONE;
            }
            (*limb, carry) = limb.process_carries();
        }
        if carry {
            ever_carried = true;
            self.0.push(ONE);
        }
        ever_carried
    }

    pub(crate) fn is_palindrome(&self, other: &Self) -> bool {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to check if an empty integer is a palindrome");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        self == other
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to get the length of an empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        unsafe { ((self.0.len() - 1) * 64) + self.0.last().unwrap_unchecked().len() }
    }

    pub(crate) fn pack(self) -> Self {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried pack an empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
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

        Self(output_vec)
    }

    #[must_use]
    pub fn unpack(self) -> Self {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to unpack an empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        let mut output: Vec<Limb> = Vec::with_capacity(self.0.len() * 2);

        for limb in &self.0 {
            let (low, high) = limb.unpack();
            if !low.is_empty() {
                output.push(low);
            }
            if !high.is_empty() {
                output.push(high);
            }
        }

        Integer(output)
    }

    #[inline]
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        let mut output: Vec<u8> = Vec::with_capacity(self.0.len() * 64);
        for limb in &self.0 {
            output.extend_from_slice(&limb.into_bytes());
        }
        output
    }

    #[must_use]
    #[inline]
    pub fn from_bytes(input: Vec<[u8; 64]>) -> Self {
        let mut output = Vec::with_capacity(input.len());
        for limb in &input {
            output.push(Limb::from_bytes(*limb));
        }
        Self(output)
    }

    #[inline]
    pub(crate) fn into_checkpoint(self, iteration: usize) -> Checkpoint {
        Checkpoint {
            iteration,
            integer: self.pack().into_bytes(),
        }
    }

    #[inline]
    pub(crate) fn from_checkpoint(input: Checkpoint) -> (Self, usize) {
        (
            Self::from_bytes(Self::chop(input.integer).unwrap()).unpack(),
            input.iteration,
        )
    }

    #[inline]
    fn add_into_self(&mut self, rhs: &Self) -> bool {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to add an empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        if self.0.len() != rhs.0.len() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to add two integers of different lengths");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        for (self_limb, other_limb) in self.0.iter_mut().zip(rhs.0.iter()) {
            *self_limb = *self_limb + *other_limb;
        }

        self.process_carries()
    }

    #[must_use]
    #[inline]
    pub fn chop(data: Vec<u8>) -> Option<Vec<[u8; 64]>> {
        data.chunks(64).map(|chunk| chunk.try_into().ok()).collect()
    }

    pub fn display_raw(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct LimbRawDisplay<'a>(&'a Limb);

        impl std::fmt::Display for LimbRawDisplay<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                // Delegate the formatting call to `Limb::display_raw`.
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

impl std::ops::Add for Integer {
    type Output = (Self, bool);

    fn add(self, other: Self) -> Self::Output {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to add an empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        if self.0.len() != other.0.len() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to add two integers of different lengths");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        // just add each limb to each limb
        // each digit will never overflow, so no special care needs to be taken with the adding

        let mut output_vec: Vec<Limb> = Vec::with_capacity(self.0.len());

        for (self_limb, other_limb) in self.0.iter().zip(other.0.iter()) {
            output_vec.push(*self_limb + *other_limb);
        }

        let mut output = Self(output_vec);
        let ever_carried = output.process_carries();
        (output, ever_carried)
    }
}

impl std::fmt::Debug for Integer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Integer(")?;
        for (i, limb) in self.0.iter().enumerate() {
            write!(f, "\n{i:}: {limb:#?}")?;
        }
        write!(f, "\n)")
    }
}

impl std::fmt::Display for Integer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output_string = String::new();
        for limb in self.0.iter().rev() {
            write!(output_string, "{limb}")?;
        }
        write!(f, "{}", output_string.trim_start_matches('0'))
    }
}

impl std::cmp::PartialEq for Integer {
    fn eq(&self, other: &Self) -> bool {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to compare an empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
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

impl std::cmp::Eq for Integer {}

/// A base-10 integer. The limbs grow left-to-right, so the most significant limb is the last one in the vector
#[macro_export]
macro_rules! integer {
    ($value:expr) => {{
        let value_str: &str = $value;
        let mut limbs: Vec<Limb> = Vec::new();
        let mut current_limb_digits: Vec<u8> = Vec::new();

        for digit in value_str.chars().rev() {
            if !digit.is_digit(10) {
                panic!("Invalid digit: {}", digit);
            }
            current_limb_digits.push(digit.to_digit(10).unwrap() as u8);

            if current_limb_digits.len() == 64 {
                let mut limb_bytes: [u8; 64] = [0; 64];
                for (i, &digit) in current_limb_digits.iter().enumerate() {
                    limb_bytes[i] = digit;
                }
                limbs.push(Limb(u8x64::from(limb_bytes)));
                current_limb_digits.clear();
            }
        }

        if !current_limb_digits.is_empty() {
            let mut limb_bytes: [u8; 64] = [0; 64];
            for (i, &digit) in current_limb_digits.iter().enumerate() {
                limb_bytes[i] = digit;
            }
            limbs.push(Limb(u8x64::from(limb_bytes)));
        }

        Integer(limbs)
    }};
}

#[derive(Debug, Clone, Copy)]
pub struct PackedLimb(u8x64);

pub(crate) type LimbPair = (Limb, Limb);

impl PackedLimb {
    #[inline]
    pub(crate) fn len(&self) -> std::num::NonZeroU32 {
        if self.0 == u8x64::splat(0) {
            #[cfg(debug_assertions)]
            unreachable!("Tried to get the length of an empty packed integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        let (low_limb, high_limb): LimbPair = (*self).into();
        let result = if high_limb.is_empty() {
            low_limb.len()
        } else {
            high_limb.len()
        };

        #[cfg(debug_assertions)]
        return std::num::NonZeroU32::new(result as u32).unwrap();

        #[cfg(not(debug_assertions))]
        unsafe {
            return std::num::NonZeroU32::new_unchecked(result as u32);
        }
    }
}

impl From<__m512i> for PackedLimb {
    #[inline]
    fn from(val: __m512i) -> Self {
        Self(u8x64::from(val))
    }
}

impl const From<u8x64> for PackedLimb {
    #[inline]
    fn from(val: u8x64) -> Self {
        Self(val)
    }
}

impl From<LimbPair> for PackedLimb {
    #[inline]
    fn from(val: LimbPair) -> Self {
        let limb2_shifted = unsafe { _mm512_slli_epi64(val.1.into(), 4) };
        let output: Self = unsafe { _mm512_or_si512(val.0.into(), limb2_shifted) }.into();
        debug_assert_eq!(
            output.0,
            unsafe { _mm512_xor_si512(val.0.into(), limb2_shifted) }.into()
        );

        output
    }
}

impl From<PackedLimb> for LimbPair {
    #[inline]
    fn from(val: PackedLimb) -> Self {
        unsafe {
            let limb2_mask: __m512i = u8x64::splat(0xF0).into();
            let limb2_shifted = _mm512_and_si512(val.0.into(), limb2_mask);

            let limb1 = _mm512_xor_si512(val.0.into(), limb2_shifted);
            let limb2 = _mm512_srli_epi64(limb2_shifted, 4);

            (limb1.into(), limb2.into())
        }
    }
}

#[derive(Debug)]
pub struct PackedInteger(Vec<PackedLimb>);

impl From<Integer> for PackedInteger {
    #[inline]
    fn from(val: Integer) -> Self {
        let mut limbs: Vec<PackedLimb> = Vec::with_capacity(val.0.len() / 2);

        for limb_pair in val.0.chunks(2) {
            match limb_pair.len() {
                2 => {
                    limbs.push((limb_pair[0], limb_pair[1]).into());
                }
                1 => {
                    limbs.push((limb_pair[0], Limb::new()).into());
                }
                _ => {
                    break;
                }
            }
        }

        Self(limbs)
    }
}

impl From<PackedInteger> for Integer {
    #[inline]
    fn from(val: PackedInteger) -> Self {
        if val.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to convert an empty packed integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        let mut limbs: Vec<Limb> = Vec::with_capacity(val.0.len() * 2);

        for packed_limbs in &val.0 {
            let (limb1, limb2) = LimbPair::from(*packed_limbs);
            limbs.push(limb1);
            limbs.push(limb2);
        }

        if unsafe { limbs.last().unwrap_unchecked() } == &Limb::new() {
            limbs.pop();
        }

        Self(limbs)
    }
}

#[allow(unused)]
impl PackedInteger {
    #[inline]
    fn len(&self) -> std::num::NonZeroUsize {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to get the length of an empty packed integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        let last_limb = unsafe { self.0.last().unwrap_unchecked() };
        let _last_limb_vec: __m512i = last_limb.0.into();

        todo!()
    }

    fn fused_reverse_add_asm(&mut self, reversed: &mut Integer) -> bool {
        let mut _ever_carried: bool = false;

        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to reverse and add empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        let skip_len = 64u32 - u32::from(unsafe { self.0.last().unwrap_unchecked() }.len());

        if reversed.0.len() < self.0.len() {
            debug_assert_eq!(self.0.len(), reversed.0.len() + 1);
            reversed.0.push(Limb::new());
            reversed.0.push(Limb::new());
        } else if reversed.0.len() == self.0.len() {
            reversed.0.push(Limb::new());
        }

        //reversed.0.clear();

        // for (idx, limb) in self.0.iter().rev().enumerate() {
        //     reversed.0[idx] = limb.reverse();
        // }

        // self.0
        //     .iter()
        //     .rev()
        //     .map(|s| Limb(s.0.reverse()))
        //     .collect_into(&mut reversed.0);

        if reversed.0.len() != (self.0.len() + 1) {
            #[cfg(debug_assertions)]
            unreachable!();

            #[cfg(not(debug_assertions))]
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }

        //reversed.0.push(Limb(u8x64::splat(0)));

        let ten_vector: __m512i = u8x64::splat(10).into();
        let one_vector: __m512i = u8x64::splat(1).into();
        let one_vector_64: __m512i = u64x8::splat(1).into();

        let mut ever_carried_byte: u8 = 0;
        let mut overflowed: u64 = 0;

        const ONE_LIMB: Limb = Limb({
            let mut arr = [0u8; 64];
            arr[0] = 1;
            u8x64::from_array(arr)
        });

        let rev_base_ptr = reversed.0.as_ptr().cast::<u8x64>();

        for (idx, limb) in self.0.iter_mut().enumerate() {
            let offset = (idx as u32 * 64) + skip_len;
            unsafe {
                let carry_mask: u64;
                std::arch::asm!(
                    r#"
                    # use overflowed as a writemask so we can reuse one_zmm
                    vpaddq {limb}{{{overflowed}}}, {limb}, {one_zmm_64}                             # add one if overflowed is set; we can use the quadword variant because addition will never cross byte boundaries
                    # using smaller mask sizes still clears the rest of the register
                    kxorb {overflowed}, {overflowed}, {overflowed}                                  # clear overflowed
                    kxorb {carry_mask_preserved}, {carry_mask_preserved}, {carry_mask_preserved}    # clear carry_mask_preserved
                    vpaddq {limb}, {limb}, [{base_ptr} + {offset:r}]                                  # add the vectors together; we can use the quadword variant again for the same reason

                    2: # carry processing loop
                    vpcmpub {carry_mask_kreg}, {limb}, {ten_zmm}, 5                                 # find the digits that are >= 10 and store them in carry_mask_kreg
                    korq {carry_mask_preserved}, {carry_mask_preserved}, {carry_mask_kreg}          # and carry_mask_kreg into carry_mask_preserved

                    ktestq {carry_mask_kreg}, {carry_mask_kreg}                                     # see if there are any carries
                    jz 3f                                                                           # if there are no carries, we are done
                    mov {ever_carried}, 1                                                           # there were carries because we didn't jump

                    vpsubb {limb}{{{carry_mask_kreg}}}, {limb}, {ten_zmm}                           # subtract 10 from those that triggered carries
                    kshiftlq {carry_mask_kreg}, {carry_mask_kreg}, 1                                # shift the mask left to use for carry propogation
                    vpaddb {limb}{{{carry_mask_kreg}}}, {limb}, {one_zmm}                           # propogate the carries
                    jmp 2b                                                                          # loop again because if there are no new carries it'll be caught earlier

                    3:                                                                              # done
                    "#,
                    base_ptr = in(reg) rev_base_ptr, // pointer to the reversed limb
                    // using a pointer lets us avoid loading it manually, since
                    // `vpaddb` can just take a memory address as an input
                    offset = in(reg) offset,
                    limb = inout(zmm_reg) limb.0, // the limb that is getting modified
                    overflowed = in(kreg) overflowed, // indicate if we need to add 1 to the next limb
                    one_zmm = in(zmm_reg) one_vector, // a vector of 1s
                    one_zmm_64 = in(zmm_reg) one_vector_64, // a vector of 1s, but as 64-bit quadwords
                    ten_zmm = in(zmm_reg) ten_vector, // a vector of 10s
                    carry_mask_kreg = lateout(kreg) _, // tmp kreg for carry processing
                    carry_mask_preserved = lateout(kreg) carry_mask, // non_shifted kreg to determine if overflow needs to be set
                    ever_carried = inout(reg_byte) ever_carried_byte, // if the addition ever carried, this `Integer` cannot be a palindrome
                    options(nostack)
                );

                overflowed = (carry_mask & 0x8000_0000_0000_0000u64 != 0).into();
            }
        }

        // if overflowed != 0 {
        //     self.0.push(ONE_LIMB);
        // }

        _ever_carried
    }
}

#[cfg(test)]
mod tests;
