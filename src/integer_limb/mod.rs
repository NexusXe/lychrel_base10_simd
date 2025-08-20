use std::arch::x86_64::*;
use std::fmt::Write;
#[cfg(not(debug_assertions))]
use std::hint::unreachable_unchecked;
use std::hint::{cold_path, likely};
use std::intrinsics::const_eval_select;
use std::simd::prelude::*;

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
                if arr1_64b[i] != arr2_64b[i] {
                    return false;
                } else {
                    i -= 1;
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
        Limb(val.into())
    }
}

impl const From<u8x64> for Limb {
    #[inline]
    fn from(val: u8x64) -> Self {
        Limb(val)
    }
}

#[allow(dead_code)]
impl Limb {
    #[inline]
    const fn new() -> Self {
        Limb(u8x64::splat(0))
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
        Limb(digits)
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
        if !self.has_carries() {
            return (*self, false);
        }

        const ONE_VEC: u8x64 = u8x64::splat(1);
        const COMPARE_VEC: u8x64 = u8x64::splat(10);
        let compare: __m512i = __m512i::from(COMPARE_VEC);
        let mut digits: __m512i = __m512i::from(self.0);
        let mut carries_past_last: bool = false;

        for _ in 0..64usize {
            let mut carries = unsafe { _mm512_cmpge_epu8_mask(digits, compare) };

            // "unneccesarily" repeating this 64 times seems worth it to make this branchless,
            // but adding this makes this function comically faster for larger Integers
            if carries == 0 {
                break;
            }

            digits = unsafe { _mm512_mask_sub_epi8(digits, carries, digits, compare) };
            carries_past_last |= carries & 0x8000000000000000u64 != 0; // the most we can ever carry is 1, so we track if this bit is ever set; it getting set multiple times is irrelevant to the final result
            carries <<= 1;

            // now add the carries to the next digit
            digits = unsafe { _mm512_mask_add_epi8(digits, carries, digits, ONE_VEC.into()) };
        }

        (digits.into(), carries_past_last)
    }

    #[inline]
    fn reverse(self) -> Self {
        let self_u8x64: u8x64 = self.into();
        self_u8x64.reverse().into()
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
        unsafe { _mm512_or_epi64(self_vector, other_shifted) }.into()
    }

    fn unpack(self) -> (Self, Self) {
        let self_vector: __m512i = self.into();
        let high_bytes: __m512i = u8x64::splat(0xF0).into();
        let low_bytes: __m512i = u8x64::splat(0x0F).into();

        let high_vector_shifted = unsafe { _mm512_and_epi64(self_vector, high_bytes) };
        let high_vector = unsafe { _mm512_srli_epi64(high_vector_shifted, 4) };

        let low_vector = unsafe { _mm512_and_epi64(self_vector, low_bytes) };

        (low_vector.into(), high_vector.into())
    }

    fn into_bytes(self) -> [u8; 64] {
        let self_simd: u8x64 = self.into();
        self_simd.into()
    }

    fn from_bytes(input: [u8; 64]) -> Self {
        Limb(u8x64::from(input))
    }

    #[inline]
    const fn is_empty(&self) -> bool {
        let zero: Limb = Limb(u8x64::splat(0));
        self == &zero
    }

    fn display_raw(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let digits: [u8; 64] = self.0.into();
        for i in digits.iter().rev() {
            write!(f, "{i}")?;
        }
        Ok(())
    }
}

impl Default for Limb {
    fn default() -> Self {
        Limb::new()
    }
}

impl std::ops::Add for Limb {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        //unsafe { _mm512_add_epi8(self.into(), other.into()) }.into()
        //Limb(self.0 + other.0) // should compile to be the same
        unsafe { _mm512_add_epi64(self.into(), other.into()) }.into() // actually is probably faster because each number will never overflow its byte boundary
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
        for i in digits.iter() {
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
    pub(crate) fn new(iteration: usize, integer: Vec<u8>) -> Self {
        Checkpoint { iteration, integer }
    }

    pub(crate) fn data(self) -> (usize, Vec<u8>) {
        (self.iteration, self.integer)
    }
}

#[allow(dead_code)]
impl Integer {
    #[inline]
    pub(crate) fn reverse_into_integer(&self, output: &mut Integer) {
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

        // method 1:
        // example with 4-digit limbs:
        // 123456 is represented as 6543 2100
        // reversed, we should expect 654321 which is represented as 1234 5600
        // plain reversal yields 0012 3456
        // to fix this, we can add a padding limb to the end and shift all of the data over:
        // 0012 3456 0000
        // 1234 5600 00

        // output_vec.push(Limb::new());
        // let vec_beginning_ptr = output_vec.as_mut_ptr() as *mut u8;
        // let desired_view_ptr = unsafe { (vec_beginning_ptr).add(skip_len) };
        // if skip_len != 0 {
        //     debug_assert_eq!(unsafe { *vec_beginning_ptr }, 0);
        // }
        // debug_assert_ne!(unsafe { *desired_view_ptr }, 0);

        // unsafe {
        //     std::ptr::copy_nonoverlapping(desired_view_ptr, vec_beginning_ptr, (self.0.len()) * 64);
        // }

        // method 2:
        output_vec.push(Limb::new());

        let vec_beginning_ptr = output_vec.as_mut_ptr() as *mut u8;
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

    #[allow(unused_variables)]
    fn fused_reverse_add(&mut self) -> bool {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to reverse and add empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        struct SpilloverDigits {
            data: u8x64,
            len: usize,
        }

        impl Iterator for SpilloverDigits {
            type Item = u8;

            fn next(&mut self) -> Option<Self::Item> {
                if self.len == 0 {
                    return None;
                }
                self.len -= 1;
                Some(self.data[self.len])
            }
        }

        let msl_digits: usize = unsafe { self.0.last().unwrap_unchecked() }.len();
        let spillover_length: usize = 64 - msl_digits;

        debug_assert_ne!(msl_digits, 0, "Most significant limb is empty");

        todo!()
    }

    #[inline]
    pub fn fused_reverse_add_asm(&mut self) -> bool {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to reverse and add empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
        }

        let skip_len = 64 - unsafe { self.0.last().unwrap_unchecked() }.len();

        let mut reversed_limbs: Vec<u8x64> = self.0.iter().rev().map(|s| s.0.reverse()).collect();

        if self.0.len() != reversed_limbs.len() {
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }

        reversed_limbs.push(u8x64::splat(0));

        let ten_vector: __m512i = u8x64::splat(10).into();
        let one_vector: __m512i = u8x64::splat(1).into();

        let mut ever_carried_byte: u8 = 0;
        let mut overflowed: u64 = 0;

        const ONE_LIMB: Limb = Limb({
            let mut arr = [0u8; 64];
            arr[0] = 1;
            u8x64::from_array(arr)
        });

        for (limb, rev_limb) in self
            .0
            .iter_mut()
            .zip(reversed_limbs[..reversed_limbs.len() - 1].iter())
        {
            let rev_ptr = rev_limb as *const u8x64;

            unsafe {
                let carry_mask: u64;
                std::arch::asm!(
                    r#"
                    # use overflowed as a writemask so we can reuse one_zmm
                    vpaddb {limb}{{{overflowed}}}, {limb}, {one_zmm} # add one if overflowed is set
                    kxorq {overflowed}, {overflowed}, {overflowed} # clear overflowed

                    # kxorq {carry_mask_kreg}, {carry_mask_kreg}, {carry_mask_kreg} # clear carry_mask_kreg
                    # knotq {carry_mask_kreg}, {carry_mask_kreg} # invert carry_mask_kreg the first time
                    kxorq {carry_mask_preserved}, {carry_mask_preserved}, {carry_mask_preserved} # clear carry_mask_preserved
                    vpaddb {limb}, {limb}, [{0} + rcx] # add the vectors together

                    2: # carry processing loop
                    vpcmpub {carry_mask_kreg}, {limb}, {ten_zmm}, 5 # find the digits that are >= 10 and store them in carry_mask_kreg
                    korq {carry_mask_preserved}, {carry_mask_preserved}, {carry_mask_kreg} # copy carry_mask_kreg to carry_mask_preserved

                    ktestq {carry_mask_kreg}, {carry_mask_kreg} # see if carry_tmp is zero

                    jz 3f # if there are no carries, we are done
                    mov {ever_carried}, 1 # since there are carries, set ever_carried

                    vpsubb {limb}{{{carry_mask_kreg}}}, {limb}, {ten_zmm} # subtract 10 from those that triggered carries
                    kshiftlq {carry_mask_kreg}, {carry_mask_kreg}, 1 # shift the mask left to use for carry propogation
                    vpaddb {limb}{{{carry_mask_kreg}}}, {limb}, {one_zmm} # propogate the carries
                    ktestq {carry_mask_kreg}, {carry_mask_kreg} # see if the overflow was the only carry
                    jnz 2b # if it wasn't, loop again because there might be new carries to process
                    
                    3: # done
                    "#,
                    in(reg) rev_ptr, // pointer to the reversed limb
                    // using a pointer lets us avoid loading it manually, since
                    // `vpaddb` can just take a memory address as an input
                    in("rcx") skip_len, // use rcx so that `skip_len` is also in `cl` for `shl`
                    limb = inout(zmm_reg) limb.0, // the limb that is getting modified
                    overflowed = in(kreg) overflowed, // indicate if we need to add 1 to the next limb
                    one_zmm = in(zmm_reg) one_vector, // a vector of 1s
                    ten_zmm = in(zmm_reg) ten_vector, // a vector of 10s
                    carry_mask_kreg = out(kreg) _, // tmp kreg for carry processing
                    carry_mask_preserved = out(kreg) carry_mask, // non_shifted kreg to determine if overflow needs to be set
                    ever_carried = inout(reg_byte) ever_carried_byte, // if the addition ever carried, this `Integer` cannot be a palindrome
                );
                if carry_mask & 0x8000000000000000u64 != 0 {
                    overflowed = 1;
                } else {
                    overflowed = 0;
                }
            }
        }

        if overflowed != 0 {
            self.0.push(ONE_LIMB);
        };
        ever_carried_byte != 0
    }

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
                if self_digit != other_digit {
                    write!(self_string, "\x1b[31m{self_digit}\x1b[0m").unwrap();
                    write!(other_string, "\x1b[31m{other_digit}\x1b[0m").unwrap();
                } else {
                    write!(self_string, "{self_digit}").unwrap();
                    write!(other_string, "{other_digit}").unwrap();
                }
            }
            writeln!(self_string, "]").unwrap();
            writeln!(other_string, "]").unwrap();
        }

        let mut output_string = String::with_capacity(self_string.len() + other_string.len());
        write!(output_string, "{}\n{}", self_string, other_string).unwrap();
        output_string
    }

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

        for limb in self.0.iter_mut() {
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
                    break;
                }
            }
        }

        Integer(output_vec)
    }

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

        for limb in self.0.iter() {
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
        for limb in self.0.iter() {
            output.extend_from_slice(&limb.into_bytes());
        }
        output
    }

    #[inline]
    pub fn from_bytes(input: Vec<[u8; 64]>) -> Integer {
        let mut output = Vec::with_capacity(input.len());
        for limb in input.iter() {
            output.push(Limb::from_bytes(*limb));
        }
        Integer(output)
    }

    #[inline]
    pub(crate) fn into_checkpoint(self, iteration: usize) -> Checkpoint {
        Checkpoint {
            iteration,
            integer: self.pack().into_bytes(),
        }
    }

    #[inline]
    pub(crate) fn from_checkpoint(input: Checkpoint) -> (Integer, usize) {
        (
            Integer::from_bytes(Integer::chop(input.integer).unwrap()).unpack(),
            input.iteration,
        )
    }

    #[inline]
    pub(crate) fn add_into_self(&mut self, rhs: &Self) -> bool {
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

    #[inline]
    pub fn chop(data: Vec<u8>) -> Option<Vec<[u8; 64]>> {
        data.chunks(64).map(|chunk| chunk.try_into().ok()).collect()
    }

    pub fn display_raw(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output_string = String::new();

        struct LimbRawDisplay<'a>(&'a Limb);

        impl<'a> std::fmt::Display for LimbRawDisplay<'a> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                // Delegate the formatting call to `Limb::display_raw`.
                self.0.display_raw(f)
            }
        }

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

        let mut output = Integer(output_vec);
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

#[cfg(test)]
mod tests;
