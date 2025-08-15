use std::arch::x86_64::*;
use std::fmt::Write;
use std::simd::prelude::*;

/// A 64-byte vector of u8, representing a single "limb" of a large integer.
/// Each byte represents a single digit in base 10, with the least significant digit at index 0.
/// Thus, the digits are stored in reverse order.
#[derive(Clone, Copy)]
pub(crate) struct Limb(pub(crate) u8x64);

impl std::cmp::PartialEq for Limb {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl std::cmp::Eq for Limb {}

impl From<Limb> for __m512i {
    fn from(val: Limb) -> Self {
        val.0.into()
    }
}

impl const From<Limb> for u8x64 {
    fn from(val: Limb) -> Self {
        val.0
    }
}

impl From<__m512i> for Limb {
    fn from(val: __m512i) -> Self {
        Limb(val.into())
    }
}

impl const From<u8x64> for Limb {
    fn from(val: u8x64) -> Self {
        Limb(val)
    }
}

#[allow(dead_code)]
impl Limb {
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

    fn has_carries(&self) -> bool {
        let self_vector: __m512i = (*self).into();
        let compare: __m512i = __m512i::from(u8x64::splat(10));
        let carries = unsafe { _mm512_cmpge_epu8_mask(self_vector, compare) };
        carries != 0
    }

    fn process_carries(self) -> (Self, bool) {
        if !self.has_carries() {
            return (self, false);
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

    fn reverse(self) -> Self {
        let self_u8x64: u8x64 = self.into();
        self_u8x64.reverse().into()
    }

    fn len(&self) -> usize {
        let zero: __m512i = __m512i::from(u8x64::splat(0));
        let digit_mask = unsafe { _mm512_cmpeq_epu8_mask(self.0.into(), zero) };
        64 - (digit_mask.leading_ones() as usize)
    }

    pub(crate) fn pack(self, other: Self) -> Self {
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

    pub(crate) fn unpack(self) -> (Self, Self) {
        let self_vector: __m512i = self.into();
        let high_bytes: __m512i = u8x64::splat(0xF0).into();
        let low_bytes: __m512i = u8x64::splat(0x0F).into();

        let high_vector_shifted = unsafe { _mm512_and_epi64(self_vector, high_bytes) };
        let high_vector = unsafe { _mm512_srli_epi64(high_vector_shifted, 4) };

        let low_vector = unsafe { _mm512_and_epi64(self_vector, low_bytes) };

        (low_vector.into(), high_vector.into())
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
        Limb(self.0 + other.0)
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
pub(crate) struct Integer(pub(crate) Vec<Limb>);

#[allow(dead_code)]
impl Integer {
    pub(crate) fn reverse_into_integer(&self, output: &mut Integer) {
        #[cfg(debug_assertions)]
        if self.0.is_empty() {
            unreachable!("Cannot reverse an empty integer");
        }

        #[cfg(not(debug_assertions))]
        if self.0.is_empty() {
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }

        let output_vec: &mut Vec<Limb> = &mut output.0;
        output_vec.clear();

        for limb in self.0.iter().rev() {
            output_vec.push(limb.reverse());
        }
        // at this point, the contents of the limbs and the order of the limbs are reversed
        // however, the digits are misaligned

        let most_significant_limb: Limb = unsafe { *self.0.last().unwrap_unchecked() }; // safe because of the check at the top
        let skip_len: usize = 64 - most_significant_limb.len();

        // example with 4-digit limbs:
        // 123456 is represented as 6543 2100
        // reversed, we should expect 654321 which is represented as 1234 5600
        // plain reversal yields 0012 3456
        // to fix this, we can add a padding limb to the end and shift all of the data over:
        // 0012 3456 0000
        // 1234 5600

        output_vec.push(Limb::new());

        let vec_beginning_ptr = output_vec.as_mut_ptr() as *mut u8;
        let desired_view_ptr = unsafe { (vec_beginning_ptr).add(skip_len) };
        //debug_assert_eq!(unsafe{*vec_beginning_ptr}, 0);
        debug_assert_ne!(unsafe { *desired_view_ptr }, 0);
        unsafe {
            std::ptr::copy(desired_view_ptr, vec_beginning_ptr, (self.0.len()) * 64);
        }

        output_vec.pop();
    }

    fn process_carries(&mut self) {
        #[cfg(debug_assertions)]
        if self.0.is_empty() {
            unreachable!("Tried to process carries in an empty integer");
        }

        #[cfg(not(debug_assertions))]
        if self.0.is_empty() {
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }

        const ONE: Limb = {
            let mut array: [u8; 64] = [0u8; 64];
            array[0] = 1;
            Limb(u8x64::from_array(array))
        };

        let mut carry: bool = false;

        for limb in self.0.iter_mut() {
            if carry {
                *limb = *limb + ONE;
            }
            (*limb, carry) = limb.process_carries();
        }
        if carry {
            self.0.push(ONE);
        }
    }

    pub(crate) fn is_palindrome(&self, other: &Self) -> bool {
        #[cfg(debug_assertions)]
        if self.0.is_empty() {
            unreachable!("Tried to check if an empty integer is a palindrome");
        }

        #[cfg(not(debug_assertions))]
        if self.0.is_empty() {
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }

        self == other
    }

    pub(crate) fn len(&self) -> usize {
        #[cfg(debug_assertions)]
        if self.0.is_empty() {
            unreachable!("Tried to get the length of an empty integer");
        }

        #[cfg(not(debug_assertions))]
        if self.0.is_empty() {
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }

        unsafe { ((self.0.len() - 1) * 64) + self.0.last().unwrap_unchecked().len() }
    }

    pub(crate) fn pack(self) -> Self {
        #[cfg(debug_assertions)]
        if self.0.is_empty() {
            unreachable!("Tried pack an empty integer");
        }

        #[cfg(not(debug_assertions))]
        if self.0.is_empty() {
            unsafe {
                std::hint::unreachable_unchecked();
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
                _ => {break;}

            }
        }

        Integer(output_vec)
    }

    pub(crate) fn unpack(self) -> Self {
        #[cfg(debug_assertions)]
        if self.0.is_empty() {
            unreachable!("Tried to unpack an empty integer");
        }

        #[cfg(not(debug_assertions))]
        if self.0.is_empty() {
            unsafe {
                std::hint::unreachable_unchecked();
            }
        }

        let mut output: Vec<Limb> = Vec::with_capacity(self.0.len() * 2);

        for limb in self.0.iter() {
            let (low, high) = limb.unpack();
            output.push(low);
            output.push(high);
        }

        Integer(output)
    }
}

impl std::ops::Add for Integer {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        #[cfg(debug_assertions)]
        {
            if self.0.is_empty() {
                unreachable!("Tried to add an empty integer");
            }

            if self.0.len() != other.0.len() {
                unreachable!("Tried to add two integers of different lengths");
            }
        }

        #[cfg(not(debug_assertions))]
        {
            if self.0.is_empty() {
                unsafe {
                    std::hint::unreachable_unchecked();
                }
            }

            if self.0.len() != other.0.len() {
                unsafe {
                    std::hint::unreachable_unchecked();
                }
            }
        }

        // just add each limb to each limb
        // each digit will never overflow, so no special care needs to be taken with the adding

        let mut output_vec: Vec<Limb> = Vec::with_capacity(self.0.len());

        for (self_limb, other_limb) in self.0.iter().zip(other.0.iter()) {
            output_vec.push(*self_limb + *other_limb);
        }

        let mut output = Integer(output_vec);
        output.process_carries();
        output
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
        #[cfg(debug_assertions)]
        {
            if self.0.is_empty() {
                unreachable!("Tried to compare an empty integer");
            }

            if self.0.len() != other.0.len() {
                unreachable!("Tried to compare two integers of different lengths");
            }
        }

        #[cfg(not(debug_assertions))]
        {
            if self.0.is_empty() {
                unsafe {
                    std::hint::unreachable_unchecked();
                }
            }

            if self.0.is_empty() {
                unsafe {
                    std::hint::unreachable_unchecked();
                }
            }
        }

        for (a, b) in self.0.iter().zip(other.0.iter()) {
            if std::hint::likely(a != b) {
                return false;
            }
        }
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
