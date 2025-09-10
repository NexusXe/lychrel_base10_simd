#![allow(unused)]
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::alloc::{AllocError, Allocator, Layout, Global as GlobalAllocator};
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
    type WideVecScalar = u64;
}

#[cfg(all(
    not(any(
        target_feature = "avx512f",
        target_feature = "avx2",
        target_feature = "sve",
        feature = "64-byte-limbs"
    )),
    target_feature = "sse",
    target_feature = "neon",
    target_feature = "simd128"
))] // 128-bit vectors
mod values {
    pub const LV_LEN: usize = 16;
    type WideVecScalar = u64;
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
    type WideVecScalar = u64;
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
    type WideVecScalar = u32;
}

#[cfg(all(target_pointer_width = "16", not(feature = "64-byte-limbs")))]
mod values {
    pub const LV_LEN: usize = 2;
    type WideVecScalar = u16;
}

pub use values::*;

pub type LimbVecScalar = u8;
pub type LimbVec = Simd<LimbVecScalar, LV_LEN>;
type LimbVecMask = Mask<LimbVecScalar, LV_LEN>;

const WV_LEN: usize = LV_LEN / (WideVecScalar::BITS as usize / LimbVecScalar::BITS as usize);
type WideVec = Simd<WideVecScalar, WV_LEN>;

const fn assert_good_vec_sizes() {
    assert!(std::mem::size_of::<LimbVec>() == std::mem::size_of::<WideVec>());
}

const _: () = assert_good_vec_sizes();

#[cfg(not(target_family = "wasm"))]
mod huge_page_alloc {
    use std::mem::zeroed;
    use std::ptr;
    use std::alloc::{AllocError, Allocator, Layout};
    use std::alloc::Global as GlobalAllocator;

    #[cfg(target_os = "windows")]
    use windows::{
        Win32::{
            Foundation::{CloseHandle, GetLastError, HANDLE, LUID},
            Security::{
                AdjustTokenPrivileges, LUID_AND_ATTRIBUTES, LookupPrivilegeValueW,
                SE_PRIVILEGE_ENABLED, TOKEN_ADJUST_PRIVILEGES, TOKEN_PRIVILEGES, TOKEN_QUERY,
            },
            System::{
                Memory::{
                    GetLargePageMinimum, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT,
                    MEM_EXTENDED_PARAMETER, MEM_EXTENDED_PARAMETER_0, MEM_EXTENDED_PARAMETER_1,
                    MEM_LARGE_PAGES, MEM_RELEASE, MEM_RESERVE,
                    MemExtendedParameterAddressRequirements, MemExtendedParameterAttributeFlags,
                    PAGE_READWRITE, VirtualAlloc, VirtualAlloc2, VirtualFree,
                },
                Threading::{GetCurrentProcess, OpenProcessToken},
            },
        },
        core::{Result as WinResult, w},
    };

    #[derive(Clone, Copy)]
    pub struct HugePageAllocator;

    impl HugePageAllocator {
        #[cfg(target_os = "windows")]
        fn enable_memory_lock_privilege(process_handle: HANDLE) -> WinResult<()> {
            unsafe {
                let mut token_handle: HANDLE = zeroed();

                OpenProcessToken(
                    process_handle,
                    TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES,
                    &mut token_handle,
                )?;

                let mut luid: LUID = zeroed();

                LookupPrivilegeValueW(None, w!("SeLockMemoryPrivilege"), &mut luid)?;

                let token_privileges = TOKEN_PRIVILEGES {
                    PrivilegeCount: 1,
                    Privileges: [LUID_AND_ATTRIBUTES {
                        Luid: luid,
                        Attributes: SE_PRIVILEGE_ENABLED,
                    }],
                };

                AdjustTokenPrivileges(
                    token_handle,
                    false,
                    Some(&token_privileges),
                    size_of::<TOKEN_PRIVILEGES>() as u32,
                    None,
                    None,
                )?;

                let last_error = GetLastError();

                CloseHandle(token_handle)?;

                if last_error.is_err() {
                    return Err(last_error.to_hresult().into());
                }

                Ok(())
            }
        }

        #[cfg(target_family = "windows")]
        pub(crate) fn init() -> Result<Self, Box<dyn std::error::Error>> {
            let process_handle = unsafe { GetCurrentProcess() };
            Self::enable_memory_lock_privilege(process_handle)?;
            Ok(Self)
        }

        #[cfg(target_family = "unix")]
        pub(crate) fn init() -> Result<Self> {
            todo!()
        }

        #[cfg(all(not(target_family = "windows"), not(target_family = "unix")))]
        pub(crate) fn init() -> Result<Self> {
            unimplemented!()
        }
    }

    unsafe impl Allocator for HugePageAllocator {
        #[cfg(target_os = "windows")]
        fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
            const HUGE_PAGE_SIZE_BYTES: usize = 1024 * 1024 * 1024;

            #[cfg(debug_assertions)]
            {
                let large_page_size = unsafe { GetLargePageMinimum() };
                assert!(HUGE_PAGE_SIZE_BYTES.is_multiple_of(large_page_size));
            }

            let size = layout.size();

            if size == 0 {
                return Ok(ptr::NonNull::slice_from_raw_parts(layout.dangling(), 0));
            }

            unsafe {
                let large_page_size = GetLargePageMinimum();
                let aligned_size = (size.div_ceil(large_page_size) + 1) * large_page_size;
                //let alignment = layout.align().div_ceil(large_page_size) * large_page_size;
                //let alignment = HUGE_PAGE_SIZE_BYTES;
                let alignment = (layout.align().div_ceil(large_page_size)) * large_page_size;

                let mut requirements = MEM_ADDRESS_REQUIREMENTS {
                    LowestStartingAddress: zeroed(),
                    HighestEndingAddress: zeroed(),
                    Alignment: alignment,
                };

                let extended_parameter_1 = MEM_EXTENDED_PARAMETER {
                    Anonymous1: MEM_EXTENDED_PARAMETER_0 {
                        _bitfield: MemExtendedParameterAddressRequirements.0 as u64,
                    },
                    Anonymous2: MEM_EXTENDED_PARAMETER_1 {
                        Pointer: &mut requirements as *mut MEM_ADDRESS_REQUIREMENTS
                            as *mut std::os::raw::c_void,
                    },
                };

                let extended_parameter_2 = MEM_EXTENDED_PARAMETER {
                    Anonymous1: MEM_EXTENDED_PARAMETER_0 {
                        _bitfield: MemExtendedParameterAttributeFlags.0 as u64, // Specify MEM_LARGE_PAGES
                    },
                    //Anonymous2: MEM_EXTENDED_PARAMETER_1 { ULong64: 16u64 },
                    Anonymous2: MEM_EXTENDED_PARAMETER_1 { ULong64: 8u64 },
                };

                let allocation_size = aligned_size;
                //let allocation_size = aligned_size.div_ceil(HUGE_PAGE_SIZE_BYTES) * HUGE_PAGE_SIZE_BYTES;
                //eprintln!("Allocating {allocation_size} bytes");

                let ptr = VirtualAlloc2(
                    None,
                    None,
                    allocation_size,
                    MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                    PAGE_READWRITE.0,
                    Some(&mut [extended_parameter_1, extended_parameter_2]),
                );

                if ptr.is_null() {
                    let error = windows::core::Error::from_thread();
                    eprintln!("HugePageAlloc failed: {error}");
                    return Err(AllocError);
                }

                let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, aligned_size);

                Ok(ptr::NonNull::new(slice).unwrap())
            }
        }

        unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, _layout: Layout) {
            //eprintln!("Deallocating {:} bytes", _layout.size());
            let result = unsafe { VirtualFree(ptr.as_ptr() as *mut _, 0, MEM_RELEASE) };
            match result {
                Ok(()) => {}
                Err(error) => {
                    panic!("{error}");
                }
            }
        }
    }
}

#[cfg(not(target_family = "wasm"))]
pub(crate) use huge_page_alloc::*;

/// A 64-byte vector of u8, representing a single "limb" of a large integer.
/// Each byte represents a single digit in base 10, with the least significant digit at index 0.
/// Thus, the digits are stored in reverse order.
#[derive(Clone, Copy)]
pub(crate) struct Limb(pub(crate) LimbVec);

impl const std::cmp::PartialEq for Limb {
    fn eq(&self, other: &Self) -> bool {
        const fn eq_const(lhs: LimbVec, rhs: LimbVec) -> bool {
            let arr1 = lhs.to_array();
            let arr2 = rhs.to_array();
            let arr1_64b: [WideVecScalar; WV_LEN] = unsafe { transmute(arr1) };
            let arr2_64b: [WideVecScalar; WV_LEN] = unsafe { transmute(arr2) };
            let mut i: usize = WV_LEN;
            while i > 0 {
                if arr1_64b[i] == arr2_64b[i] {
                    i -= 1;
                } else {
                    return false;
                }
            }
            false
        }

        fn eq_rt(lhs: LimbVec, rhs: LimbVec) -> bool {
            lhs == rhs
        }

        const_eval_select((self.0, other.0), eq_const, eq_rt)
    }
}

impl std::cmp::Eq for Limb {}

#[cfg(all(
    target_feature = "avx512f",
    not(feature = "no-avx"),
    target_pointer_width = "64"
))]
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

#[cfg(all(
    target_feature = "avx512f",
    not(feature = "no-avx"),
    target_pointer_width = "64"
))]
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
    pub(crate) const fn new() -> Self {
        Self(LimbVec::splat(0))
    }

    pub(crate) fn new_from_value(value: u128) -> Self {
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
        Limb(self.0.reverse())
    }

    #[inline]
    fn len(&self) -> usize {
        let zeros = LimbVec::splat(0);
        let eq_mask = self.0.simd_ne(zeros);
        let bitmask = eq_mask.to_bitmask();
        (LV_LEN - (bitmask.leading_zeros() as usize - (64 - LV_LEN)))
    }

    fn pack(self, other: Self) -> Self {
        debug_assert!(!self.has_carries());
        debug_assert!(!other.has_carries());

        debug_assert_eq!(LimbVec::splat(0), self.0 & LimbVec::splat(0xF0));
        debug_assert_eq!(LimbVec::splat(0), other.0 & LimbVec::splat(0xF0));

        unsafe {
            let other_u64: WideVec = transmute(other.0);
            let other_shifted: LimbVec = transmute(other_u64 << 4);
            Limb(self.0 ^ other_shifted)
        }
    }

    #[inline]
    fn unpack(&self) -> (Self, Self) {
        (Limb((self.0 << 4) >> 4), Limb(self.0 >> 4))
    }

    const fn into_bytes(self) -> [LimbVecScalar; LV_LEN] {
        self.0.to_array()
    }

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

    // /// A "Simple Example" for "byte-granularity bit rotate"
    // /// As far as I'm concerned, this is a function that uses arcane magic to rotate each byte by 4 bits
    // #[allow(dead_code)]
    // #[inline]
    // pub(crate) fn ror4_galois(input: LimbVec) -> LimbVec {
    //     // TODO: make portable..?
    //     let input_vec: __m512i = input.into();
    //     unsafe {
    //         let rorb4: __m512i = _mm512_set1_epi64(0x1020408001020408);
    //         _mm512_gf2p8affine_epi64_epi8(input_vec, rorb4, 0x0).into()
    //     }
    // }
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
            Limb(transmute::<WideVec, LimbVec>(output_64))
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
pub struct Integer<T: Allocator + Clone + Copy>(pub(crate) Vec<Limb, T>);

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
impl<T: Allocator + Clone + Copy> Integer<T> {
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
    pub(crate) fn reverse_into_integer(&self, output: &mut Integer<GlobalAllocator>) {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to reverse an empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
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

    fn reverse_interleave_x2(lhs: &mut LimbVec, rhs: &mut LimbVec) {
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
        use std::ptr::read_unaligned;
        // TODO: make portable...?
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

        let skip_len = LV_LEN - self.0[total_limbs - 1].len();

        let limbs_ptr = self.0.as_mut_ptr() as *mut LimbVec;
        let rev_ptr = &mut self.0[total_limbs - 1].0 as *mut LimbVec;

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
                let limb_ptr = &limb.0 as *const LimbVec;

                let reversed_limb: LimbVec = read_unaligned(limb_ptr.byte_add(skip_len)) >> 4;

                limb.0 = (limb.0 << 4) >> 4;

                *limb = Limb(transmute::<WideVec, LimbVec>(
                    transmute::<LimbVec, WideVec>(limb.0)
                        + transmute::<LimbVec, WideVec>(reversed_limb),
                ));

                // target-cpu = x86-64-v2:  287.6 sec
                // target-cpu = x86-64-v3:  266.1 sec
                // target-cpu = x86-64-v4:  15.1 sec
                // target-cpu = znver5:     14.5 sec
                #[cfg(all(
                    target_feature = "avx512f",
                    not(feature = "no-avx"),
                    target_pointer_width = "64"
                ))]
                {
                    *limb = _mm512_mask_add_epi64(
                        limb.0.into(),
                        overflowed as u8,
                        limb.0.into(),
                        _mm512_set1_epi64(1),
                    )
                    .into();
                }

                #[cfg(any(
                    not(target_feature = "avx512f"),
                    feature = "no-avx",
                    not(target_pointer_width = "64")
                ))]
                if overflowed {
                    limb.0.as_mut_array()[0] += 1;
                }

                overflowed = false;

                #[cfg(all(
                    target_feature = "avx512bw",
                    not(feature = "no-avx"),
                    target_pointer_width = "64"
                ))]
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

                #[cfg(any(
                    not(target_feature = "avx512f"),
                    feature = "no-avx",
                    not(target_pointer_width = "64")
                ))]
                loop {
                    let carry_mask = limb.0.simd_ge(LimbVec::splat(10));
                    if carry_mask.test(LV_LEN - 1) {
                        overflowed = true;
                    } else if !carry_mask.any() {
                        cold_path();
                        break;
                    }

                    ever_carried = true;

                    for (idx, byte) in limb.0.as_mut_array().iter_mut().enumerate() {
                        if carry_mask.test(idx) {
                            *byte -= 10;
                        }

                        if carry_mask.shift_elements_right::<1usize>(false).test(idx) {
                            *byte += 1;
                        }
                    }
                }
            }
        }

        if overflowed {
            unsafe { *(rev_ptr.add(1) as *mut u8) = 1 }; // this limb is already zeroed for padding, so just set one byte
        } else {
            self.0.pop();
        }
        ever_carried
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

        unsafe { ((self.0.len() - 1) * LV_LEN) + self.0.last().unwrap_unchecked().len() }
    }

    pub(crate) fn pack(self) -> Integer<GlobalAllocator> {
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

        Integer::<GlobalAllocator>(output_vec)
    }

    #[must_use]
    pub fn unpack(self, allocator: T) -> Integer<T> {
        if self.0.is_empty() {
            #[cfg(debug_assertions)]
            unreachable!("Tried to unpack an empty integer");

            #[cfg(not(debug_assertions))]
            unsafe {
                unreachable_unchecked();
            }
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
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        let mut output: Vec<LimbVecScalar> = Vec::with_capacity(self.0.len() * LV_LEN);
        for limb in &self.0 {
            output.extend_from_slice(&limb.into_bytes());
        }
        output
    }

    #[must_use]
    #[inline]
    pub fn from_bytes(input: Vec<[LimbVecScalar; LV_LEN]>, allocator: T) -> Integer<T> {
        let mut output = Vec::with_capacity_in(input.len(), allocator);
        for limb in &input {
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
    pub(crate) fn from_checkpoint(input: Checkpoint, allocator: T) -> (Integer<T>, usize) {
        let chopped_data = Integer::<T>::chop(input.integer).unwrap();
        let packed_integer = Integer::from_bytes(chopped_data, allocator);
        let integer = packed_integer.unpack(allocator);
        (integer, input.iteration)
    }

    #[must_use]
    #[inline]
    pub fn chop(data: Vec<u8>) -> Option<Vec<[LimbVecScalar; LV_LEN]>> {
        data.chunks(LV_LEN)
            .map(|chunk| chunk.try_into().ok())
            .collect()
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

impl<T: Allocator + Clone + Copy> std::cmp::Eq for Integer<T> {}

/// A base-10 integer. The limbs grow left-to-right, so the most significant limb is the last one in the vector
#[macro_export]
macro_rules! integer {
    ($value:expr) => {{
        let value_str: &str = $value;
        let mut limbs: Vec<Limb> =
            Vec::with_capacity(value_str.len() / $crate::integer_limb::LV_LEN + 1);
        let mut current_limb_digits: Vec<u8> = Vec::new();

        for digit in value_str.chars().rev() {
            if !digit.is_digit(10) {
                panic!("Invalid digit: {}", digit);
            }
            current_limb_digits.push(digit.to_digit(10).unwrap() as u8);

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
