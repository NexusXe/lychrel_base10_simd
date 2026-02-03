#![allow(clippy::module_inception)]

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
