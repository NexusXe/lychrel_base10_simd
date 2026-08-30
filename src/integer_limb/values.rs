//! Selects the limb vector width for the target.
//!
//! An ordered cascade: the first matching arm wins, and the final `else` makes
//! it total, so exactly one width is always selected. Arms are in descending
//! width order, which is what determines the width a given target receives.

/// Length of a limb vector in `u8` lanes, i.e. decimal digits per limb.
pub const LV_LEN: usize = if cfg!(any(feature = "64-byte-limbs", target_feature = "avx512f")) {
    // 512-bit vectors
    64
} else if cfg!(target_feature = "avx2") {
    // 256-bit vectors
    32
} else if cfg!(any(
    target_feature = "sve",
    target_feature = "simd128",
    target_feature = "sse",
    target_feature = "neon",
    target_feature = "altivec"
)) {
    // 128-bit vectors. `altivec` covers powerpc64 and powerpc64le; it is
    // implied by vsx and by the power8/power9/power10 vector features.
    16
} else if cfg!(all(target_feature = "fxsr", target_pointer_width = "32")) {
    // 64-bit vectors, 32-bit pointer
    8
} else if cfg!(target_pointer_width = "32") {
    // 32-bit vectors, 32-bit pointer
    4
} else if cfg!(target_pointer_width = "16") {
    // 16-bit vectors, 16-bit pointer
    2
} else {
    // Reasonable fallback for zerocopy on targets with no SIMD at all.
    64
};

/// The wide scalar is derived from [`LV_LEN`] rather than chosen alongside it,
/// so the two cannot disagree. `WV_LEN` in the parent module divides `LV_LEN`
/// by this type's width, and a static assertion there requires `LimbVec` and
/// `WideVec` to be the same size, so the scalar must be `min(8, LV_LEN)` bytes.
/// An `LV_LEN` with no impl below is a compile error, which is the intent.
pub struct WideFor<const N: usize>;

/// Maps a limb width to the scalar its wide view is built from.
pub trait HasWideScalar {
    type Scalar;
}

impl HasWideScalar for WideFor<64> {
    type Scalar = u64;
}
impl HasWideScalar for WideFor<32> {
    type Scalar = u64;
}
impl HasWideScalar for WideFor<16> {
    type Scalar = u64;
}
impl HasWideScalar for WideFor<8> {
    type Scalar = u64;
}
impl HasWideScalar for WideFor<4> {
    type Scalar = u32;
}
impl HasWideScalar for WideFor<2> {
    type Scalar = u16;
}

pub type WideVecScalar = <WideFor<LV_LEN> as HasWideScalar>::Scalar;
