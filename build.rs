fn main() {
    #[cfg(target_family = "windows")] {
    // This script runs before the main compilation.
    // It compiles the manifest and tells rustc to link it.
    embed_resource::compile("lychrel_base10_simd-manifest.rc", embed_resource::NONE)
        .manifest_optional()
        .unwrap();
    }
}
