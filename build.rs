fn main() {
    #[cfg(target_family = "windows")]
    if std::env::var("PROFILE").unwrap() == "release" {
        // Embed Windows 11 manifest for no particular reason
        embed_resource::compile("lychrel_base10_simd-manifest.rc", embed_resource::NONE)
            .manifest_optional()
            .unwrap();
    }
}
