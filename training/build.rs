fn main() {
    #[cfg(feature = "cuda")]
    {
        let mut build = cc::Build::new();
        build.cuda(true);

        // Always target A100 (sm_80) and Ada/RTX 4090 (sm_89)
        build.flag("-gencode=arch=compute_80,code=sm_80");
        build.flag("-gencode=arch=compute_89,code=sm_89");

        // Blackwell targets require CUDA 12.8+ -- detect nvcc version
        let nvcc_version = std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        // Parse "release X.Y" from nvcc output
        let cuda_major_minor: Option<(u32, u32)> = nvcc_version
            .lines()
            .find(|l| l.contains("release"))
            .and_then(|l| {
                let after = l.split("release ").nth(1)?;
                let ver = after.split(',').next()?;
                let mut parts = ver.split('.');
                let major = parts.next()?.trim().parse().ok()?;
                let minor = parts.next()?.trim().parse().ok()?;
                Some((major, minor))
            });
        if let Some((major, minor)) = cuda_major_minor {
            eprintln!("Detected CUDA {}.{}", major, minor);
            if major > 12 || (major == 12 && minor >= 8) {
                build.flag("-gencode=arch=compute_100,code=sm_100"); // Blackwell data center
                build.flag("-gencode=arch=compute_120,code=sm_120"); // Blackwell consumer
            }
        }

        build
            .flag("-O3")
            .flag("--use_fast_math") // HW intrinsics for exp/log/sigmoid in elementwise kernels
            .flag("--ftz=false") // Preserve denorms (i32 token IDs stored in f32 buffers)
            .file("csrc/ssm_scan.cu")
            .file("csrc/elementwise_ops.cu")
            .file("csrc/cublas_ops.cu")
            .compile("nm_kernels");

        println!("cargo:rerun-if-changed=csrc/");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
    }
}
