fn main() {
    // HIP/ROCm backend. Checked first so `--features hip` wins if both are on.
    #[cfg(feature = "hip")]
    {
        let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "hipcc".to_string());

        let mut build = cc::Build::new();
        build.compiler(&hipcc).cpp(true);

        // Target the installed GPU. gfx1201 is RDNA4 (Radeon AI PRO R9700).
        // Override with e.g. HIP_ARCH=gfx1100 to build for a different card.
        let arch = std::env::var("HIP_ARCH").unwrap_or_else(|_| "gfx1201".to_string());
        build.flag(&format!("--offload-arch={}", arch));

        // The kernels hardcode a 32-lane warp (see the static_assert in
        // ssm_scan.cu). Wave32 is the RDNA default, but state it explicitly so a
        // toolchain default flip becomes a build error rather than silent
        // corruption in the shuffle reductions.
        build.flag("-mno-wavefrontsize64");

        build
            .flag("-O3")
            .flag("-ffast-math") // was --use_fast_math; HW intrinsics for exp/log/sigmoid
            .file("csrc_hip/ssm_scan.cu")
            .file("csrc_hip/elementwise_ops.cu")
            .file("csrc_hip/cublas_ops.cu")
            .compile("nm_kernels");

        println!("cargo:rerun-if-changed=csrc_hip/");
        println!("cargo:rerun-if-env-changed=HIPCC");
        println!("cargo:rerun-if-env-changed=HIP_ARCH");
        println!("cargo:rustc-link-lib=amdhip64"); // HIP runtime, replaces cudart
        println!("cargo:rustc-link-lib=hipblas"); // replaces cublas
    }

    // NVIDIA backend. Retained for reference; csrc_hip/ is the active port.
    #[cfg(all(feature = "cuda", not(feature = "hip")))]
    {
        let mut build = cc::Build::new();
        build.cuda(true);

        // Always target A100 (sm_80), Ada/RTX 4090 (sm_89), and H100 (sm_90)
        build.flag("-gencode=arch=compute_80,code=sm_80");
        build.flag("-gencode=arch=compute_89,code=sm_89");
        build.flag("-gencode=arch=compute_90,code=sm_90");

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
            .file("csrc_cuda/ssm_scan.cu")
            .file("csrc_cuda/elementwise_ops.cu")
            .file("csrc_cuda/cublas_ops.cu")
            .compile("nm_kernels");

        println!("cargo:rerun-if-changed=csrc_cuda/");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
    }
}
