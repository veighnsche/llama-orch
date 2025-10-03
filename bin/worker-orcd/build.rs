use std::env;
use std::path::{Path, PathBuf};

fn main() {
    // Rebuild if config changes
    println!("cargo:rerun-if-changed=../../.llorch.toml");
    
    // Read local config (default: CUDA enabled)
    let config = read_local_config();
    let should_build_cuda = determine_cuda_build(&config);
    
    if should_build_cuda {
        build_with_cuda();
    } else {
        build_without_cuda();
    }
}

#[derive(Default)]
struct BuildConfig {
    cuda: bool,
    auto_detect_cuda: bool,
}

fn read_local_config() -> BuildConfig {
    let config_path = Path::new("../../.llorch.toml");
    
    if !config_path.exists() {
        // Default: CUDA enabled
        return BuildConfig {
            cuda: true,
            auto_detect_cuda: false,
        };
    }
    
    let content = match std::fs::read_to_string(config_path) {
        Ok(c) => c,
        Err(_) => return BuildConfig { cuda: true, auto_detect_cuda: false },
    };
    
    let value: toml::Value = match content.parse() {
        Ok(v) => v,
        Err(_) => return BuildConfig { cuda: true, auto_detect_cuda: false },
    };
    
    BuildConfig {
        cuda: value.get("build")
            .and_then(|b| b.get("cuda"))
            .and_then(|c| c.as_bool())
            .unwrap_or(true),
        auto_detect_cuda: value.get("build")
            .and_then(|b| b.get("auto_detect_cuda"))
            .and_then(|a| a.as_bool())
            .unwrap_or(false),
    }
}

fn determine_cuda_build(config: &BuildConfig) -> bool {
    // Explicit feature flag takes precedence
    if cfg!(feature = "cuda") {
        return true;
    }
    
    // Check local config
    if !config.cuda {
        return false;
    }
    
    // Auto-detect if enabled
    if config.auto_detect_cuda {
        detect_cuda()
    } else {
        // Default from config (true by default)
        config.cuda
    }
}

fn detect_cuda() -> bool {
    // Check for nvcc in PATH
    if std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok()
    {
        println!("cargo:warning=CUDA detected via nvcc");
        return true;
    }
    
    // Check CUDA_PATH environment variable
    if env::var("CUDA_PATH").is_ok() {
        println!("cargo:warning=CUDA detected via CUDA_PATH");
        return true;
    }
    
    // Check common installation paths
    let cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
    ];
    
    for path in &cuda_paths {
        if Path::new(path).exists() {
            println!("cargo:warning=CUDA detected at {}", path);
            return true;
        }
    }
    
    false
}

fn build_with_cuda() {
    println!("cargo:warning=Building WITH CUDA support");
    
    let cuda_dir = PathBuf::from("cuda");
    
    // Build CUDA library with CMake
    let dst = cmake::Config::new(&cuda_dir)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_TESTING", "OFF")
        .build();
    
    // Link the static library
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=worker_cuda");
    
    // Link CUDA runtime
    println!("cargo:rustc-link-lib=cudart");
    
    // Add CUDA library path
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        // Default CUDA installation paths
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    }
    
    // Rebuild if CUDA sources change
    println!("cargo:rerun-if-changed=cuda/src");
    println!("cargo:rerun-if-changed=cuda/include");
    println!("cargo:rerun-if-changed=cuda/kernels");
    println!("cargo:rerun-if-changed=cuda/CMakeLists.txt");
    
    // Enable cuda cfg for conditional compilation
    println!("cargo:rustc-cfg=feature=\"cuda\"");
}

fn build_without_cuda() {
    println!("cargo:warning=Building WITHOUT CUDA support (stub mode)");
    println!("cargo:warning=To enable CUDA:");
    println!("cargo:warning=  1. Copy .llorch.toml.example to .llorch.toml");
    println!("cargo:warning=  2. Set build.cuda = true");
    println!("cargo:warning=  3. Or use: cargo build --features cuda");
}
