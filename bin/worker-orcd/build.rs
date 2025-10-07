// Build script for worker-orcd with CUDA support
//
// FIXES APPLIED:
// - 2025-10-04: Added find_cuda_root() to detect CUDA toolkit at /opt/cuda (not in PATH on CachyOS)
// - 2025-10-04: Set CMAKE_CUDA_COMPILER explicitly for CMake CUDA language support
// - 2025-10-04: Added stdc++ linking for C++ exception handling and RTTI
// - 2025-10-06: Walk all CUDA/C++ source files and emit cargo:rerun-if-changed for incremental builds
//
// -- Cascade

use std::env;
use std::fs;
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
        return BuildConfig { cuda: true, auto_detect_cuda: false };
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
        cuda: value
            .get("build")
            .and_then(|b| b.get("cuda"))
            .and_then(|c| c.as_bool())
            .unwrap_or(true),
        auto_detect_cuda: value
            .get("build")
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
    find_cuda_root().is_some()
}

/// Find CUDA toolkit root directory
///
/// FIX (2025-10-04 - Cascade): This function is critical for systems where nvcc
/// is not in PATH (e.g., CachyOS with CUDA installed via pacman at /opt/cuda).
/// CMake's CUDA language support requires either nvcc in PATH or explicit
/// CMAKE_CUDA_COMPILER setting. This function enables the latter.
fn find_cuda_root() -> Option<PathBuf> {
    // Check CUDA_PATH environment variable first
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let path = PathBuf::from(cuda_path);
        if path.exists() {
            println!("cargo:warning=CUDA detected via CUDA_PATH");
            return Some(path);
        }
    }

    // Check for nvcc in PATH and derive root from it
    if let Ok(output) = std::process::Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            if let Ok(nvcc_path) = String::from_utf8(output.stdout) {
                let nvcc_path = nvcc_path.trim();
                // nvcc is typically at /path/to/cuda/bin/nvcc
                if let Some(bin_dir) = Path::new(nvcc_path).parent() {
                    if let Some(cuda_root) = bin_dir.parent() {
                        println!("cargo:warning=CUDA detected via nvcc at {}", cuda_root.display());
                        return Some(cuda_root.to_path_buf());
                    }
                }
            }
        }
    }

    // Check common installation paths (critical for Arch/CachyOS)
    let cuda_paths = ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"];

    for path in &cuda_paths {
        let path_buf = PathBuf::from(path);
        if path_buf.exists() {
            println!("cargo:warning=CUDA detected at {}", path);
            return Some(path_buf);
        }
    }

    None
}

/// Walk a directory recursively and emit cargo:rerun-if-changed for all files matching extensions
fn register_source_files(dir: &Path, extensions: &[&str]) {
    if !dir.exists() {
        return;
    }

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        
        if path.is_dir() {
            // Skip build directories
            if path.file_name().and_then(|n| n.to_str()) == Some("build") {
                continue;
            }
            register_source_files(&path, extensions);
        } else if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if extensions.contains(&ext) {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
    }
}

fn build_with_cuda() {
    println!("cargo:warning=Building WITH CUDA support");

    let cuda_dir = PathBuf::from("cuda");

    // Find CUDA toolkit
    let cuda_root = find_cuda_root();

    // Build CUDA library with CMake
    let mut config = cmake::Config::new(&cuda_dir);
    config
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_TESTING", "OFF")
        .always_configure(false);  // Don't reconfigure CMake unless CMakeLists.txt changes

    // Set CUDA toolkit path if found
    // FIX (2025-10-04 - Cascade): CMAKE_CUDA_COMPILER is REQUIRED when nvcc is not in PATH.
    // Without this, CMake's CUDA language support fails with "No CMAKE_CUDA_COMPILER could be found"
    let cuda_path = if let Some(ref cuda_path) = cuda_root {
        println!("cargo:warning=Using CUDA toolkit at: {}", cuda_path.display());
        config.define("CUDAToolkit_ROOT", cuda_path.to_str().unwrap());

        // Explicitly set CUDA compiler path (critical for non-PATH installations)
        let nvcc_path = cuda_path.join("bin").join("nvcc");
        if nvcc_path.exists() {
            config.define("CMAKE_CUDA_COMPILER", nvcc_path.to_str().unwrap());
        }
        // TEAM FREE [Review]
        // Category: Build configuration
        // Hypothesis: nvcc_path.exists() check (line 192) passes but nvcc not executable; CMake configure fails with cryptic error.
        // Evidence: No check for execute permission; symlink or permission issue → exists() true but unusable.
        // Risk: Build failure on some systems (e.g., NFS mounts with noexec); hard to diagnose.
        // Confidence: Low
        // Next step: Add executable check or let CMake fail with clear error (current behavior acceptable).
        cuda_path.clone()
    } else {
        panic!(
            "CUDA toolkit not found. Please either:\n\
             1. Install CUDA toolkit and ensure nvcc is in PATH\n\
             2. Set CUDA_PATH environment variable\n\
             3. Set auto_detect_cuda = true in .llorch.toml to skip CUDA build when not available\n\
             4. Set cuda = false in .llorch.toml to disable CUDA support"
        );
    };

    let dst = config.build();

    // Add CUDA library path using the detected root (must be before linking)
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path.display());
    println!("cargo:rustc-link-search=native={}/lib", cuda_path.display());
    // Fallback to system library paths
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");

    // Link the static library
    let lib_path = dst.join("lib");
    let worker_cuda_lib = lib_path.join("libworker_cuda.a");

    println!("cargo:rustc-link-search=native={}", lib_path.display());

    // FIX (2025-10-04 - Cascade): Link our library first with whole-archive,
    // then link dependencies. The linker resolves symbols left-to-right.

    // Link our library with whole-archive to ensure all symbols are included
    // TEAM FREE [Review]
    // Category: Build configuration
    // Hypothesis: --whole-archive (line 224) forces all .o files into binary; if libworker_cuda.a has unused code, bloats binary size.
    // Evidence: Whole-archive prevents linker from dead-code elimination; necessary for C++ static init but increases size.
    // Risk: Larger binary (~10-30% bloat); slower link times; not a bug but suboptimal.
    // Confidence: Low
    // Next step: Profile binary size; if bloat significant, use selective symbol export instead of whole-archive.
    println!("cargo:rustc-link-arg=-Wl,--whole-archive");
    println!("cargo:rustc-link-arg={}", worker_cuda_lib.display());
    println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

    // Now link dependencies that our library needs
    println!("cargo:rustc-link-arg=-lstdc++");
    println!("cargo:rustc-link-arg=-lcudart");
    println!("cargo:rustc-link-arg=-lcudadevrt");
    println!("cargo:rustc-link-arg=-lcublas");

    // Register all CUDA/C++ source and header files for incremental builds
    // This ensures Cargo detects changes to individual files, not just directories
    let cuda_extensions = &["cu", "cpp", "h", "hpp", "cuh"];
    
    register_source_files(&cuda_dir.join("src"), cuda_extensions);
    register_source_files(&cuda_dir.join("include"), cuda_extensions);
    register_source_files(&cuda_dir.join("kernels"), cuda_extensions);
    
    // Also watch CMakeLists.txt files
    println!("cargo:rerun-if-changed=cuda/CMakeLists.txt");
    if cuda_dir.join("src/CMakeLists.txt").exists() {
        println!("cargo:rerun-if-changed=cuda/src/CMakeLists.txt");
    }
    if cuda_dir.join("kernels/CMakeLists.txt").exists() {
        println!("cargo:rerun-if-changed=cuda/kernels/CMakeLists.txt");
    }

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
