//! Build script for vram-residency
//!
//! Compiles CUDA kernels and links them into the Rust binary.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Determine if we should build with real CUDA
/// 
/// Auto-detects GPU and CUDA toolkit availability.
/// Can be overridden with VRAM_RESIDENCY_FORCE_MOCK=1 env var.
fn should_use_real_cuda() -> bool {
    // Allow explicit override to force mock mode
    if env::var("VRAM_RESIDENCY_FORCE_MOCK").is_ok() {
        return false;
    }
    
    // Check for nvidia-smi (GPU detection)
    let has_gpu = Command::new("nvidia-smi")
        .arg("--query-gpu=count")
        .arg("--format=csv,noheader")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    
    if !has_gpu {
        return false;
    }
    
    // Ensure CUDA paths are in environment for nvcc detection
    setup_cuda_paths();
    
    // Check for nvcc (CUDA toolkit)
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let has_nvcc = Command::new(&nvcc)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    
    has_gpu && has_nvcc
}

/// Setup CUDA paths in environment for build
fn setup_cuda_paths() {
    // Add common CUDA paths to PATH if they exist
    if let Ok(current_path) = env::var("PATH") {
        let mut paths = vec![current_path];
        
        if std::path::Path::new("/opt/cuda/bin").exists() {
            paths.insert(0, "/opt/cuda/bin".to_string());
        }
        if std::path::Path::new("/usr/local/cuda/bin").exists() {
            paths.insert(0, "/usr/local/cuda/bin".to_string());
        }
        
        env::set_var("PATH", paths.join(":"));
    }
}

/// Detect GPU compute capability and return appropriate sm_XX architecture
fn detect_gpu_architecture() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
        .ok()?;
    
    if !output.status.success() {
        return None;
    }
    
    let compute_cap = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    
    // Convert compute capability (e.g., "8.6") to architecture (e.g., "sm_86")
    let arch = compute_cap.replace('.', "");
    Some(format!("sm_{}", arch))
}

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels/vram_ops.cu");
    println!("cargo:rerun-if-changed=src/cuda_ffi/mock_cuda.c");
    
    // Auto-detect GPU and CUDA availability
    let should_build_cuda = should_use_real_cuda();
    
    if !should_build_cuda {
        println!("cargo:warning=Building with mock VRAM (no GPU/CUDA detected)");
        println!("cargo:warning=Tests will use mock VRAM allocator");
        
        // Build mock CUDA for testing
        build_mock_cuda();
        return;
    }
    
    println!("cargo:warning=GPU detected - building with real CUDA");
    println!("cargo:warning=Tests will run on real GPU VRAM");
    
    // Find nvcc compiler
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    
    // Check if nvcc is available
    let nvcc_check = Command::new(&nvcc)
        .arg("--version")
        .output();
    
    if nvcc_check.is_err() {
        println!("cargo:warning=nvcc not found, skipping CUDA build");
        println!("cargo:warning=Install CUDA Toolkit or set NVCC environment variable");
        return;
    }
    
    println!("cargo:warning=Building CUDA kernels with nvcc");
    
    // Output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_dir = PathBuf::from("cuda/kernels");
    
    // Compile vram_ops.cu
    let vram_ops_cu = cuda_dir.join("vram_ops.cu");
    let vram_ops_o = out_dir.join("vram_ops.o");
    
    // Detect GPU compute capability
    let gpu_arch = detect_gpu_architecture().unwrap_or_else(|| "sm_86".to_string());
    println!("cargo:warning=Compiling for GPU architecture: {}", gpu_arch);
    
    let status = Command::new(&nvcc)
        .arg("-c")
        .arg(&vram_ops_cu)
        .arg("-o")
        .arg(&vram_ops_o)
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg("-O3")
        .arg(format!("--gpu-architecture={}", gpu_arch))
        .status()
        .expect("Failed to compile CUDA kernel");
    
    if !status.success() {
        panic!("nvcc compilation failed");
    }
    
    // Create static library
    let lib_path = out_dir.join("libvram_cuda.a");
    
    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&lib_path)
        .arg(&vram_ops_o)
        .status()
        .expect("Failed to create static library");
    
    if !ar_status.success() {
        panic!("ar failed to create static library");
    }
    
    // Link the library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=vram_cuda");
    
    // Add CUDA library search paths
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    }
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    
    println!("cargo:rustc-link-lib=dylib=cudart");
    
    println!("cargo:warning=CUDA kernels compiled successfully");
}

/// Build mock CUDA implementation for testing
fn build_mock_cuda() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mock_cuda_c = PathBuf::from("src/cuda_ffi/mock_cuda.c");
    let mock_cuda_o = out_dir.join("mock_cuda.o");
    
    // Compile mock_cuda.c
    let status = Command::new("cc")
        .arg("-c")
        .arg(&mock_cuda_c)
        .arg("-o")
        .arg(&mock_cuda_o)
        .arg("-fPIC")
        .arg("-O2")
        .status()
        .expect("Failed to compile mock CUDA");
    
    if !status.success() {
        panic!("cc compilation of mock_cuda.c failed");
    }
    
    // Create static library
    let lib_path = out_dir.join("libmock_cuda.a");
    
    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&lib_path)
        .arg(&mock_cuda_o)
        .status()
        .expect("Failed to create mock CUDA library");
    
    if !ar_status.success() {
        panic!("ar failed to create mock CUDA library");
    }
    
    // Link the library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=mock_cuda");
    
    println!("cargo:warning=Mock CUDA compiled for testing");
}
