//! Build script for vram-residency
//!
//! Compiles CUDA kernels and links them into the Rust binary.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels/vram_ops.cu");
    
    // Check if we should build CUDA (only in production mode)
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let build_cuda = env::var("VRAM_RESIDENCY_BUILD_CUDA")
        .unwrap_or_else(|_| "0".to_string()) == "1";
    
    if !build_cuda {
        println!("cargo:warning=Skipping CUDA build (set VRAM_RESIDENCY_BUILD_CUDA=1 to enable)");
        println!("cargo:warning=Tests will use mock VRAM allocator");
        return;
    }
    
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
    
    let status = Command::new(&nvcc)
        .arg("-c")
        .arg(&vram_ops_cu)
        .arg("-o")
        .arg(&vram_ops_o)
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg("-O3")
        .arg("--gpu-architecture=sm_60") // Compute Capability 6.0+
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
    println!("cargo:rustc-link-lib=dylib=cudart");
    
    println!("cargo:warning=CUDA kernels compiled successfully");
}
