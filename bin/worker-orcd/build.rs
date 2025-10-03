use std::env;
use std::path::PathBuf;

fn main() {
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
}
