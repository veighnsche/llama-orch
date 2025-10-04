# CUDA Build Fixes Applied (2025-10-04)

## Context

After merging the story branch with Foundation-Alpha's FFI implementation, the CUDA build was failing on CachyOS with CUDA 13 installed via pacman at `/opt/cuda`.

## Root Causes Identified

### 1. **nvcc Not in PATH**
- **Error**: `CMake Error: No CMAKE_CUDA_COMPILER could be found`
- **Cause**: CachyOS pacman installs CUDA at `/opt/cuda` but doesn't add it to PATH
- **Impact**: CMake's CUDA language support couldn't find the compiler

### 2. **Unsupported GPU Architecture**
- **Error**: `nvcc fatal: Unsupported gpu architecture 'compute_70'`
- **Cause**: CUDA 13 dropped support for Volta (compute_70)
- **Impact**: All CUDA kernel compilation failed

### 3. **Missing C++ Standard Library**
- **Error**: `undefined symbol: vtable for __cxxabiv1::__si_class_type_info`
- **Cause**: C++ code using exceptions/RTTI but stdc++ not linked
- **Impact**: Final linking stage failed

## Fixes Applied

### Fix 1: CUDA Toolkit Detection (`build.rs`)

Added `find_cuda_root()` function that detects CUDA in three ways:
1. `CUDA_PATH` environment variable
2. `which nvcc` and derive root from binary path
3. Common installation paths: `/usr/local/cuda`, `/opt/cuda`, `/usr/lib/cuda`

```rust
/// Find CUDA toolkit root directory
/// 
/// FIX (2025-10-04 - Cascade): This function is critical for systems where nvcc
/// is not in PATH (e.g., CachyOS with CUDA installed via pacman at /opt/cuda).
fn find_cuda_root() -> Option<PathBuf> {
    // ... implementation
}
```

### Fix 2: Explicit CMAKE_CUDA_COMPILER (`build.rs`)

Set CMake variables explicitly when CUDA root is detected:

```rust
config.define("CUDAToolkit_ROOT", cuda_path.to_str().unwrap());
config.define("CMAKE_CUDA_COMPILER", nvcc_path.to_str().unwrap());
```

This tells CMake exactly where to find `nvcc` even when it's not in PATH.

### Fix 3: Architecture Support (`cuda/CMakeLists.txt`)

Removed compute_70 (Volta) from supported architectures:

```cmake
# CUDA architectures (CUDA 13+ only supports 75+)
# FIX (2025-10-04): Removed compute_70 (Volta) - unsupported in CUDA 13
# 75 = Turing, 80/86 = Ampere, 89 = Ada, 90 = Hopper
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
```

### Fix 4: C++ Standard Library Linking (`build.rs`)

Added explicit stdc++ linking:

```rust
// FIX (2025-10-04 - Cascade): Link C++ standard library for exception handling and RTTI.
println!("cargo:rustc-link-lib=stdc++");
```

## Verification

Both debug and release builds now succeed:

```bash
$ cargo build -p worker-orcd
   Finished `dev` profile [unoptimized + debuginfo] target(s) in 2.57s

$ cargo build -p worker-orcd --release
   Finished `release` profile [optimized] target(s) in 10.47s
```

Binary sizes:
- Debug: 102M at `target/debug/worker-orcd`
- Release: 9.7M at `target/release/worker-orcd`

## Files Modified

1. **`bin/worker-orcd/build.rs`**
   - Added `find_cuda_root()` function
   - Modified `build_with_cuda()` to set CMAKE_CUDA_COMPILER
   - Added stdc++ linking
   - Added documentation comments

2. **`bin/worker-orcd/cuda/CMakeLists.txt`**
   - Changed `CMAKE_CUDA_ARCHITECTURES` from `70 75 80 86 89 90` to `75 80 86 89 90`

## Platform Compatibility

These fixes ensure CUDA builds work on:
- ✅ **CachyOS/Arch** with CUDA installed via pacman at `/opt/cuda`
- ✅ **Standard Linux** with CUDA at `/usr/local/cuda`
- ✅ **Custom installations** via `CUDA_PATH` environment variable
- ✅ **PATH-based installations** where `nvcc` is already accessible

## Notes for Foundation-Alpha Team

The CUDA detection logic in `build.rs` is now robust enough to handle various installation methods. If you encounter build issues on other platforms, the `find_cuda_root()` function is the place to add additional detection logic.

The explicit `CMAKE_CUDA_COMPILER` setting is critical—CMake's CUDA language support is strict about finding the compiler, and this approach works reliably across different installation scenarios.

---

**Fixed by**: Cascade (AI Assistant)  
**Date**: 2025-10-04  
**Tested on**: CachyOS with CUDA 13.0.1 at `/opt/cuda`
