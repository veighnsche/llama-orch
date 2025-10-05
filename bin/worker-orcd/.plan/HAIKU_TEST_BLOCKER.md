# Haiku Test Blocker - CUDA Memory Corruption

**Date**: 2025-10-05  
**Status**: 🔴 BLOCKED  
**Priority**: CRITICAL  

## Summary

The haiku generation test is blocked by a critical CUDA memory corruption issue. After successfully loading the first 1-2 tensors (~260MB), all subsequent `cudaMalloc` calls fail with "misaligned address" error, indicating CUDA has entered a bad state.

## Symptoms

1. ✅ First tensor loads successfully: `output.weight` (Q8_0, 259MB)
2. ✅ Sometimes second tensor loads
3. ❌ All subsequent `cudaMalloc` calls fail with "misaligned address"
4. ❌ Error cascades - CUDA is in corrupted state
5. ❌ Only 2/291 tensors load before failure

## Error Pattern

```
  [1/291] Loaded output.weight (259.65625 MB, type=Q8_0)  ← SUCCESS
❌ cudaMalloc failed: misaligned address (size=1792 bytes)  ← FAILURE
❌ cudaMalloc failed: misaligned address (size=256 bytes)   ← CASCADE
❌ cudaMalloc failed: misaligned address (size=78848 bytes) ← CASCADE
... (all subsequent allocations fail)
```

## Root Cause Analysis

### Conflicting Code Paths Discovered

**YES - You were right!** There are multiple half-baked quantization implementations fighting each other:

1. **Rust GPU Dequant** (`src/cuda/gguf_dequant.rs`) - NEW, uses CUDA kernels
2. **Rust CPU Dequant** (`worker-gguf/src/`) - DELETED (we removed Q5_0, Q6_K, Q8_0)
3. **C++ Loader** (`cuda/src/model/qwen_weight_loader.cpp`) - OLD, loads quantized data WITHOUT dequant

### The Conflict

The C++ `QwenWeightLoader::load_tensor_to_vram()` (line 175-220) has this warning:

```cpp
// WARNING: This is loading quantized weights (Q4_K_M) directly without dequantization!
// This will cause NaN and garbage values. Need to implement dequantization.
fprintf(stderr, "⚠️  Loading %s: type=%u, size=%zu bytes (QUANTIZED - NOT DEQUANTIZED!)\n",
```

### Likely Cause

The "misaligned address" error from `cudaMalloc` is **extremely unusual** - `cudaMalloc` handles alignment automatically. This suggests:

1. **Previous CUDA error not cleared** - A kernel launch failed but error wasn't checked
2. **Memory corruption** - The Q8_0 dequant kernel is writing out of bounds
3. **Invalid kernel launch** - Kernel parameters are wrong, causing GPU fault
4. **Missing symbols** - CUDA kernels not properly linked

## What We've Tried

### ✅ Completed
1. Created Q4_K, Q5_0, Q6_K, Q8_0 CUDA kernels
2. Added FFI bindings (`gguf_dequant.cuh`)
3. Created Rust wrappers (`gguf_dequant.rs`)
4. Updated weight loader to use GPU dequant
5. Removed old CPU dequant code
6. Added error recovery (skip failed tensors)
7. Added CUDA error clearing (`cudaGetLastError()`)
8. Forced CMake rebuild

### ❌ Still Failing
- CUDA enters bad state after first tensor
- "misaligned address" error cascades
- Only 2/291 tensors load

## Debugging Steps Needed

### 1. Verify Kernel Compilation
```bash
# Check if symbols exist
nm target/debug/build/worker-orcd-*/out/build/libworker_cuda.a | grep dequant

# Should see:
# q4k_dequant_launch
# q5_0_dequant_launch
# q6_k_dequant_launch
# q8_0_dequant_launch
```

### 2. Test Kernels in Isolation
Create standalone CUDA test:
```cpp
// test_q8_0_kernel.cu
int main() {
    // Allocate test data
    uint8_t* d_input;
    half* d_output;
    cudaMalloc(&d_input, 34 * 1000); // 1000 blocks
    cudaMalloc(&d_output, 32000 * sizeof(half));
    
    // Launch kernel
    q8_0_dequant_launch(d_output, d_input, 32000, 0);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    printf("Kernel error: %s\n", cudaGetErrorString(err));
    
    cudaFree(d_input);
    cudaFree(d_output);
}
```

### 3. Add Kernel Error Checking
In `gguf_dequant.rs`, after each kernel launch:
```rust
// Synchronize to catch kernel errors
extern "C" { fn cudaDeviceSynchronize() -> c_int; }
let sync_err = cudaDeviceSynchronize();
if sync_err != 0 {
    eprintln!("Kernel execution failed: {}", sync_err);
}
```

### 4. Check Kernel Launch Parameters
Verify block/grid dimensions are valid:
- Q8_0: grid=(num_blocks), block=(32)
- Q6_K: grid=(num_blocks), block=(256)
- etc.

### 5. Memory Bounds Check
Add bounds checking in kernels:
```cuda
__global__ void q8_0_dequant_kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;  // ← Add this
    // ... rest of kernel
}
```

## Recommended Fix Strategy

### Option A: Simplify (FASTEST - 2 hours)
1. **Disable GPU dequant temporarily**
2. Use C++ loader with CPU dequant
3. Get haiku working first
4. Fix GPU dequant later

### Option B: Debug GPU Dequant (4-6 hours)
1. Create standalone kernel tests
2. Verify each kernel works in isolation
3. Add comprehensive error checking
4. Fix memory corruption
5. Re-integrate

### Option C: Hybrid Approach (3 hours)
1. Keep Q4_K on CPU (most common format)
2. Only use GPU for Q5_0, Q6_K, Q8_0
3. Add fallback to CPU if GPU fails
4. Get test passing, optimize later

## Files Involved

### CUDA Kernels
- `cuda/kernels/q4_k_dequant.cu`
- `cuda/kernels/q5_0_dequant.cu`
- `cuda/kernels/q6_k_dequant.cu`
- `cuda/kernels/q8_0_dequant.cu`
- `cuda/kernels/gguf_dequant.cuh`

### Rust FFI
- `src/cuda/gguf_dequant.rs` (378 lines)
- `src/cuda/weight_loader.rs` (load_tensor_gpu)
- `src/cuda/ffi.rs`

### C++ Side
- `cuda/src/ffi_weight_loading.cpp`
- `cuda/src/model/qwen_weight_loader.cpp`

### Build
- `cuda/CMakeLists.txt`
- `build.rs`

## Impact

**BLOCKS**:
- ❌ Haiku generation test
- ❌ Real GPU inference
- ❌ M0 milestone completion
- ❌ Model loading for any quantized format

**WORKS**:
- ✅ CUDA context initialization
- ✅ First tensor loads (proves CUDA works)
- ✅ Kernel compilation
- ✅ FFI bindings

## Next Steps

**IMMEDIATE** (to unblock haiku test):
1. Try Option A - disable GPU dequant, use CPU fallback
2. Get ONE haiku generated (proves pipeline works)
3. Come back to GPU dequant debugging

**AFTER HAIKU WORKS**:
1. Create isolated kernel tests
2. Debug memory corruption
3. Add comprehensive error checking
4. Re-enable GPU dequant

## Time Estimate

- **Option A (CPU fallback)**: 2 hours → Haiku working tonight
- **Option B (Fix GPU dequant)**: 4-6 hours → Haiku tomorrow
- **Option C (Hybrid)**: 3 hours → Haiku working, partial GPU

## Recommendation

**Use Option A**: Get the haiku working with CPU dequant FIRST. This proves:
- ✅ Model loading works
- ✅ Tokenizer works
- ✅ Inference pipeline works
- ✅ SSE streaming works

Then debug GPU dequant separately without blocking the milestone.

---

**Status**: Awaiting decision on approach  
**Blocker**: CUDA memory corruption after first tensor  
**Impact**: Cannot load model, cannot generate haiku
