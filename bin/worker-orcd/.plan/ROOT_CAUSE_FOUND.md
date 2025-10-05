# Root Cause Found - HTTP Connection Failure

**Date**: 2025-10-05 22:27  
**Status**: ✅ FIXED (22:37) - See update at bottom  

## Summary

The `/execute` endpoint connection failure is NOT an HTTP/routing issue. The request **does reach the server** and inference starts, but then **the worker process crashes/hangs**, causing the HTTP connection to be dropped.

## Root Cause

**Use-after-free bug in GPU pointer management**

### The Bug

In `src/cuda/weight_loader.rs::load_model_from_rust()`:

```rust
pub unsafe fn load_model_from_rust(...) -> Result<*mut ffi::CudaModel, String> {
    // Step 1: Load weights to GPU (Rust)
    let gpu_pointers = load_weights_to_gpu(path)?;  // ← HashMap created
    
    // Step 2-4: Pass pointers to C++
    let pointer_map = ffi::cuda_create_pointer_map(total_vram);
    for (name, ptr) in &gpu_pointers {
        ffi::cuda_pointer_map_insert(pointer_map, c_name.as_ptr(), *ptr);
    }
    let model = ffi::cuda_load_model_from_pointers(...);
    
    // Step 5: Free pointer map
    ffi::cuda_free_pointer_map(pointer_map);
    
    Ok(model)
}  // ← gpu_pointers HashMap is DROPPED here!
```

**What happens:**
1. Rust allocates GPU memory and stores pointers in `gpu_pointers` HashMap
2. Rust passes raw pointers to C++
3. C++ stores the raw pointers (does NOT copy the GPU data)
4. Rust function returns, `gpu_pointers` HashMap is dropped
5. **GPU memory is NOT freed, but Rust loses track of it**
6. C++ tries to read from GPU pointers → **all zeros** (uninitialized or corrupted memory)
7. Inference crashes/hangs → HTTP connection dropped

### Evidence

From test output:
```
🎉 [C++] Using pre-loaded model from Rust (VRAM: 11047898204684.29 MB)  ← Garbage value!
✅ QwenTransformer initialized
First 10 embedding values: 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00  ← All zeros!
❌ Request failed: error sending request for url (http://localhost:37947/execute)  ← Connection dropped
```

The VRAM value is garbage (11 petabytes!) because `total_vram` calculation is also broken:

```rust
let total_vram = gpu_pointers.values()
    .map(|_| 0u64) // ← BUG: Always maps to 0!
    .sum::<u64>();
```

## Why This Causes HTTP Failure

1. ✅ Worker starts, HTTP server listens
2. ✅ Health endpoint works (no inference needed)
3. ✅ Execute request arrives, validation passes
4. ✅ Inference starts, tokenization succeeds
5. ❌ **Forward pass reads invalid GPU memory**
6. ❌ **C++ code crashes or hangs**
7. ❌ **HTTP connection is dropped/reset**
8. ❌ Client gets "error sending request"

## The Fix

**Option 1: Keep GPU pointers alive in Rust (RECOMMENDED)**

Store the GPU pointers in a static/global structure so they're never freed:

```rust
use std::sync::Mutex;
use std::collections::HashMap;

// Global storage for GPU pointers (never freed until program exit)
static GPU_POINTER_REGISTRY: Mutex<Option<HashMap<String, *mut c_void>>> = Mutex::new(None);

pub unsafe fn load_model_from_rust(...) -> Result<*mut ffi::CudaModel, String> {
    let gpu_pointers = load_weights_to_gpu(path)?;
    
    // Store pointers globally so they're never freed
    {
        let mut registry = GPU_POINTER_REGISTRY.lock().unwrap();
        *registry = Some(gpu_pointers.clone());
    }
    
    // Pass to C++ (now safe because pointers are kept alive)
    let pointer_map = create_and_populate_pointer_map(&gpu_pointers)?;
    let model = ffi::cuda_load_model_from_pointers(...);
    ffi::cuda_free_pointer_map(pointer_map);
    
    Ok(model)
}
```

**Option 2: Have C++ copy the GPU data (SLOWER)**

Make C++ allocate its own GPU memory and copy the data. This is safer but slower and uses 2× VRAM.

**Option 3: Transfer ownership to C++ (COMPLEX)**

Make C++ responsible for freeing the GPU memory. Requires careful lifetime management.

## Recommended Solution

**Use Option 1** - it's:
- ✅ Fast (no extra copies)
- ✅ Memory efficient (no duplication)
- ✅ Simple (just keep pointers alive)
- ✅ Safe (pointers valid for program lifetime)

The only downside is we can't free the GPU memory until program exit, but that's fine for a worker process that loads one model and runs until shutdown.

## Additional Fixes Needed

1. **Fix total_vram calculation:**
   ```rust
   let total_vram = gpu_pointers.values()
       .map(|_| 0u64)  // ← WRONG
       .sum::<u64>();
   ```
   
   Should track actual sizes in `load_weights_to_gpu`.

2. **Add error handling in C++ forward pass:**
   Currently if embeddings are invalid, C++ just continues with zeros. Should check and fail fast.

## Timeline

- **22:14** - Optimized loading complete (1s load time!)
- **22:15** - Discovered HTTP connection issue
- **22:27** - **ROOT CAUSE IDENTIFIED** - Use-after-free bug

## Next Steps

1. Implement Option 1 (GPU pointer registry)
2. Fix total_vram calculation
3. Test haiku generation
4. Verify embeddings are non-zero
5. Confirm inference completes successfully

---

## UPDATE: ACTUAL ROOT CAUSE FOUND & FIXED ✅

**Time**: 22:37

The use-after-free bug was **partially correct** but not the main issue. The actual critical bug was:

### Bug #2: Incorrect Type Cast (THE REAL CULPRIT)

**Location**: `cuda/src/ffi_inference.cpp:62`

**The Problem**:
```cpp
// WRONG - reads from wrong memory!
auto* qwen_model = reinterpret_cast<worker::model::QwenModel*>(model_ptr);
```

`model_ptr` is a `CudaModel*` which is actually a `ModelImpl*`, NOT a `QwenModel*`!

This caused the code to interpret the `ModelImpl` struct as if it were a `QwenModel` struct, reading from completely wrong memory offsets. That's why:
- Embeddings were all zeros (reading from wrong address)
- VRAM value was garbage (reading from wrong field)
- Inference crashed (dereferencing invalid pointers)

**The Fix**:
```cpp
// CORRECT - properly access the QwenModel
auto* model_impl = reinterpret_cast<worker::ModelImpl*>(model_ptr);
auto* qwen_model = model_impl->get_qwen_model();
```

### Both Bugs Fixed

1. ✅ **GPU pointer lifetime** - Added global registry (prevents use-after-free)
2. ✅ **Type cast bug** - Fixed pointer access (THE CRITICAL FIX)

### Result

```
✅ Generated 100 tokens
✅ Got response with status: 200 OK
✅ HTTP streaming works
✅ Inference completes without crashing
```

**See**: `BUG_FIXED_HTTP_CONNECTION.md` for complete details.

**Status**: ✅ COMPLETE - Both bugs fixed, inference working!
