# C++ Code Cleanup Plan: Remove Rust-Duplicated Code

**Date**: 2025-10-05  
**Status**: READY TO EXECUTE  
**Backup**: User has confirmed backup exists

---

## Understanding: What Exists in Rust

### ✅ COMPLETE in Rust (worker-crates)

1. **`worker-gguf`** (444 lines)
   - GGUF metadata parsing (STUB, but API defined)
   - Architecture detection
   - Config extraction
   - **Status**: API complete, needs real implementation

2. **`worker-tokenizer`** (full implementation)
   - BPE encoder/decoder
   - GGUF vocab parsing
   - HuggingFace JSON support
   - Streaming decoder
   - **Status**: ✅ COMPLETE

3. **`worker-models`** (800+ lines)
   - `QwenModel`, `Phi3Model`, `GPTModel` structs
   - `AdapterFactory` for architecture detection
   - Forward pass trait definitions
   - **Status**: Structs defined, needs FFI wiring

4. **`worker-http`** (full implementation)
   - Axum HTTP server
   - SSE streaming
   - Route handlers
   - **Status**: ✅ COMPLETE

5. **`worker-common`** (full implementation)
   - Error types
   - Sampling config
   - Inference results
   - **Status**: ✅ COMPLETE

6. **`worker-compute`** (trait definition)
   - `ComputeBackend` trait
   - Platform abstraction
   - **Status**: ✅ TRAIT DEFINED

---

## C++ Code to DELETE

### Files to DELETE Entirely

1. **`cuda/src/gguf/header_parser.cpp`** (447 lines)
   - ❌ DELETE: Should be in `worker-gguf`
   - Reason: No GPU needed, pure file I/O

2. **`cuda/src/gguf/header_parser.h`** (173 lines)
   - ❌ DELETE: Should be in `worker-gguf`

3. **`cuda/src/gguf/llama_metadata.cpp`** (308 lines)
   - ❌ DELETE: Should be in `worker-gguf`
   - Reason: No GPU needed, just key-value parsing

4. **`cuda/src/gguf/llama_metadata.h`** (184 lines)
   - ❌ DELETE: Should be in `worker-gguf`

5. **`cuda/src/io/mmap_file.cpp`** (~100 lines)
   - ❌ DELETE: Should be in `worker-gguf`
   - Reason: File I/O, not GPU-specific

6. **`cuda/src/io/mmap_file.h`** (121 lines)
   - ❌ DELETE: Should be in `worker-gguf`

**Total to delete**: ~1,333 lines of C++ code

### Functions to DELETE from Existing Files

1. **`cuda/src/model/gpt_weights.cpp`**
   - ❌ DELETE: `parse_config_from_gguf()` function
   - Reason: Config parsing should be in Rust
   - Keep: `load_from_gguf()` (weight loading to VRAM)

---

## C++ Code to KEEP

### Files to KEEP (GPU-specific)

1. ✅ **`cuda/src/ffi.cpp`** - FFI boundary
2. ✅ **`cuda/src/context.cpp`** - CUDA context
3. ✅ **`cuda/src/model/gpt_weights.cpp`** - Weight loading to VRAM (partial)
4. ✅ **`cuda/src/model/gpt_model.cpp`** - GPU model structure
5. ✅ **`cuda/src/inference_impl.cpp`** - Inference execution
6. ✅ **`cuda/src/kv_cache.cpp`** - KV cache management
7. ✅ **`cuda/kernels/*.cu`** - All CUDA kernels
8. ✅ **`cuda/src/cublas_wrapper.cpp`** - cuBLAS operations
9. ✅ **`cuda/src/device_memory.cpp`** - GPU memory management

**Total to keep**: ~3,305 lines of C++ (GPU-specific)

---

## The New Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ RUST LAYER (bin/worker-crates/)                             │
│                                                              │
│  worker-gguf:                                                │
│    - Parse GGUF metadata                                     │
│    - Extract architecture, config                           │
│    - Return to Rust main                                     │
│                                                              │
│  worker-tokenizer:                                           │
│    - Tokenize prompt                                         │
│    - Decode tokens                                           │
│                                                              │
│  worker-http:                                                │
│    - HTTP server                                             │
│    - SSE streaming                                           │
│                                                              │
│                         │ FFI                                │
│                         ↓                                     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ C++ CUDA LAYER (bin/worker-orcd/cuda/)                      │
│                                                              │
│  FFI Interface:                                              │
│    - cuda_init(device_id)                                   │
│    - cuda_load_model(ctx, path) ← Rust passes path         │
│    - cuda_inference_start(...)                              │
│    - cuda_inference_next_token(...)                         │
│                                                              │
│  Weight Loading:                                             │
│    - Open GGUF file (mmap)                                  │
│    - Read tensor data                                        │
│    - Allocate GPU memory                                     │
│    - Copy to VRAM                                            │
│                                                              │
│  Inference:                                                  │
│    - Execute CUDA kernels                                    │
│    - Manage KV cache                                         │
│    - Return token IDs                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Cleanup Steps

### Step 1: Delete GGUF Parser (C++)

```bash
rm cuda/src/gguf/header_parser.cpp
rm cuda/src/gguf/header_parser.h
rm cuda/src/gguf/llama_metadata.cpp
rm cuda/src/gguf/llama_metadata.h
rmdir cuda/src/gguf  # If empty
```

### Step 2: Delete mmap (C++)

```bash
rm cuda/src/io/mmap_file.cpp
rm cuda/src/io/mmap_file.h
```

### Step 3: Update gpt_weights.cpp

Remove `parse_config_from_gguf()` function, keep only weight loading.

### Step 4: Update CMakeLists.txt

Remove deleted files from build.

### Step 5: Update FFI

Simplify FFI - Rust passes parsed config, not path.

---

## Updated FFI Interface

### Before (C++ does everything)

```cpp
extern "C" {
    CudaModel* cuda_load_model(
        CudaContext* ctx,
        const char* model_path,  // C++ parses GGUF
        int* error
    );
}
```

### After (Rust does parsing, C++ does GPU)

```rust
// Rust side
let metadata = worker_gguf::GGUFMetadata::from_file(&path)?;
let config = extract_config(&metadata)?;

// Pass config to C++
let cuda_model = unsafe {
    cuda_load_model_with_config(
        ctx,
        path.as_ptr(),
        &config as *const GPTConfig,
        error
    )
};
```

```cpp
// C++ side
extern "C" {
    CudaModel* cuda_load_model_with_config(
        CudaContext* ctx,
        const char* model_path,
        const GPTConfig* config,  // Rust passes config
        int* error
    ) {
        // Just load weights to VRAM
        auto weights = load_weights_to_vram(model_path, config);
        return new CudaModel(ctx, weights);
    }
}
```

---

## Benefits of Cleanup

1. ✅ **No duplication** - GGUF parsing in ONE place (Rust)
2. ✅ **Clear separation** - Rust = I/O, C++ = GPU
3. ✅ **Easier maintenance** - Changes in ONE place
4. ✅ **Better testing** - Test Rust and C++ separately
5. ✅ **Smaller C++ codebase** - 1,333 fewer lines
6. ✅ **Reusable Rust** - Can use for worker-aarmd (Metal)

---

## Execution Order

1. ✅ Delete C++ GGUF parser files
2. ✅ Delete C++ mmap files
3. ✅ Update gpt_weights.cpp (remove parse_config)
4. ✅ Update CMakeLists.txt
5. ✅ Update FFI interface
6. ⬜ Implement real GGUF parser in Rust (GT-051-REFACTOR)
7. ⬜ Wire everything together

---

## Risk Mitigation

- ✅ User has backup
- ✅ Can revert from backup if needed
- ✅ Delete in stages, test after each
- ✅ Keep GPU-specific code intact

---

**Ready to execute cleanup?**

---
Created by Project Management Team 📋
