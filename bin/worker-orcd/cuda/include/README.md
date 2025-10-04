# Worker CUDA FFI Interface

**Status**: 🔒 **LOCKED** (2025-10-04)  
**Version**: 1.0  
**Spec**: M0-W-1052, CUDA-4030, CUDA-4011

---

## Overview

This directory contains the **LOCKED** FFI interface between the Rust layer and C++/CUDA layer in `worker-orcd`. The interface is stable and must not change without team coordination.

---

## Header Files

### `worker_ffi.h` - Main FFI Interface

The primary header that defines all FFI functions. This is the only header Rust needs to include (it includes the others).

**Functions**: 14 total
- Context Management: 3 functions
- Model Loading: 3 functions
- Inference Execution: 3 functions
- Health & Monitoring: 4 functions
- Error Handling: 1 function

### `worker_types.h` - Opaque Handle Types

Defines the three opaque handle types used across the FFI boundary:
- `CudaContext` - CUDA device context
- `CudaModel` - Loaded model in VRAM
- `InferenceResult` - Active inference session

### `worker_errors.h` - Error Codes

Defines the error code enumeration and error message function:
- `CudaErrorCode` enum (0-8, 99)
- `cuda_error_message()` function

---

## Design Principles

### FFI Boundary Rules

1. **C Linkage**: All functions use `extern "C"` for stable ABI
2. **No Exceptions**: All errors returned via out-parameters (error codes)
3. **Opaque Handles**: Rust never accesses C++ internals directly
4. **UTF-8 Strings**: All string parameters are null-terminated UTF-8
5. **NULL Safety**: All functions handle NULL pointers gracefully
6. **Error Codes**: Positive integers (0 = success)
7. **Static Strings**: Error messages use static storage (no allocation)
8. **Single-Threaded**: Each context is single-threaded (no concurrent calls)
9. **Explicit Cleanup**: Rust must call free functions (no automatic cleanup)

### Memory Ownership

- **Rust owns**: Nothing allocated by C++
- **C++ owns**: All CUDA resources (contexts, models, inference state)
- **Rust must**: Call destroy/unload/free functions to release resources
- **C++ must**: Never free Rust-allocated memory

### Thread Safety

- **Context**: NOT thread-safe. Single-threaded access only.
- **Model**: NOT thread-safe. Single-threaded access only.
- **InferenceResult**: NOT thread-safe. Single-threaded access only.
- **Read-only queries**: Safe to call concurrently (e.g., `cuda_get_vram_usage`)

---

## Usage Example (Rust)

```rust
use std::ffi::{CStr, CString};

// Initialize CUDA context
let mut error_code = 0;
let ctx = unsafe { cuda_init(0, &mut error_code) };
if ctx.is_null() {
    let error_msg = unsafe { 
        CStr::from_ptr(cuda_error_message(error_code)).to_string_lossy() 
    };
    panic!("CUDA init failed: {}", error_msg);
}

// Load model
let model_path = CString::new("/path/to/model.gguf").unwrap();
let mut vram_bytes = 0;
let model = unsafe {
    cuda_load_model(ctx, model_path.as_ptr(), &mut vram_bytes, &mut error_code)
};
if model.is_null() {
    let error_msg = unsafe { 
        CStr::from_ptr(cuda_error_message(error_code)).to_string_lossy() 
    };
    panic!("Model load failed: {}", error_msg);
}

// Start inference
let prompt = CString::new("Write a haiku").unwrap();
let result = unsafe {
    cuda_inference_start(model, prompt.as_ptr(), 100, 0.7, 42, &mut error_code)
};
if result.is_null() {
    panic!("Inference start failed");
}

// Generate tokens
let mut token_buffer = vec![0u8; 256];
let mut token_index = 0;
while unsafe {
    cuda_inference_next_token(
        result,
        token_buffer.as_mut_ptr() as *mut i8,
        token_buffer.len() as i32,
        &mut token_index,
        &mut error_code
    )
} {
    let token = CStr::from_bytes_until_nul(&token_buffer)
        .unwrap()
        .to_string_lossy();
    print!("{}", token);
}

// Cleanup
unsafe {
    cuda_inference_free(result);
    cuda_unload_model(model);
    cuda_destroy(ctx);
}
```

---

## Usage Example (C++)

```cpp
#include "worker_ffi.h"
#include <iostream>

int main() {
    int error_code = 0;
    
    // Initialize CUDA context
    CudaContext* ctx = cuda_init(0, &error_code);
    if (!ctx) {
        std::cerr << "CUDA init failed: " 
                  << cuda_error_message(error_code) << std::endl;
        return 1;
    }
    
    // Load model
    uint64_t vram_bytes = 0;
    CudaModel* model = cuda_load_model(
        ctx, "/path/to/model.gguf", &vram_bytes, &error_code
    );
    if (!model) {
        std::cerr << "Model load failed: " 
                  << cuda_error_message(error_code) << std::endl;
        cuda_destroy(ctx);
        return 1;
    }
    
    std::cout << "Model loaded: " << vram_bytes << " bytes" << std::endl;
    
    // Start inference
    InferenceResult* result = cuda_inference_start(
        model, "Write a haiku", 100, 0.7, 42, &error_code
    );
    if (!result) {
        std::cerr << "Inference start failed" << std::endl;
        cuda_unload_model(model);
        cuda_destroy(ctx);
        return 1;
    }
    
    // Generate tokens
    char token_buffer[256];
    int token_index = 0;
    while (cuda_inference_next_token(
        result, token_buffer, sizeof(token_buffer), &token_index, &error_code
    )) {
        std::cout << token_buffer;
    }
    std::cout << std::endl;
    
    // Cleanup
    cuda_inference_free(result);
    cuda_unload_model(model);
    cuda_destroy(ctx);
    
    return 0;
}
```

---

## Error Handling

All functions that can fail use out-parameters for error codes:

```c
int error_code = 0;
CudaContext* ctx = cuda_init(0, &error_code);
if (!ctx) {
    const char* error_msg = cuda_error_message(error_code);
    fprintf(stderr, "Error: %s\n", error_msg);
}
```

### Error Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | `CUDA_SUCCESS` | Operation succeeded |
| 1 | `CUDA_ERROR_INVALID_DEVICE` | Invalid GPU device ID |
| 2 | `CUDA_ERROR_OUT_OF_MEMORY` | Insufficient VRAM |
| 3 | `CUDA_ERROR_MODEL_LOAD_FAILED` | Model loading failed |
| 4 | `CUDA_ERROR_INFERENCE_FAILED` | Inference execution failed |
| 5 | `CUDA_ERROR_INVALID_PARAMETER` | Invalid function parameter |
| 6 | `CUDA_ERROR_KERNEL_LAUNCH_FAILED` | CUDA kernel launch failed |
| 7 | `CUDA_ERROR_VRAM_RESIDENCY_FAILED` | VRAM residency check failed |
| 8 | `CUDA_ERROR_DEVICE_NOT_FOUND` | No CUDA devices found |
| 99 | `CUDA_ERROR_UNKNOWN` | Unknown error |

---

## Verification

### Compilation Tests

Run the verification script to test header compilation:

```bash
cd bin/worker-orcd/cuda/tests
./verify_ffi_headers.sh
```

This verifies:
- Headers compile with C compiler (gcc)
- Headers compile with C++ compiler (g++)
- Include guards work correctly
- All functions are declared
- All error codes are defined

### Unit Tests

Run the GTest suite to verify interface correctness:

```bash
cd bin/worker-orcd
cargo build  # Builds CUDA library via build.rs
cargo test   # Runs Rust tests
```

---

## Change Control

### Lock Status: 🔒 LOCKED

This interface is **LOCKED** as of 2025-10-04. Any changes require:

1. **Written justification** (why the change is necessary)
2. **Impact analysis** (which teams are affected)
3. **Approval from PM** (Foundation-Alpha self-review)
4. **Notification to all teams** (Llama, GPT, Foundation)
5. **Version bump** (update version history)

See `bin/worker-orcd/.plan/coordination/FFI_INTERFACE_LOCKED.md` for full change control process.

---

## Specification References

- **M0-W-1052**: C API Interface
- **M0-W-1050**: Rust Layer Responsibilities
- **M0-W-1051**: C++/CUDA Layer Responsibilities
- **CUDA-4030**: FFI Boundary Specification
- **CUDA-4011**: FFI Boundary Enforcement

Full spec: `bin/.specs/01_M0_worker_orcd.md` §4.2 FFI Boundaries

---

## Team Coordination

### Downstream Dependencies

The following teams are **UNBLOCKED** by this interface:

- **Foundation Team**: FT-007 (Rust FFI bindings)
- **Llama Team**: LT-000 (Llama prep work)
- **GPT Team**: GT-000 (GPT prep work)

### Contact

For questions or change requests, contact:
- **Foundation-Alpha** (FFI interface owner)
- **PM Team** (coordination and approval)

---

**Lock Date**: 2025-10-04  
**Version**: 1.0  
**Status**: 🔒 LOCKED

---
Built by Foundation-Alpha 🏗️
