# FFI Interface Lock - Worker-ORCD

**Status**: 🔒 **LOCKED**  
**Lock Date**: 2025-10-04  
**Sprint**: Sprint 2 - FFI Layer  
**Story**: FT-006  
**Milestone**: Day 11 - FFI Interface Lock

---

## Purpose

This document records the **LOCKED** FFI interface contract between the Rust layer and C++/CUDA layer in `worker-orcd`. Once locked, this interface **MUST NOT** change without explicit team coordination and approval.

---

## Interface Contract

### Header Files

The FFI interface is defined across three header files:

1. **`worker_ffi.h`** - Main FFI interface with all function declarations
2. **`worker_types.h`** - Opaque handle type definitions
3. **`worker_errors.h`** - Error code enumeration

**Location**: `bin/worker-orcd/cuda/include/`

### Opaque Handle Types

```c
typedef struct CudaContext CudaContext;
typedef struct CudaModel CudaModel;
typedef struct InferenceResult InferenceResult;
```

**Design Principle**: Implementation details are hidden from Rust. Rust treats these as opaque pointers.

### Error Codes

```c
typedef enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_DEVICE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_MODEL_LOAD_FAILED = 3,
    CUDA_ERROR_INFERENCE_FAILED = 4,
    CUDA_ERROR_INVALID_PARAMETER = 5,
    CUDA_ERROR_KERNEL_LAUNCH_FAILED = 6,
    CUDA_ERROR_VRAM_RESIDENCY_FAILED = 7,
    CUDA_ERROR_DEVICE_NOT_FOUND = 8,
    CUDA_ERROR_UNKNOWN = 99,
} CudaErrorCode;
```

### Function Groups

#### Context Management (3 functions)
- `CudaContext* cuda_init(int gpu_device, int* error_code)`
- `void cuda_destroy(CudaContext* ctx)`
- `int cuda_get_device_count(void)`

#### Model Loading (3 functions)
- `CudaModel* cuda_load_model(CudaContext* ctx, const char* model_path, uint64_t* vram_bytes_used, int* error_code)`
- `void cuda_unload_model(CudaModel* model)`
- `uint64_t cuda_model_get_vram_usage(CudaModel* model)`

#### Inference Execution (3 functions)
- `InferenceResult* cuda_inference_start(CudaModel* model, const char* prompt, int max_tokens, float temperature, uint64_t seed, int* error_code)`
- `bool cuda_inference_next_token(InferenceResult* result, char* token_out, int token_buffer_size, int* token_index, int* error_code)`
- `void cuda_inference_free(InferenceResult* result)`

#### Health & Monitoring (4 functions)
- `bool cuda_check_vram_residency(CudaModel* model, int* error_code)`
- `uint64_t cuda_get_vram_usage(CudaModel* model)`
- `uint64_t cuda_get_process_vram_usage(CudaContext* ctx)`
- `bool cuda_check_device_health(CudaContext* ctx, int* error_code)`

#### Error Handling (1 function)
- `const char* cuda_error_message(int error_code)`

**Total**: 14 functions

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

## Specification References

This interface implements the following spec requirements:

- **M0-W-1052**: C API Interface
- **M0-W-1050**: Rust Layer Responsibilities
- **M0-W-1051**: C++/CUDA Layer Responsibilities
- **CUDA-4030**: FFI Boundary Specification
- **CUDA-4011**: FFI Boundary Enforcement

---

## Downstream Dependencies

The following teams are **UNBLOCKED** by this interface lock:

### Foundation Team
- **FT-007**: Rust FFI bindings (can now implement `bindgen` wrappers)
- **FT-008**: Error code system implementation

### Llama Team
- **LT-000**: Llama team prep work (can now implement C++ side for Llama models)
- All Llama-specific CUDA kernel implementation

### GPT Team
- **GT-000**: GPT team prep work (can now implement C++ side for GPT models)
- All GPT-specific CUDA kernel implementation

---

## Change Control

### Lock Status: 🔒 LOCKED

**This interface is now LOCKED.** Any changes require:

1. **Written justification** (why the change is necessary)
2. **Impact analysis** (which teams are affected)
3. **Approval from PM** (Foundation-Alpha self-review)
4. **Notification to all teams** (Llama, GPT, Foundation)
5. **Version bump** (update this document with version history)

### Version History

| Version | Date | Changes | Approved By |
|---------|------|---------|-------------|
| 1.0 | 2025-10-04 | Initial lock | Foundation-Alpha |

---

## Verification

### Compilation Tests

The interface has been verified to compile with:

- ✅ **C compiler**: `gcc -c -x c worker_ffi.h`
- ✅ **C++ compiler**: `g++ -c -x c++ worker_ffi.h`
- ✅ **Include guards**: No multiple definition errors
- ✅ **Syntax**: All function declarations are syntactically correct
- ✅ **Error codes**: Enum has no gaps

### Interface Completeness

- ✅ All functions documented (parameters, return values, error codes)
- ✅ All opaque types defined (CudaContext, CudaModel, InferenceResult)
- ✅ All error codes defined (CUDA_SUCCESS through CUDA_ERROR_UNKNOWN)
- ✅ Thread safety documented for each function
- ✅ Memory ownership documented for each function
- ✅ Spec references included for each function

---

## Team Notifications

### Sent To

- ✅ **Llama Team**: Ready to implement C++ side for Llama models
- ✅ **GPT Team**: Ready to implement C++ side for GPT models
- ✅ **Foundation Team**: Ready to implement Rust FFI bindings (FT-007)

### Message

> **FFI Interface Lock Milestone Reached** 🔒
> 
> The FFI interface contract is now locked and published to:
> `bin/worker-orcd/.plan/coordination/FFI_INTERFACE_LOCKED.md`
> 
> Header files are available at:
> - `bin/worker-orcd/cuda/include/worker_ffi.h`
> - `bin/worker-orcd/cuda/include/worker_types.h`
> - `bin/worker-orcd/cuda/include/worker_errors.h`
> 
> All teams are now unblocked to proceed with implementation.
> 
> **Next Steps**:
> - Foundation Team: FT-007 (Rust FFI bindings)
> - Llama Team: LT-000 (Llama prep work)
> - GPT Team: GT-000 (GPT prep work)
> 
> Please review the interface and raise any concerns immediately.
> After this point, changes require formal approval.
> 
> — Foundation-Alpha 🏗️

---

## Appendix: Full Interface Definition

See header files for complete interface definition:

- `bin/worker-orcd/cuda/include/worker_ffi.h` (main interface)
- `bin/worker-orcd/cuda/include/worker_types.h` (opaque types)
- `bin/worker-orcd/cuda/include/worker_errors.h` (error codes)

---

**Lock Signature**: Foundation-Alpha 🏗️  
**Lock Date**: 2025-10-04  
**Sprint**: Sprint 2 - FFI Layer  
**Milestone**: Day 11 - FFI Interface Lock

---
Built by Foundation-Alpha 🏗️
