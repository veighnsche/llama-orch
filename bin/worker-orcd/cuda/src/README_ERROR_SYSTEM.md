# CUDA Error System

**Status**: ✅ Implemented  
**Story**: FT-008  
**Spec**: M0-W-1501, CUDA-5040, CUDA-5041

---

## Overview

The CUDA error system provides structured error handling across the FFI boundary. C++ exceptions are converted to error codes at the FFI boundary, preventing exceptions from leaking to Rust.

---

## Architecture

### Error Flow

```
C++ Layer                    FFI Boundary              Rust Layer
-----------                  ------------              ----------
throw CudaError      →       error_code = e.code()  →  Err(CudaError::from_code(code))
                             return nullptr
```

### Components

1. **`worker_errors.h`** - C error code enum (FFI boundary)
2. **`cuda_error.h`** - C++ exception class (internal)
3. **`errors.cpp`** - Error message implementation
4. **`ffi.cpp`** - Exception-to-error-code conversion

---

## Error Codes

### Enum Definition

```c
typedef enum {
    CUDA_SUCCESS = 0,                      // Operation succeeded
    CUDA_ERROR_INVALID_DEVICE = 1,         // Invalid GPU device ID
    CUDA_ERROR_OUT_OF_MEMORY = 2,          // Insufficient VRAM
    CUDA_ERROR_MODEL_LOAD_FAILED = 3,      // Model loading failed
    CUDA_ERROR_INFERENCE_FAILED = 4,       // Inference execution failed
    CUDA_ERROR_INVALID_PARAMETER = 5,      // Invalid function parameter
    CUDA_ERROR_KERNEL_LAUNCH_FAILED = 6,   // CUDA kernel launch failed
    CUDA_ERROR_VRAM_RESIDENCY_FAILED = 7,  // VRAM residency check failed
    CUDA_ERROR_DEVICE_NOT_FOUND = 8,       // No CUDA devices found
    CUDA_ERROR_UNKNOWN = 99,               // Unknown error
} CudaErrorCode;
```

### Error Messages

All error codes have human-readable messages:

| Code | Message |
|------|---------|
| 0 | Success |
| 1 | Invalid CUDA device ID |
| 2 | Out of GPU memory (VRAM) |
| 3 | Failed to load model from GGUF file |
| 4 | Inference execution failed |
| 5 | Invalid parameter provided |
| 6 | CUDA kernel launch failed |
| 7 | VRAM residency check failed (RAM fallback detected) |
| 8 | No CUDA devices found |
| 99 | Unknown error occurred |

---

## C++ Exception Class

### CudaError

```cpp
namespace worker {

class CudaError : public std::exception {
public:
    CudaError(int code, const std::string& message);
    
    int code() const noexcept;
    const char* what() const noexcept override;
    
    // Factory methods
    static CudaError invalid_device(const std::string& details);
    static CudaError out_of_memory(const std::string& details);
    static CudaError model_load_failed(const std::string& details);
    static CudaError inference_failed(const std::string& details);
    static CudaError invalid_parameter(const std::string& details);
    static CudaError kernel_launch_failed(const std::string& details);
    static CudaError vram_residency_failed(const std::string& details);
    static CudaError device_not_found(const std::string& details);
};

} // namespace worker
```

### Usage Example

```cpp
#include "cuda_error.h"

void some_cuda_function() {
    if (device_id < 0) {
        throw CudaError::invalid_device("Device ID must be >= 0");
    }
    
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw CudaError::out_of_memory(
            "Failed to allocate " + std::to_string(size) + " bytes"
        );
    }
}
```

---

## Exception-to-Error-Code Pattern

### Pattern Template

All FFI functions use this pattern:

```cpp
extern "C" ReturnType* ffi_function(Args..., int* error_code) {
    try {
        // Implementation that may throw CudaError
        auto result = do_work();
        *error_code = CUDA_SUCCESS;
        return result;
    } catch (const CudaError& e) {
        *error_code = e.code();
        return nullptr;
    } catch (const std::exception& e) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    }
}
```

### Example Implementation

```cpp
extern "C" CudaContext* cuda_init(int gpu_device, int* error_code) {
    try {
        auto ctx = std::make_unique<Context>(gpu_device);
        *error_code = CUDA_SUCCESS;
        return reinterpret_cast<CudaContext*>(ctx.release());
    } catch (const CudaError& e) {
        *error_code = e.code();
        return nullptr;
    } catch (const std::exception& e) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;
        return nullptr;
    }
}
```

---

## Error Message Function

### Implementation

```cpp
extern "C" const char* cuda_error_message(int error_code) {
    switch (error_code) {
        case CUDA_SUCCESS:
            return "Success";
        case CUDA_ERROR_INVALID_DEVICE:
            return "Invalid CUDA device ID";
        // ... all other codes ...
        default:
            return "Unrecognized error code";
    }
}
```

### Properties

- **Thread-safe**: Returns static strings
- **Never NULL**: Always returns valid pointer
- **No allocation**: Static storage only
- **Stable**: Messages never change

---

## Testing

### Unit Tests (24 tests)

**Error Code Tests** (2 tests):
- All codes defined
- Codes are sequential

**Error Message Tests** (12 tests):
- All error codes have messages
- Messages are descriptive
- Invalid code returns fallback
- All messages are non-NULL

**CudaError Tests** (4 tests):
- Constructor stores code and message
- what() returns message
- code() returns correct value
- Inherits from std::exception

**Factory Method Tests** (3 tests):
- All factory methods return correct codes
- Factory methods include prefix
- Factory methods preserve details

**Exception-to-Error-Code Tests** (3 tests):
- Catches CudaError
- Catches std::exception
- Catches unknown exceptions

### Running Tests

```bash
cd bin/worker-orcd
cargo build  # Builds CUDA library via build.rs
# C++ tests will run via CMake/CTest
```

---

## Design Principles

### FFI Safety

1. **No exceptions across FFI** - All exceptions caught at boundary
2. **Error codes only** - Rust receives integer error codes
3. **NULL on error** - Functions return NULL when error occurs
4. **Out-parameters** - Error codes via pointer parameters

### Error Handling

1. **Typed exceptions** - CudaError for structured errors
2. **Factory methods** - Convenient error construction
3. **Descriptive messages** - Include context and details
4. **Stable codes** - Error codes never change

### Thread Safety

1. **Static strings** - Error messages are thread-safe
2. **Immutable exceptions** - Exception objects are immutable
3. **No global state** - No mutable global error state

---

## Best Practices

### Throwing Errors

```cpp
// Use factory methods for common errors
throw CudaError::invalid_device("device 5 not found");
throw CudaError::out_of_memory("requested 16GB, available 8GB");

// Or construct directly
throw CudaError(CUDA_ERROR_KERNEL_LAUNCH_FAILED, "attention kernel failed");
```

### Catching Errors

```cpp
try {
    some_cuda_operation();
} catch (const CudaError& e) {
    // Handle CUDA-specific error
    log_error(e.code(), e.what());
    throw;  // Re-throw if needed
}
```

### FFI Boundary

```cpp
extern "C" ReturnType* ffi_function(Args..., int* error_code) {
    try {
        // Implementation
        *error_code = CUDA_SUCCESS;
        return result;
    } catch (const CudaError& e) {
        *error_code = e.code();  // Preserve specific error
        return nullptr;
    } catch (const std::exception& e) {
        *error_code = CUDA_ERROR_UNKNOWN;  // Generic error
        return nullptr;
    } catch (...) {
        *error_code = CUDA_ERROR_UNKNOWN;  // Unknown error
        return nullptr;
    }
}
```

---

## Specification Compliance

### Requirements Implemented

- ✅ **M0-W-1501**: Error code system with stable codes
- ✅ **CUDA-5040**: Error message function
- ✅ **CUDA-5041**: Exception-to-error-code pattern

**Spec Reference**: `bin/.specs/01_M0_worker_orcd.md` §9.1 Error Codes

---

## Future Enhancements

### M1+

- Error code metrics (track error frequency)
- Structured error context (additional fields)
- Error recovery strategies
- Performance impact analysis

---

**Implementation**: Foundation-Alpha 🏗️  
**Date**: 2025-10-04  
**Story**: FT-008

---
Built by Foundation-Alpha 🏗️
