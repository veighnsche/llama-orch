# FT-008: Error Code System (C++)

**Team**: Foundation-Alpha  
**Sprint**: Sprint 2 - FFI Layer  
**Size**: S (1 day)  
**Days**: 14 - 14  
**Spec Ref**: M0-W-1501, CUDA-5040, CUDA-5041

---

## Story Description

Implement C++ error code system with exception-to-error-code conversion pattern. This provides structured error handling across the FFI boundary without C++ exceptions leaking to Rust.

---

## Acceptance Criteria

- [ ] CudaErrorCode enum defined with all error codes from spec
- [ ] CudaError exception class with code and message
- [ ] cuda_error_message() function returns human-readable messages
- [ ] Exception-to-error-code wrapper pattern for all FFI functions
- [ ] Unit tests validate error code conversion
- [ ] Error messages are descriptive and actionable
- [ ] No exceptions thrown across FFI boundary
- [ ] Error codes are stable (documented in header)

---

## Dependencies

### Upstream (Blocks This Story)
- FT-006: FFI interface definition (Expected completion: Day 11)

### Downstream (This Story Blocks)
- FT-009: Rust error conversion needs C++ error codes
- FT-010: CUDA context init needs error handling

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/include/worker_errors.h` - Error code definitions
- `bin/worker-orcd/cuda/src/errors.cpp` - Error message implementation
- `bin/worker-orcd/cuda/src/cuda_error.h` - CudaError exception class
- `bin/worker-orcd/cuda/src/cuda_error.cpp` - Exception implementation

### Key Interfaces
```cpp
// worker_errors.h
#ifndef WORKER_ERRORS_H
#define WORKER_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_DEVICE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_MODEL_LOAD_FAILED = 3,
    CUDA_ERROR_INFERENCE_FAILED = 4,
    CUDA_ERROR_INVALID_PARAMETER = 5,
    CUDA_ERROR_KERNEL_LAUNCH_FAILED = 6,
    CUDA_ERROR_VRAM_RESIDENCY_FAILED = 7,
    CUDA_ERROR_UNKNOWN = 99,
} CudaErrorCode;

const char* cuda_error_message(int error_code);

#ifdef __cplusplus
}
#endif

#endif // WORKER_ERRORS_H

// cuda_error.h (C++ only)
#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

#include <exception>
#include <string>
#include "worker_errors.h"

namespace worker {

class CudaError : public std::exception {
public:
    CudaError(int code, const std::string& message)
        : code_(code), message_(message) {}
    
    CudaError(int code, const char* message)
        : code_(code), message_(message) {}
    
    int code() const noexcept { return code_; }
    const char* what() const noexcept override { return message_.c_str(); }
    
    // Factory methods for common errors
    static CudaError invalid_device(const std::string& details) {
        return CudaError(CUDA_ERROR_INVALID_DEVICE, "Invalid device: " + details);
    }
    
    static CudaError out_of_memory(const std::string& details) {
        return CudaError(CUDA_ERROR_OUT_OF_MEMORY, "Out of memory: " + details);
    }
    
    static CudaError model_load_failed(const std::string& details) {
        return CudaError(CUDA_ERROR_MODEL_LOAD_FAILED, "Model load failed: " + details);
    }
    
    static CudaError inference_failed(const std::string& details) {
        return CudaError(CUDA_ERROR_INFERENCE_FAILED, "Inference failed: " + details);
    }
    
    static CudaError invalid_parameter(const std::string& details) {
        return CudaError(CUDA_ERROR_INVALID_PARAMETER, "Invalid parameter: " + details);
    }
    
    static CudaError kernel_launch_failed(const std::string& details) {
        return CudaError(CUDA_ERROR_KERNEL_LAUNCH_FAILED, "Kernel launch failed: " + details);
    }
    
private:
    int code_;
    std::string message_;
};

} // namespace worker

#endif // CUDA_ERROR_H

// errors.cpp
#include "worker_errors.h"

extern "C" const char* cuda_error_message(int error_code) {
    switch (error_code) {
        case CUDA_SUCCESS:
            return "Success";
        case CUDA_ERROR_INVALID_DEVICE:
            return "Invalid CUDA device ID";
        case CUDA_ERROR_OUT_OF_MEMORY:
            return "Out of GPU memory (VRAM)";
        case CUDA_ERROR_MODEL_LOAD_FAILED:
            return "Failed to load model from GGUF file";
        case CUDA_ERROR_INFERENCE_FAILED:
            return "Inference execution failed";
        case CUDA_ERROR_INVALID_PARAMETER:
            return "Invalid parameter provided";
        case CUDA_ERROR_KERNEL_LAUNCH_FAILED:
            return "CUDA kernel launch failed";
        case CUDA_ERROR_VRAM_RESIDENCY_FAILED:
            return "VRAM residency check failed";
        case CUDA_ERROR_UNKNOWN:
            return "Unknown error occurred";
        default:
            return "Unrecognized error code";
    }
}

// Exception-to-error-code wrapper pattern
extern "C" CudaContext* cuda_init(int gpu_device, int* error_code) {
    try {
        auto ctx = std::make_unique<Context>(gpu_device);
        *error_code = CUDA_SUCCESS;
        return reinterpret_cast<CudaContext*>(ctx.release());
    } catch (const worker::CudaError& e) {
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

### Implementation Notes
- Error codes are stable (never change values)
- Error messages are human-readable and actionable
- CudaError exception class wraps code + message
- Factory methods provide convenient error construction
- Exception-to-error-code pattern catches all exceptions
- Unknown exceptions mapped to CUDA_ERROR_UNKNOWN
- Error messages are static strings (no allocation)
- Thread-safe (error messages are const char*)

---

## Testing Strategy

### Unit Tests
- Test cuda_error_message() for all error codes
- Test cuda_error_message() with invalid code returns fallback
- Test CudaError constructor stores code and message
- Test CudaError::what() returns message
- Test CudaError factory methods create correct errors
- Test exception-to-error-code wrapper catches CudaError
- Test exception-to-error-code wrapper catches std::exception
- Test exception-to-error-code wrapper catches unknown exceptions

### Integration Tests
- Test FFI function returns correct error code on failure
- Test error message accessible from Rust via cuda_error_message()

### Manual Verification
1. Compile C++ code: `cmake --build build/`
2. Run C++ unit tests: `./build/tests/error_tests`
3. Verify all tests pass
4. Check error messages are descriptive

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (8+ tests)
- [ ] Integration tests passing (2+ tests)
- [ ] Documentation updated (error code docs, exception class docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §9.1 Error Codes (M0-W-1501)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §9.1 CUDA Error Codes (CUDA-5040, CUDA-5041)
- Related Stories: FT-006 (FFI interface), FT-009 (Rust error conversion)

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **Exception caught at FFI boundary**
   ```rust
   // From Rust side after FFI call
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "error_caught",
       target: function_name.to_string(),
       error_kind: Some(error_code.to_string()),
       human: format!("Exception caught in {}: {}", function_name, error_message),
       ..Default::default()
   });
   ```

**Why this matters**: Exception-to-error-code conversion is critical for FFI safety. Narration helps track which C++ exceptions are most common and where they occur.

**Note**: C++ layer doesn't directly emit narration (no tracing in C++). Narration happens on Rust side after error code conversion.

---
*Narration guidance added by Narration-Core Team 🎀*

---

## 🔍 Testing Team Requirements

**From**: Testing Team (Pre-Development Audit)

### Unit Testing Requirements
- **Test cuda_error_message() for all error codes** (CUDA_SUCCESS through CUDA_ERROR_UNKNOWN)
- **Test cuda_error_message() with invalid code returns fallback** ("Unrecognized error code")
- **Test CudaError constructor stores code and message** (state validation)
- **Test CudaError::what() returns message** (exception interface)
- **Test CudaError::code() returns correct code** (getter)
- **Test CudaError factory methods create correct errors** (invalid_device, out_of_memory, etc.)
- **Test exception-to-error-code wrapper catches CudaError** (typed exception)
- **Test exception-to-error-code wrapper catches std::exception** (generic exception)
- **Test exception-to-error-code wrapper catches unknown exceptions** (catch-all)

### Integration Testing Requirements
- **Test FFI function returns correct error code on failure** (error propagation)
- **Test error message accessible from Rust via cuda_error_message()** (cross-language)
- **Test exception thrown in C++ becomes error code in Rust** (FFI boundary)

### BDD Testing Requirements (VERY IMPORTANT)
- **Scenario**: CudaError exception thrown
  - Given a CUDA operation that fails
  - When a CudaError is thrown
  - Then the error code should be set correctly
  - And the error message should be descriptive
- **Scenario**: Exception converted to error code at FFI boundary
  - Given a C++ function that throws CudaError
  - When called from FFI wrapper
  - Then the error code should be returned via out-parameter
  - And the function should return NULL
- **Scenario**: Unknown exception caught
  - Given a C++ function that throws std::runtime_error
  - When called from FFI wrapper
  - Then CUDA_ERROR_UNKNOWN should be returned
  - And the function should return NULL

### Critical Paths to Test
- Error code enum completeness (all codes defined)
- Error message quality (descriptive and actionable)
- Exception-to-error-code conversion (all exception types)
- FFI boundary safety (no exceptions leak to Rust)

### Edge Cases
- Error code 0 (CUDA_SUCCESS)
- Error code out of enum range
- Very long error messages
- Nested exceptions
- Exception during exception handling

---
Test opportunities identified by Testing Team 🔍
