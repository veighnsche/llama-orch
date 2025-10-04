# FT-006: FFI Interface Definition

**Team**: Foundation-Alpha  
**Sprint**: Sprint 2 - FFI Layer  
**Size**: M (2 days)  
**Days**: 10 - 11  
**Spec Ref**: M0-W-1052, CUDA-4030, CUDA-4011

---

## Story Description

Define complete C API interface for Rust-CUDA FFI boundary. This is a **CRITICAL** milestone that locks the FFI contract for all downstream work. Once complete, Llama and GPT teams can proceed with CUDA implementation.

---

## Acceptance Criteria

- [ ] C header file defines all FFI functions with opaque handle types
- [ ] Context management functions: `cuda_init`, `cuda_destroy`, `cuda_get_device_count`
- [ ] Model loading functions: `cuda_load_model`, `cuda_unload_model`, `cuda_model_get_vram_usage`
- [ ] Inference functions: `cuda_inference_start`, `cuda_inference_next_token`, `cuda_inference_free`
- [ ] Health functions: `cuda_check_vram_residency`, `cuda_get_vram_usage`, `cuda_check_device_health`
- [ ] Error handling: `cuda_error_message`, error codes enum
- [ ] All functions use out-parameters for error codes (no exceptions across FFI)
- [ ] Documentation comments for every function (parameters, return values, error codes)
- [ ] Header file published to `coordination/FFI_INTERFACE_LOCKED.md` after review

---

## Dependencies

### Upstream (Blocks This Story)
- FT-005: Request validation (Expected completion: Day 6)

### Downstream (This Story Blocks)
- **CRITICAL**: FT-007 (Rust FFI bindings)
- **CRITICAL**: LT-000 (Llama team prep work - waits for FFI lock)
- **CRITICAL**: GT-000 (GPT team prep work - waits for FFI lock)
- FT-008: Error code system implementation

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/include/worker_ffi.h` - C API header (CRITICAL)
- `bin/worker-orcd/cuda/include/worker_types.h` - Type definitions
- `bin/worker-orcd/cuda/include/worker_errors.h` - Error codes
- `coordination/FFI_INTERFACE_LOCKED.md` - Published interface contract

### Key Interfaces
```c
#ifndef WORKER_FFI_H
#define WORKER_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types (implementation hidden from Rust)
typedef struct CudaContext CudaContext;
typedef struct CudaModel CudaModel;
typedef struct InferenceResult InferenceResult;

// Error codes
typedef enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_DEVICE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_MODEL_LOAD_FAILED = 3,
    CUDA_ERROR_INFERENCE_FAILED = 4,
    CUDA_ERROR_INVALID_PARAMETER = 5,
    CUDA_ERROR_KERNEL_LAUNCH_FAILED = 6,
    CUDA_ERROR_UNKNOWN = 99,
} CudaErrorCode;

// ============================================================================
// Context Management
// ============================================================================

/**
 * Initialize CUDA context for specified GPU device.
 * 
 * @param gpu_device GPU device ID (0, 1, 2, ...)
 * @param error_code Output parameter for error code
 * @return Opaque context handle, or NULL on failure
 * 
 * Error codes:
 *   CUDA_ERROR_INVALID_DEVICE - Invalid device ID
 *   CUDA_ERROR_UNKNOWN - CUDA initialization failed
 */
CudaContext* cuda_init(int gpu_device, int* error_code);

/**
 * Destroy CUDA context and free all resources.
 * 
 * @param ctx Context handle from cuda_init
 */
void cuda_destroy(CudaContext* ctx);

/**
 * Get number of available CUDA devices.
 * 
 * @return Number of CUDA devices, or 0 on error
 */
int cuda_get_device_count(void);

// ============================================================================
// Model Loading
// ============================================================================

/**
 * Load model from GGUF file to VRAM.
 * 
 * @param ctx Context handle
 * @param model_path Absolute path to .gguf file (null-terminated)
 * @param vram_bytes_used Output parameter for VRAM bytes allocated
 * @param error_code Output parameter for error code
 * @return Opaque model handle, or NULL on failure
 * 
 * Error codes:
 *   CUDA_ERROR_MODEL_LOAD_FAILED - File not found or invalid format
 *   CUDA_ERROR_OUT_OF_MEMORY - Insufficient VRAM
 *   CUDA_ERROR_INVALID_PARAMETER - Invalid model_path
 */
CudaModel* cuda_load_model(
    CudaContext* ctx,
    const char* model_path,
    uint64_t* vram_bytes_used,
    int* error_code
);

/**
 * Unload model and free VRAM.
 * 
 * @param model Model handle from cuda_load_model
 */
void cuda_unload_model(CudaModel* model);

/**
 * Get current VRAM usage for model.
 * 
 * @param model Model handle
 * @return VRAM bytes used, or 0 if model is NULL
 */
uint64_t cuda_model_get_vram_usage(CudaModel* model);

// ============================================================================
// Inference Execution
// ============================================================================

/**
 * Start inference job with given prompt and parameters.
 * 
 * @param model Model handle
 * @param prompt Input prompt (null-terminated UTF-8)
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature (0.0-2.0)
 * @param seed Random seed for reproducibility
 * @param error_code Output parameter for error code
 * @return Opaque inference result handle, or NULL on failure
 * 
 * Error codes:
 *   CUDA_ERROR_OUT_OF_MEMORY - Insufficient VRAM for KV cache
 *   CUDA_ERROR_INVALID_PARAMETER - Invalid parameters
 *   CUDA_ERROR_INFERENCE_FAILED - Inference initialization failed
 */
InferenceResult* cuda_inference_start(
    CudaModel* model,
    const char* prompt,
    int max_tokens,
    float temperature,
    uint64_t seed,
    int* error_code
);

/**
 * Generate next token in inference sequence.
 * 
 * @param result Inference result handle
 * @param token_out Output buffer for token text (UTF-8)
 * @param token_buffer_size Size of token_out buffer
 * @param token_index Output parameter for token index
 * @param error_code Output parameter for error code
 * @return true if token generated, false if sequence complete or error
 * 
 * Error codes:
 *   CUDA_ERROR_INFERENCE_FAILED - Kernel execution failed
 *   CUDA_ERROR_OUT_OF_MEMORY - VRAM exhausted during generation
 */
bool cuda_inference_next_token(
    InferenceResult* result,
    char* token_out,
    int token_buffer_size,
    int* token_index,
    int* error_code
);

/**
 * Free inference result and associated resources.
 * 
 * @param result Inference result handle
 */
void cuda_inference_free(InferenceResult* result);

// ============================================================================
// Health & Monitoring
// ============================================================================

/**
 * Check VRAM residency for model weights.
 * 
 * @param model Model handle
 * @param error_code Output parameter for error code
 * @return true if all weights resident in VRAM, false otherwise
 */
bool cuda_check_vram_residency(CudaModel* model, int* error_code);

/**
 * Get current VRAM usage for model.
 * 
 * @param model Model handle
 * @return VRAM bytes used
 */
uint64_t cuda_get_vram_usage(CudaModel* model);

/**
 * Get process-wide VRAM usage.
 * 
 * @param ctx Context handle
 * @return Total VRAM bytes used by this process
 */
uint64_t cuda_get_process_vram_usage(CudaContext* ctx);

/**
 * Check CUDA device health.
 * 
 * @param ctx Context handle
 * @param error_code Output parameter for error code
 * @return true if device is healthy, false otherwise
 */
bool cuda_check_device_health(CudaContext* ctx, int* error_code);

// ============================================================================
// Error Handling
// ============================================================================

/**
 * Get human-readable error message for error code.
 * 
 * @param error_code Error code from any FFI function
 * @return Null-terminated error message string (static storage)
 */
const char* cuda_error_message(int error_code);

#ifdef __cplusplus
}
#endif

#endif // WORKER_FFI_H
```

### Implementation Notes
- **CRITICAL**: This interface is LOCKED after review - no changes without team coordination
- All functions use out-parameters for error codes (no exceptions across FFI boundary)
- Opaque handles prevent Rust from accessing C++ internals
- All string parameters are null-terminated UTF-8
- All pointers can be NULL (functions must check)
- Error codes are positive integers (0 = success)
- Error messages are static strings (no allocation/deallocation needed)
- Thread safety: Each context is single-threaded (no concurrent calls on same context)
- Memory ownership: Rust owns nothing allocated by C++ (must call free functions)

---

## Testing Strategy

### Unit Tests
- Test header compiles with C compiler (gcc, clang)
- Test header compiles with C++ compiler (g++, clang++)
- Test header has include guards
- Test all function declarations are syntactically correct
- Test error code enum has no gaps

### Integration Tests
- None (interface definition only, no implementation yet)

### Manual Verification
1. Compile header: `gcc -c -x c cuda/include/worker_ffi.h -o /tmp/test.o`
2. Compile header: `g++ -c -x c++ cuda/include/worker_ffi.h -o /tmp/test.o`
3. Verify no compilation errors
4. Review interface with team
5. Publish to `coordination/FFI_INTERFACE_LOCKED.md`

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Header file compiles with C and C++ compilers
- [ ] All functions documented with parameter descriptions and error codes
- [ ] Interface reviewed by PM (self-review)
- [ ] **CRITICAL**: Published to `coordination/FFI_INTERFACE_LOCKED.md`
- [ ] Story marked complete in day-tracker.md
- [ ] Llama and GPT teams notified (FFI lock milestone reached)

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §4.2 FFI Boundaries (M0-W-1052)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §4.2 C API Interface (CUDA-4030)
- Related Stories: FT-007 (Rust bindings), FT-008 (error codes)
- FFI Best Practices: https://doc.rust-lang.org/nomicon/ffi.html

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Milestone**: 🔒 **FFI INTERFACE LOCK** (Day 11)

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities (v0.2.0)

**From**: Narration-Core Team  
**Updated**: 2025-10-04 (v0.2.0 - Production Ready with Builder Pattern & Axum Middleware)

### Milestone Event to Narrate

#### FFI Interface Locked (INFO level) ✅
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD};

// NEW v0.2.0: Builder pattern with all modes
Narration::new(ACTOR_WORKER_ORCD, "ffi_lock", "worker_ffi.h")
    .human("FFI interface locked and published to coordination/FFI_INTERFACE_LOCKED.md")
    .cute("Worker and Engine agreed on how to talk! 🤝✨ Contract signed!")
    .story("\"The contract is ready,\" announced Worker. \"Let's build together!\" 📜")
    .emit();
```

### Why This Matters

**FFI interface lock** is critical for:
- 🔓 **Milestone tracking** (unblocks Llama and GPT teams)
- 📝 **Audit trail** (when was interface finalized?)
- 🤝 **Team coordination** (contract agreement)
- 📅 **Timeline tracking** (Day 12 milestone)

**Note**: This is a design/documentation story with minimal runtime events. Narration is primarily for milestone tracking and team coordination.

### New in v0.2.0
- ✅ **7 logging levels** (INFO for milestones)
- ✅ **Story mode** for team coordination events
- ✅ **Cute mode** for celebrating milestones
- ✅ **Auto-injection** of timestamp for audit trail

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Narration Updated**: 2025-10-04 (v0.2.0)

---
Planned by Project Management Team 📋  
*Narration guidance updated by Narration-Core Team 🎀*

---

## 🔍 Testing Team Requirements

**From**: Testing Team (Pre-Development Audit)

### Unit Testing Requirements
- **Test header compiles with C compiler** (gcc, clang)
- **Test header compiles with C++ compiler** (g++, clang++)
- **Test header has include guards** (no multiple definition errors)
- **Test all function declarations are syntactically correct** (parse check)
- **Test error code enum has no gaps** (sequential values)
- **Test opaque handle types are properly declared** (forward declarations)
- **Test extern "C" blocks are correct** (C linkage)

### Integration Testing Requirements
- **None** (interface definition only, no implementation)
- **Manual review required** (FFI contract is CRITICAL)

### BDD Testing Requirements (VERY IMPORTANT)
- **Scenario**: FFI interface is locked and published
  - Given the FFI header is complete
  - When the interface is reviewed
  - Then it should be published to coordination/FFI_INTERFACE_LOCKED.md
  - And Llama and GPT teams should be notified
- **Scenario**: Header compiles in C mode
  - Given worker_ffi.h
  - When compiled with gcc -c -x c
  - Then compilation should succeed
- **Scenario**: Header compiles in C++ mode
  - Given worker_ffi.h
  - When compiled with g++ -c -x c++
  - Then compilation should succeed

### Critical Paths to Test
- Header compilation (C and C++)
- Include guard correctness
- Function signature completeness
- Error code enum completeness
- Documentation completeness

### Edge Cases
- Multiple inclusion of header
- Mixing C and C++ compilation units
- NULL pointer parameters
- Error code out of range

### Interface Contract Validation
- **All functions documented** (parameters, return values, error codes)
- **All opaque types defined** (CudaContext, CudaModel, InferenceResult)
- **All error codes defined** (CUDA_SUCCESS through CUDA_ERROR_UNKNOWN)
- **No breaking changes after lock** (interface is immutable)

---
Test opportunities identified by Testing Team 🔍
