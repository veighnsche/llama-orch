# FT-010: CUDA Context Initialization

**Team**: Foundation-Alpha  
**Sprint**: Sprint 2 - FFI Layer  
**Size**: M (2 days)  
**Days**: 16 - 17  
**Spec Ref**: M0-W-1010, M0-W-1400, CUDA-5101, CUDA-5120

---

## Story Description

Implement CUDA context initialization in C++ with VRAM-only enforcement. This establishes the CUDA runtime environment and configures device settings for deterministic, VRAM-only operation.

---

## Acceptance Criteria

- [ ] Context class initializes CUDA device with cudaSetDevice()
- [ ] Disables Unified Memory via cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0)
- [ ] Sets cache config via cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)
- [ ] Validates device ID against cudaGetDeviceCount()
- [ ] Retrieves device properties via cudaGetDeviceProperties()
- [ ] Unit tests validate context initialization
- [ ] Integration tests validate VRAM-only enforcement
- [ ] Error handling for invalid device ID
- [ ] Cleanup via cudaDeviceReset() in destructor

---

## Dependencies

### Upstream (Blocks This Story)
- FT-007: Rust FFI bindings (Expected completion: Day 13)
- FT-008: C++ error code system (Expected completion: Day 14)
- FT-009: Rust error conversion (Expected completion: Day 15)

### Downstream (This Story Blocks)
- FT-013: Device memory RAII needs context
- FT-024: Integration tests need context initialization
- **CRITICAL**: Llama and GPT teams need context for CUDA work

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/include/context.h` - Context class header
- `bin/worker-orcd/cuda/src/context.cpp` - Context implementation
- `bin/worker-orcd/cuda/tests/context_test.cpp` - Unit tests

### Key Interfaces
```cpp
// context.h
#ifndef WORKER_CONTEXT_H
#define WORKER_CONTEXT_H

#include <cuda_runtime.h>
#include <memory>
#include "cuda_error.h"

namespace worker {

class Context {
public:
    /**
     * Initialize CUDA context for specified device.
     * 
     * @param gpu_device Device ID (0, 1, 2, ...)
     * @throws CudaError if device invalid or initialization fails
     */
    explicit Context(int gpu_device);
    
    /**
     * Cleanup CUDA context.
     */
    ~Context();
    
    // Non-copyable, non-movable (owns CUDA context)
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;
    
    /**
     * Get device ID.
     */
    int device() const { return device_; }
    
    /**
     * Get device properties.
     */
    const cudaDeviceProp& properties() const { return props_; }
    
    /**
     * Get device name.
     */
    const char* device_name() const { return props_.name; }
    
    /**
     * Get compute capability (e.g., 86 for SM_86).
     */
    int compute_capability() const {
        return props_.major * 10 + props_.minor;
    }
    
    /**
     * Get total VRAM in bytes.
     */
    size_t total_vram() const { return props_.totalGlobalMem; }
    
    /**
     * Get free VRAM in bytes.
     */
    size_t free_vram() const;
    
    /**
     * Get number of available CUDA devices.
     */
    static int device_count();
    
private:
    int device_;
    cudaDeviceProp props_;
};

} // namespace worker

#endif // WORKER_CONTEXT_H

// context.cpp
#include "context.h"
#include <cuda_runtime.h>

namespace worker {

Context::Context(int gpu_device) : device_(gpu_device) {
    // Check device count
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        throw CudaError::invalid_device(
            std::string("Failed to get device count: ") + cudaGetErrorString(err)
        );
    }
    
    // Validate device ID
    if (gpu_device < 0 || gpu_device >= device_count) {
        throw CudaError::invalid_device(
            "Device ID " + std::to_string(gpu_device) + 
            " out of range (0-" + std::to_string(device_count - 1) + ")"
        );
    }
    
    // Set device
    err = cudaSetDevice(gpu_device);
    if (err != cudaSuccess) {
        throw CudaError::invalid_device(
            std::string("Failed to set device: ") + cudaGetErrorString(err)
        );
    }
    
    // Get device properties
    err = cudaGetDeviceProperties(&props_, gpu_device);
    if (err != cudaSuccess) {
        throw CudaError::invalid_device(
            std::string("Failed to get device properties: ") + cudaGetErrorString(err)
        );
    }
    
    // Enforce VRAM-only mode: Disable Unified Memory
    err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0);
    if (err != cudaSuccess) {
        throw CudaError(
            CUDA_ERROR_UNKNOWN,
            std::string("Failed to disable UMA: ") + cudaGetErrorString(err)
        );
    }
    
    // Set cache config for compute workloads
    err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    if (err != cudaSuccess) {
        // Non-fatal, log warning but continue
        // (Some devices don't support cache config)
    }
}

Context::~Context() {
    // Reset device (frees all allocations)
    cudaDeviceReset();
}

size_t Context::free_vram() const {
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        return 0;
    }
    return free_bytes;
}

int Context::device_count() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

} // namespace worker

// FFI wrapper
extern "C" CudaContext* cuda_init(int gpu_device, int* error_code) {
    try {
        auto ctx = std::make_unique<worker::Context>(gpu_device);
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

extern "C" void cuda_destroy(CudaContext* ctx) {
    if (ctx) {
        delete reinterpret_cast<worker::Context*>(ctx);
    }
}

extern "C" int cuda_get_device_count() {
    return worker::Context::device_count();
}
```

### Implementation Notes
- Context owns CUDA device state (non-copyable, non-movable)
- cudaDeviceReset() in destructor frees all allocations
- UMA disabled via cudaLimitMallocHeapSize = 0 (enforces VRAM-only)
- Cache config set to prefer L1 for compute workloads
- Device properties cached for fast access
- free_vram() uses cudaMemGetInfo() for current VRAM state
- Error handling converts CUDA errors to CudaError exceptions
- FFI wrapper catches all exceptions and converts to error codes

---

## Testing Strategy

### Unit Tests
- Test Context constructor with valid device ID
- Test Context constructor with invalid device ID throws error
- Test Context::device() returns correct device ID
- Test Context::device_name() returns non-empty string
- Test Context::compute_capability() returns valid SM version
- Test Context::total_vram() returns positive value
- Test Context::free_vram() returns value <= total_vram
- Test Context::device_count() returns positive value
- Test destructor calls cudaDeviceReset()

### Integration Tests
- Test UMA is disabled after context init (verify with cudaPointerGetAttributes)
- Test cache config is set (query with cudaDeviceGetCacheConfig)
- Test multiple contexts on different devices (if multi-GPU available)
- Test context cleanup frees VRAM (check free_vram before/after)

### Manual Verification
1. Build CUDA code: `cmake --build build/`
2. Run unit tests: `./build/tests/context_test`
3. Check device properties: `nvidia-smi`
4. Verify VRAM usage: `nvidia-smi dmon -s m`

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (9+ tests)
- [ ] Integration tests passing (4+ tests)
- [ ] Documentation updated (Context class docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §2.2 CUDA Implementation (M0-W-1010)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §9.1 Context Management (M0-W-1400, CUDA-5101)
- Related Stories: FT-007 (FFI bindings), FT-013 (device memory)
- CUDA Runtime API: https://docs.nvidia.com/cuda/cuda-runtime-api/

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Milestone**: 🔓 **FFI IMPLEMENTATION COMPLETE** (Day 17)

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities (v0.2.0)

**From**: Narration-Core Team  
**Updated**: 2025-10-04 (v0.2.0 - Production Ready with Builder Pattern & Axum Middleware)

### Critical Events to Narrate

#### 1. CUDA Context Initialization Started (INFO level) ✅
```rust
use observability_narration_core::{Narration, ACTOR_INFERENCE_ENGINE};

// NEW v0.2.0: Builder pattern
Narration::new(ACTOR_INFERENCE_ENGINE, "cuda_init", &format!("GPU{}", device_id))
    .human(format!("Initializing CUDA context on GPU{}", device_id))
    .device(&format!("GPU{}", device_id))
    .cute(format!("Worker is waking up GPU{}! 🚀✨", device_id))
    .emit();
```

#### 2. CUDA Context Ready (INFO level) ✅
```rust
use observability_narration_core::{Narration, ACTOR_INFERENCE_ENGINE};

// NEW v0.2.0: Builder pattern
Narration::new(ACTOR_INFERENCE_ENGINE, "cuda_ready", &format!("GPU{}", device_id))
    .human(format!("CUDA context ready on GPU{}: {} MB free ({} ms)", device_id, free_vram_mb, elapsed.as_millis()))
    .device(&format!("GPU{}", device_id))
    .duration_ms(elapsed.as_millis() as u64)
    .cute(format!("GPU{} is awake and ready! 💚 {} MB of cozy memory available!", device_id, free_vram_mb))
    .emit();
```

**Note**: `vram_total_mb` and `vram_free_mb` are not standard fields. Use custom fields or `human` text.

#### 3. CUDA Initialization Failure (ERROR level) 🚨
```rust
use observability_narration_core::{Narration, ACTOR_INFERENCE_ENGINE};

// NEW v0.2.0: Builder with error level
Narration::new(ACTOR_INFERENCE_ENGINE, "cuda_init", &format!("GPU{}", device_id))
    .human(format!("CUDA initialization failed on GPU{}: {}", device_id, error_message))
    .device(&format!("GPU{}", device_id))
    .error_kind(&error_code)
    .emit_error();  // ← ERROR level
```

#### 4. UMA Disabled (DEBUG level) 🔍
```rust
use observability_narration_core::{Narration, ACTOR_INFERENCE_ENGINE};

// NEW v0.2.0: Builder with debug level
Narration::new(ACTOR_INFERENCE_ENGINE, "cuda_config", &format!("GPU{}", device_id))
    .human(format!("Disabled UMA on GPU{} (VRAM-only mode)", device_id))
    .device(&format!("GPU{}", device_id))
    .emit_debug();  // ← DEBUG level
```

### Testing with CaptureAdapter

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]
fn test_cuda_init_narration() {
    let adapter = CaptureAdapter::install();
    
    // Initialize CUDA context
    let ctx = CudaContext::new(0)?;
    
    // Assert initialization narrated
    adapter.assert_includes("CUDA context ready");
    adapter.assert_field("action", "cuda_ready");
    adapter.assert_field("device", "GPU0");
    
    // Verify VRAM info captured
    let captured = adapter.captured();
    assert!(captured.iter().any(|e| e.vram_free_mb.is_some()));
}
```

### Why This Matters

**CUDA initialization events** are critical for:
- 🚀 **Startup tracking** (worker readiness)
- 🐛 **Driver debugging** (CUDA version, driver issues)
- 💾 **VRAM tracking** (available memory)
- 🚨 **Alerting** on initialization failures
- 📈 **Performance** (init time tracking)

### New in v0.2.0
- ✅ **7 logging levels** (INFO for success, ERROR for failures, DEBUG for config)
- ✅ **VRAM tracking** in narration fields (`vram_total_mb`, `vram_free_mb`)
- ✅ **Duration tracking** for initialization time
- ✅ **Device info** in narration fields
- ✅ **Test assertions** for CUDA events

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
- **Test Context constructor with valid device ID** (device 0)
- **Test Context constructor with invalid device ID throws error** (device -1, device 999)
- **Test Context::device() returns correct device ID** (getter)
- **Test Context::device_name() returns non-empty string** (device properties)
- **Test Context::compute_capability() returns valid SM version** (e.g., 86 for SM_86)
- **Test Context::total_vram() returns positive value** (VRAM query)
- **Test Context::free_vram() returns value <= total_vram** (VRAM availability)
- **Test Context::device_count() returns positive value** (device enumeration)
- **Test destructor calls cudaDeviceReset()** (cleanup verification)
- **Property test**: All valid device IDs (0 to device_count-1) accepted

### Integration Testing Requirements
- **Test UMA is disabled after context init** (verify with cudaPointerGetAttributes)
- **Test cache config is set** (query with cudaDeviceGetCacheConfig)
- **Test multiple contexts on different devices** (multi-GPU if available)
- **Test context cleanup frees VRAM** (check free_vram before/after)
- **Test VRAM-only enforcement** (no unified memory allocations)

### BDD Testing Requirements (VERY IMPORTANT)
- **Scenario**: CUDA context initialization succeeds
  - Given a valid GPU device ID 0
  - When I create a Context
  - Then the context should initialize successfully
  - And device properties should be accessible
  - And UMA should be disabled
  - And cache config should be set to prefer L1
- **Scenario**: Invalid device ID rejected
  - Given an invalid device ID 999
  - When I attempt to create a Context
  - Then a CudaError should be thrown
  - And error message should indicate "out of range"
- **Scenario**: Context cleanup frees VRAM
  - Given a Context with allocated VRAM
  - When the Context is destroyed
  - Then cudaDeviceReset() should be called
  - And all VRAM should be freed

### Critical Paths to Test
- CUDA device initialization (cudaSetDevice)
- UMA disabling (cudaLimitMallocHeapSize = 0)
- Cache config (cudaFuncCachePreferL1)
- Device properties query
- Context cleanup (cudaDeviceReset)

### Edge Cases
- Device ID 0 (first GPU)
- Device ID = device_count - 1 (last GPU)
- Device ID out of range (negative, >= device_count)
- Multiple contexts on same device (should fail or serialize)
- Context destruction during CUDA operation
- VRAM exhaustion during init

---
Test opportunities identified by Testing Team 🔍
