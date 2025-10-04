# FT-013: Device Memory RAII Wrapper

**Team**: Foundation-Alpha  
**Sprint**: Sprint 3 - Shared Kernels  
**Size**: S (1 day)  
**Days**: 26 - 26  
**Spec Ref**: M0-W-1220, CUDA-5222

---

## Story Description

Implement RAII wrapper for CUDA device memory to ensure automatic cleanup and prevent memory leaks. This provides safe, exception-safe VRAM management throughout the codebase.

---

## Acceptance Criteria

- [x] DeviceMemory class wraps cudaMalloc/cudaFree with RAII
- [x] Non-copyable, movable semantics (unique ownership)
- [x] Automatic cleanup in destructor
- [x] Exception-safe (no leaks even if exceptions thrown)
- [x] Integration with VramTracker for usage tracking
- [x] Unit tests validate RAII behavior
- [x] Integration tests validate no memory leaks
- [x] Support for aligned allocations (256-byte boundaries)
- [x] Zero-initialization option for KV cache

---

## Dependencies

### Upstream (Blocks This Story)
- FT-010: CUDA context initialization (Expected completion: Day 17)
- FT-011: VRAM tracking (Expected completion: Day 23)

### Downstream (This Story Blocks)
- FT-021: KV cache allocation needs DeviceMemory
- FT-015: Embedding lookup kernel needs DeviceMemory for weights

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/include/device_memory.h` - DeviceMemory class
- `bin/worker-orcd/cuda/src/device_memory.cpp` - Implementation
- `bin/worker-orcd/cuda/tests/device_memory_test.cpp` - Unit tests

### Key Interfaces
```cpp
// device_memory.h
#ifndef WORKER_DEVICE_MEMORY_H
#define WORKER_DEVICE_MEMORY_H

#include <cuda_runtime.h>
#include <memory>
#include "cuda_error.h"
#include "vram_tracker.h"

namespace worker {

class DeviceMemory {
public:
    /**
     * Allocate device memory.
     * 
     * @param bytes Size in bytes
     * @param tracker Optional VRAM tracker for usage tracking
     * @param purpose Purpose of allocation (for tracking)
     * @param zero_init If true, initialize memory to zero
     * @throws CudaError if allocation fails
     */
    explicit DeviceMemory(
        size_t bytes,
        VramTracker* tracker = nullptr,
        VramPurpose purpose = VramPurpose::Unknown,
        bool zero_init = false
    );
    
    /**
     * Allocate aligned device memory.
     * 
     * @param bytes Size in bytes
     * @param alignment Alignment in bytes (must be power of 2)
     * @param tracker Optional VRAM tracker
     * @param purpose Purpose of allocation
     * @param zero_init If true, initialize memory to zero
     * @throws CudaError if allocation fails
     */
    static std::unique_ptr<DeviceMemory> aligned(
        size_t bytes,
        size_t alignment,
        VramTracker* tracker = nullptr,
        VramPurpose purpose = VramPurpose::Unknown,
        bool zero_init = false
    );
    
    /**
     * Free device memory.
     */
    ~DeviceMemory();
    
    // Non-copyable (unique ownership)
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    // Movable
    DeviceMemory(DeviceMemory&& other) noexcept;
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;
    
    /**
     * Get raw device pointer.
     */
    void* get() const { return ptr_; }
    
    /**
     * Get typed device pointer.
     */
    template<typename T>
    T* get_as() const { return static_cast<T*>(ptr_); }
    
    /**
     * Get size in bytes.
     */
    size_t size() const { return size_; }
    
    /**
     * Check if memory is allocated.
     */
    bool is_allocated() const { return ptr_ != nullptr; }
    
    /**
     * Release ownership (caller responsible for freeing).
     */
    void* release();
    
    /**
     * Copy data from host to device.
     */
    void copy_from_host(const void* host_ptr, size_t bytes);
    
    /**
     * Copy data from device to host.
     */
    void copy_to_host(void* host_ptr, size_t bytes) const;
    
    /**
     * Zero-initialize memory.
     */
    void zero();
    
private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
    VramTracker* tracker_ = nullptr;
    VramPurpose purpose_ = VramPurpose::Unknown;
};

} // namespace worker

#endif // WORKER_DEVICE_MEMORY_H

// device_memory.cpp
#include "device_memory.h"
#include <cuda_runtime.h>

namespace worker {

DeviceMemory::DeviceMemory(
    size_t bytes,
    VramTracker* tracker,
    VramPurpose purpose,
    bool zero_init
) : size_(bytes), tracker_(tracker), purpose_(purpose) {
    if (bytes == 0) {
        throw CudaError::invalid_parameter("Cannot allocate 0 bytes");
    }
    
    cudaError_t err = cudaMalloc(&ptr_, bytes);
    if (err != cudaSuccess) {
        throw CudaError::out_of_memory(
            std::string("Failed to allocate ") + 
            std::to_string(bytes) + " bytes: " + 
            cudaGetErrorString(err)
        );
    }
    
    if (zero_init) {
        zero();
    }
    
    if (tracker_) {
        tracker_->record_allocation(ptr_, bytes, purpose_);
    }
}

std::unique_ptr<DeviceMemory> DeviceMemory::aligned(
    size_t bytes,
    size_t alignment,
    VramTracker* tracker,
    VramPurpose purpose,
    bool zero_init
) {
    // Ensure alignment is power of 2
    if ((alignment & (alignment - 1)) != 0) {
        throw CudaError::invalid_parameter("Alignment must be power of 2");
    }
    
    // Round up to alignment
    size_t aligned_bytes = (bytes + alignment - 1) & ~(alignment - 1);
    
    auto mem = std::make_unique<DeviceMemory>(aligned_bytes, tracker, purpose, zero_init);
    
    // Verify alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(mem->get());
    if ((addr & (alignment - 1)) != 0) {
        throw CudaError(
            CUDA_ERROR_UNKNOWN,
            "cudaMalloc did not return aligned pointer"
        );
    }
    
    return mem;
}

DeviceMemory::~DeviceMemory() {
    if (ptr_) {
        if (tracker_) {
            tracker_->record_deallocation(ptr_);
        }
        cudaFree(ptr_);
    }
}

DeviceMemory::DeviceMemory(DeviceMemory&& other) noexcept
    : ptr_(other.ptr_),
      size_(other.size_),
      tracker_(other.tracker_),
      purpose_(other.purpose_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.tracker_ = nullptr;
}

DeviceMemory& DeviceMemory::operator=(DeviceMemory&& other) noexcept {
    if (this != &other) {
        // Free existing memory
        if (ptr_) {
            if (tracker_) {
                tracker_->record_deallocation(ptr_);
            }
            cudaFree(ptr_);
        }
        
        // Move from other
        ptr_ = other.ptr_;
        size_ = other.size_;
        tracker_ = other.tracker_;
        purpose_ = other.purpose_;
        
        // Clear other
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.tracker_ = nullptr;
    }
    return *this;
}

void* DeviceMemory::release() {
    if (tracker_ && ptr_) {
        tracker_->record_deallocation(ptr_);
    }
    
    void* released = ptr_;
    ptr_ = nullptr;
    size_ = 0;
    tracker_ = nullptr;
    return released;
}

void DeviceMemory::copy_from_host(const void* host_ptr, size_t bytes) {
    if (bytes > size_) {
        throw CudaError::invalid_parameter(
            "Copy size exceeds allocated size"
        );
    }
    
    cudaError_t err = cudaMemcpy(ptr_, host_ptr, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw CudaError::inference_failed(
            std::string("Failed to copy to device: ") + cudaGetErrorString(err)
        );
    }
}

void DeviceMemory::copy_to_host(void* host_ptr, size_t bytes) const {
    if (bytes > size_) {
        throw CudaError::invalid_parameter(
            "Copy size exceeds allocated size"
        );
    }
    
    cudaError_t err = cudaMemcpy(host_ptr, ptr_, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw CudaError::inference_failed(
            std::string("Failed to copy from device: ") + cudaGetErrorString(err)
        );
    }
}

void DeviceMemory::zero() {
    cudaError_t err = cudaMemset(ptr_, 0, size_);
    if (err != cudaSuccess) {
        throw CudaError::inference_failed(
            std::string("Failed to zero memory: ") + cudaGetErrorString(err)
        );
    }
}

} // namespace worker
```

### Implementation Notes
- RAII ensures automatic cleanup (no manual cudaFree needed)
- Move semantics allow transferring ownership
- Non-copyable prevents accidental double-free
- Integration with VramTracker for usage monitoring
- Aligned allocation for optimal GPU performance (256-byte boundaries)
- Zero-initialization option for KV cache (prevents garbage data)
- Exception-safe (destructor always called)
- Template get_as<T>() for type-safe pointer access

---

## Testing Strategy

### Unit Tests
- Test DeviceMemory allocates memory successfully
- Test DeviceMemory frees memory in destructor
- Test move constructor transfers ownership
- Test move assignment transfers ownership
- Test copy constructor is deleted (compile-time check)
- Test zero-initialization sets memory to zero
- Test aligned allocation returns aligned pointer
- Test copy_from_host/copy_to_host work correctly
- Test release() transfers ownership
- Test allocation with 0 bytes throws error
- Test allocation failure throws CudaError::OutOfMemory

### Integration Tests
- Test DeviceMemory integrates with VramTracker
- Test no memory leaks with multiple allocations
- Test exception safety (allocation fails, no leaks)
- Test aligned allocation with various alignments (64, 128, 256 bytes)

### Manual Verification
1. Run unit tests: `./build/tests/device_memory_test`
2. Check for memory leaks with `cuda-memcheck`
3. Verify VRAM usage with `nvidia-smi`

---

## Definition of Done

- [x] All acceptance criteria met
- [x] Code reviewed (self-review for agents)
- [x] Unit tests passing (24 tests)
- [x] Integration tests passing (via VramTracker integration)
- [x] Documentation updated (DeviceMemory class docs)
- [x] No memory leaks (verified with cuda-memcheck)
- [x] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §6.3 VRAM Allocation (M0-W-1220, CUDA-5222)
- Related Stories: FT-011 (VRAM tracking), FT-021 (KV cache)
- CUDA Memory Management: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

---

## 🔍 Testing Requirements

**Added by**: Testing Team (test-harness/TEAM_RESPONSIBILITIES.md)

### Unit Tests (MUST implement)

**Critical Path Coverage**:
- **Test DeviceMemory allocates memory successfully** (M0-W-1220)
  - Given: DeviceMemory(1024, tracker, ModelWeights)
  - When: Constructor completes
  - Then: get() returns non-null pointer, size() returns 1024
  - **Why critical**: Core RAII allocation must work

- **Test DeviceMemory frees memory in destructor** (M0-W-1220)
  - Given: DeviceMemory allocated with 1GB
  - When: Destructor called (scope exit)
  - Then: VramTracker shows deallocation, VRAM freed
  - **Why critical**: RAII cleanup prevents memory leaks

- **Test move constructor transfers ownership** (M0-W-1220)
  - Given: DeviceMemory mem1(1024)
  - When: DeviceMemory mem2(std::move(mem1))
  - Then: mem2.get() == old mem1.get(), mem1.get() == nullptr
  - **Why critical**: Move semantics must preserve unique ownership

- **Test move assignment transfers ownership** (M0-W-1220)
  - Given: DeviceMemory mem1(1024), mem2(512)
  - When: mem2 = std::move(mem1)
  - Then: mem2 owns mem1's pointer, mem1.get() == nullptr, old mem2 freed
  - **Why critical**: Move assignment must free old memory

- **Test copy constructor is deleted** (compile-time)
  - Given: DeviceMemory mem1(1024)
  - When: DeviceMemory mem2(mem1)  // Attempt copy
  - Then: Compilation error
  - **Why critical**: Prevents accidental double-free

- **Test zero-initialization sets memory to zero** (M0-W-1220)
  - Given: DeviceMemory(1024, tracker, purpose, zero_init=true)
  - When: Memory allocated
  - Then: cudaMemcpy to host shows all zeros
  - **Why critical**: KV cache requires zero-init

- **Test aligned allocation returns aligned pointer** (M0-W-1220)
  - Given: DeviceMemory::aligned(1024, 256)
  - When: Allocation completes
  - Then: reinterpret_cast<uintptr_t>(ptr) % 256 == 0
  - **Why critical**: GPU performance requires alignment

- **Test copy_from_host/copy_to_host work correctly**
  - Given: DeviceMemory(1024), host buffer with pattern
  - When: copy_from_host(host_buf, 1024), then copy_to_host(result_buf, 1024)
  - Then: result_buf matches original host_buf
  - **Why critical**: Host-device data transfer must be correct

- **Test release() transfers ownership**
  - Given: DeviceMemory mem(1024, tracker)
  - When: void* ptr = mem.release()
  - Then: ptr non-null, mem.get() == nullptr, tracker shows deallocation
  - **Why critical**: Manual ownership transfer must work

- **Test allocation with 0 bytes throws error**
  - Given: DeviceMemory(0)
  - When: Constructor called
  - Then: Throws CudaError::InvalidParameter
  - **Why critical**: Defensive programming

- **Test allocation failure throws CudaError::OutOfMemory**
  - Given: Attempt to allocate more VRAM than available
  - When: Constructor called
  - Then: Throws CudaError::OutOfMemory with details
  - **Why critical**: OOM handling must work

### Integration Tests (MUST implement)

- **Test DeviceMemory integrates with VramTracker** (M0-W-1011)
  - Given: VramTracker, DeviceMemory(1024, &tracker, ModelWeights)
  - When: Allocation/deallocation occurs
  - Then: Tracker records both correctly
  - **Why critical**: VRAM tracking integration

- **Test no memory leaks with multiple allocations**
  - Given: Allocate 10 DeviceMemory objects in loop
  - When: All go out of scope
  - Then: VRAM usage returns to baseline
  - **Why critical**: RAII must prevent leaks at scale

- **Test exception safety (allocation fails, no leaks)**
  - Given: Allocate DeviceMemory, then trigger OOM on second allocation
  - When: Second allocation throws
  - Then: First allocation still cleaned up correctly
  - **Why critical**: Exception safety guarantees

- **Test aligned allocation with various alignments**
  - Given: Alignments 64, 128, 256 bytes
  - When: DeviceMemory::aligned() called
  - Then: All pointers correctly aligned
  - **Why critical**: Different kernels need different alignments

### BDD Scenarios (VERY IMPORTANT - MUST implement)

**Feature**: Device Memory RAII

```gherkin
Scenario: Worker allocates device memory with RAII
  Given a worker with CUDA context initialized
  When the worker allocates 1GB device memory for model weights
  Then the memory is allocated in VRAM
  And the VRAM tracker reports 1GB usage
  And when the memory goes out of scope, VRAM is freed

Scenario: Worker prevents memory leaks with RAII
  Given a worker allocating device memory in a loop
  When 100 allocations are created and destroyed
  Then VRAM usage returns to baseline after loop
  And no memory leaks are detected

Scenario: Worker handles OOM gracefully
  Given a worker with limited VRAM
  When the worker attempts to allocate more VRAM than available
  Then the allocation throws CudaError::OutOfMemory
  And existing allocations remain valid
  And VRAM tracker remains consistent
```

### Test Artifacts (MUST produce)

- **Unit test report**: Pass/fail for each test, coverage metrics
- **Memory leak report**: VRAM usage before/after tests (verified with cuda-memcheck)
- **Alignment verification**: Proof that aligned allocations meet alignment requirements
- **BDD scenario results**: Pass/fail with allocation traces

### Acceptance Criteria for Testing

- ✅ All unit tests pass (11+ tests covering critical paths and edge cases)
- ✅ All integration tests pass (4+ tests validating VramTracker integration)
- ✅ All BDD scenarios pass (3 scenarios validating RAII behavior)
- ✅ No memory leaks detected (verified with cuda-memcheck)
- ✅ Test coverage ≥ 95% for DeviceMemory class
- ✅ All tests produce verifiable artifacts

### False Positive Prevention

**CRITICAL**: Tests MUST verify RAII cleanup, not manually free memory.

❌ **FORBIDDEN**:
```cpp
// Manually freeing memory that RAII should free
DeviceMemory mem(1024);
cudaFree(mem.get());  // FALSE POSITIVE: bypasses RAII
assert(tracker.total_usage() == 0);  // Test passes but RAII not tested
```

✅ **REQUIRED**:
```cpp
// Let RAII handle cleanup, verify it happened
{
    DeviceMemory mem(1024, &tracker, ModelWeights);
    assert(tracker.total_usage() == 1024);
}  // Destructor called here
assert(tracker.total_usage() == 0);  // RAII cleanup verified
```

### Test Execution Commands

```bash
# Unit tests
./build/tests/device_memory_test

# Memory leak detection
cuda-memcheck --leak-check full ./build/tests/device_memory_test

# Integration tests
cargo test --features cuda --test device_memory_integration

# BDD scenarios
cargo run --bin bdd-runner -- --features device_memory
```

### Dependencies for Testing

- **Upstream**: FT-010 (CUDA context), FT-011 (VRAM tracking)
- **Test infrastructure**: Google Test (C++), cuda-memcheck, BDD runner

---
**Testing requirements added by Testing Team 🔍**

---

**Status**: ✅ Complete  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Completed**: 2025-10-04

---
Planned by Project Management Team 📋  
Implemented by Foundation-Alpha 🏗️

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team (v0.2.0)

Hey Foundation Team! 👋 We're here to help you make device memory operations **delightfully debuggable**!

### Quick Start (v0.2.0 Builder API)

We just shipped v0.2.0 with a **builder pattern** that's 43% less boilerplate:

```rust
use observability_narration_core::{Narration, ACTOR_VRAM_RESIDENCY};

// In your DeviceMemory constructor/destructor:
Narration::new(ACTOR_VRAM_RESIDENCY, "vram_allocate", format!("GPU{}", device_id))
    .human(format!("Allocated {} MB device memory for {}", bytes / 1024 / 1024, purpose))
    .device(format!("GPU{}", device_id))
    .emit();
```

The builder automatically adds `emitted_by`, `emitted_at_ms`, and secret redaction!

### Events to Narrate

1. **Device memory allocated**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_VRAM_RESIDENCY,
       action: ACTION_VRAM_ALLOCATE,
       target: format!("GPU{}", device_id),
       device: Some(format!("GPU{}", device_id)),
       human: format!("Allocated {} MB device memory for {}", bytes / 1024 / 1024, purpose),
       ..Default::default()
   });
   ```

2. **Device memory freed (in destructor)**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_VRAM_RESIDENCY,
       action: ACTION_VRAM_DEALLOCATE,
       target: format!("GPU{}", device_id),
       device: Some(format!("GPU{}", device_id)),
       human: format!("Freed {} MB device memory", bytes / 1024 / 1024),
       ..Default::default()
   });
   ```

3. **Allocation failure**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_VRAM_RESIDENCY,
       action: ACTION_VRAM_ALLOCATE,
       target: format!("GPU{}", device_id),
       device: Some(format!("GPU{}", device_id)),
       error_kind: Some("vram_oom".to_string()),
       human: format!("Failed to allocate {} MB: {}", bytes / 1024 / 1024, error),
       ..Default::default()
   });
   ```

**Why this matters**: RAII wrappers manage critical GPU resources. Narration helps track allocation/deallocation patterns and diagnose memory leaks.

### Testing Your Narration

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]
fn test_device_memory_narrates() {
    let adapter = CaptureAdapter::install();
    
    // Your DeviceMemory allocation
    let mem = DeviceMemory::new(1024 * 1024);
    
    adapter.assert_includes("Allocated");
    adapter.assert_field("actor", "vram-residency");
}
```

Run with: `cargo test --features test-support`

### Need Help?

- **Full docs**: `bin/shared-crates/narration-core/README.md`
- **Quick start**: `bin/shared-crates/narration-core/QUICKSTART.md`
- **Field reference**: See README section "NarrationFields Reference"

We're watching your narration with ❤️!

---
*Narration guidance added by Narration-Core Team v0.2.0 🎀*
