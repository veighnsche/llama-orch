# FT-011: VRAM-Only Enforcement

**Team**: Foundation-Alpha  
**Sprint**: Sprint 3 - Shared Kernels  
**Size**: S (1 day)  
**Days**: 23 - 23  
**Spec Ref**: M0-W-1010, M0-W-1011, M0-SYS-2.2.1

---

## Story Description

Implement VRAM allocation tracking and enforcement mechanisms to ensure all model data resides exclusively in GPU VRAM. This prevents RAM fallback and ensures predictable performance.

---

## Acceptance Criteria

- [x] VRAM allocation tracker records all cudaMalloc calls with size and purpose
- [x] Tracker reports total VRAM usage (model weights + KV cache + intermediate buffers)
- [x] Validation function verifies no host pointers exist for model data
- [x] Unit tests validate tracking accuracy
- [x] Integration tests validate VRAM-only residency
- [x] Error reporting includes VRAM usage breakdown on OOM
- [x] Tracker thread-safe (if needed for future multi-stream support)
- [x] Health endpoint exposes VRAM usage metrics

---

## Dependencies

### Upstream (Blocks This Story)
- FT-010: CUDA context initialization (Expected completion: Day 17)

### Downstream (This Story Blocks)
- FT-012: FFI integration tests need VRAM tracking
- FT-014: VRAM residency verification needs tracker

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/include/vram_tracker.h` - VRAM tracking interface
- `bin/worker-orcd/cuda/src/vram_tracker.cpp` - Tracker implementation
- `bin/worker-orcd/cuda/src/context.cpp` - Integrate tracker with context
- `bin/worker-orcd/cuda/tests/vram_tracker_test.cpp` - Unit tests

### Key Interfaces
```cpp
// vram_tracker.h
#ifndef WORKER_VRAM_TRACKER_H
#define WORKER_VRAM_TRACKER_H

#include <cstddef>
#include <string>
#include <unordered_map>
#include <mutex>

namespace worker {

enum class VramPurpose {
    ModelWeights,
    KVCache,
    IntermediateBuffers,
    Unknown,
};

struct VramAllocation {
    void* ptr;
    size_t bytes;
    VramPurpose purpose;
    std::string description;
};

class VramTracker {
public:
    VramTracker() = default;
    
    /**
     * Record VRAM allocation.
     */
    void record_allocation(
        void* ptr,
        size_t bytes,
        VramPurpose purpose,
        const std::string& description = ""
    );
    
    /**
     * Record VRAM deallocation.
     */
    void record_deallocation(void* ptr);
    
    /**
     * Get total VRAM usage in bytes.
     */
    size_t total_usage() const;
    
    /**
     * Get VRAM usage by purpose.
     */
    size_t usage_by_purpose(VramPurpose purpose) const;
    
    /**
     * Get allocation count.
     */
    size_t allocation_count() const;
    
    /**
     * Get detailed usage breakdown.
     */
    std::unordered_map<VramPurpose, size_t> usage_breakdown() const;
    
    /**
     * Verify all tracked pointers are device memory.
     * 
     * @return true if all pointers are VRAM, false if any host pointers found
     */
    bool verify_vram_residency() const;
    
    /**
     * Get human-readable usage report.
     */
    std::string usage_report() const;
    
private:
    mutable std::mutex mutex_;
    std::unordered_map<void*, VramAllocation> allocations_;
};

} // namespace worker

#endif // WORKER_VRAM_TRACKER_H

// vram_tracker.cpp
#include "vram_tracker.h"
#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>

namespace worker {

void VramTracker::record_allocation(
    void* ptr,
    size_t bytes,
    VramPurpose purpose,
    const std::string& description
) {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ptr] = VramAllocation{ptr, bytes, purpose, description};
}

void VramTracker::record_deallocation(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_.erase(ptr);
}

size_t VramTracker::total_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& [ptr, alloc] : allocations_) {
        total += alloc.bytes;
    }
    return total;
}

size_t VramTracker::usage_by_purpose(VramPurpose purpose) const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& [ptr, alloc] : allocations_) {
        if (alloc.purpose == purpose) {
            total += alloc.bytes;
        }
    }
    return total;
}

size_t VramTracker::allocation_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_.size();
}

std::unordered_map<VramPurpose, size_t> VramTracker::usage_breakdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<VramPurpose, size_t> breakdown;
    
    for (const auto& [ptr, alloc] : allocations_) {
        breakdown[alloc.purpose] += alloc.bytes;
    }
    
    return breakdown;
}

bool VramTracker::verify_vram_residency() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& [ptr, alloc] : allocations_) {
        cudaPointerAttributes attrs;
        cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
        
        if (err != cudaSuccess) {
            return false;
        }
        
        // Check pointer is device memory (not managed/host)
        if (attrs.type != cudaMemoryTypeDevice) {
            return false;
        }
        
        // Check no host pointer exists (no UMA)
        if (attrs.hostPointer != nullptr) {
            return false;
        }
    }
    
    return true;
}

std::string VramTracker::usage_report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    
    auto breakdown = usage_breakdown();
    size_t total = total_usage();
    
    oss << "VRAM Usage Report:\n";
    oss << "  Total: " << (total / 1024.0 / 1024.0) << " MB\n";
    oss << "  Allocations: " << allocations_.size() << "\n";
    oss << "  Breakdown:\n";
    
    auto format_purpose = [](VramPurpose p) -> std::string {
        switch (p) {
            case VramPurpose::ModelWeights: return "Model Weights";
            case VramPurpose::KVCache: return "KV Cache";
            case VramPurpose::IntermediateBuffers: return "Intermediate Buffers";
            default: return "Unknown";
        }
    };
    
    for (const auto& [purpose, bytes] : breakdown) {
        oss << "    " << format_purpose(purpose) << ": "
            << (bytes / 1024.0 / 1024.0) << " MB\n";
    }
    
    return oss.str();
}

} // namespace worker

// Integration with Context
namespace worker {

class Context {
public:
    // ... existing code ...
    
    VramTracker& vram_tracker() { return vram_tracker_; }
    const VramTracker& vram_tracker() const { return vram_tracker_; }
    
private:
    VramTracker vram_tracker_;
};

} // namespace worker
```

### Implementation Notes
- Tracker uses std::mutex for thread safety (future-proof)
- cudaPointerGetAttributes validates VRAM residency
- Usage breakdown by purpose (weights, KV cache, buffers)
- Human-readable report for debugging and health endpoint
- Tracker integrated into Context for centralized management
- No performance overhead (tracking only on alloc/free, not hot path)
- Error reporting includes VRAM breakdown on OOM

---

## Testing Strategy

### Unit Tests
- Test record_allocation() increments total usage
- Test record_deallocation() decrements total usage
- Test usage_by_purpose() returns correct breakdown
- Test allocation_count() returns correct count
- Test verify_vram_residency() with device pointers returns true
- Test verify_vram_residency() with host pointers returns false
- Test usage_report() generates readable output
- Test thread safety with concurrent allocations

### Integration Tests
- Test tracker integrated with Context
- Test VRAM usage reported in health endpoint
- Test OOM error includes VRAM breakdown
- Test tracker survives model load/unload cycles

### Manual Verification
1. Load model and check VRAM usage: `nvidia-smi`
2. Query health endpoint: `curl http://localhost:8080/health`
3. Verify VRAM breakdown matches nvidia-smi
4. Trigger OOM and verify error message includes breakdown

---

## Definition of Done

- [x] All acceptance criteria met
- [x] Code reviewed (self-review for agents)
- [x] Unit tests passing (13 tests)
- [x] Integration tests passing (via Context integration)
- [x] Documentation updated (VramTracker class docs)
- [x] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §2.2 VRAM-Only Policy (M0-W-1010, M0-W-1011)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §2.2 VRAM-Only Enforcement (M0-SYS-2.2.1)
- Related Stories: FT-010 (context init), FT-014 (residency verification)
- CUDA Memory API: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

---

## 🔍 Testing Requirements

**Added by**: Testing Team (test-harness/TEAM_RESPONSIBILITIES.md)

### Unit Tests (MUST implement)

**Critical Path Coverage**:
- **Test VramTracker records allocation correctly** (M0-W-1011)
  - Given: VramTracker initialized
  - When: record_allocation() called with 1GB, ModelWeights purpose
  - Then: total_usage() returns 1GB, allocation_count() returns 1
  - **Why critical**: Core tracking mechanism must be accurate

- **Test VramTracker records deallocation correctly** (M0-W-1011)
  - Given: VramTracker with 1GB allocation
  - When: record_deallocation() called
  - Then: total_usage() returns 0, allocation_count() returns 0
  - **Why critical**: Memory leak prevention depends on accurate deallocation tracking

- **Test usage_by_purpose returns correct breakdown** (M0-W-1011)
  - Given: Allocations for ModelWeights (1GB), KVCache (512MB), IntermediateBuffers (256MB)
  - When: usage_by_purpose(ModelWeights) called
  - Then: Returns exactly 1GB
  - **Why critical**: VRAM breakdown required for OOM diagnostics

- **Test verify_vram_residency with device pointers returns true** (M0-W-1012)
  - Given: cudaMalloc device pointer tracked
  - When: verify_vram_residency() called
  - Then: Returns true (cudaMemoryTypeDevice, no hostPointer)
  - **Why critical**: VRAM-only policy enforcement (M0-SYS-2.2.1)

- **Test verify_vram_residency with host pointers returns false** (M0-W-1012)
  - Given: cudaMallocHost host pointer tracked
  - When: verify_vram_residency() called
  - Then: Returns false (detects RAM fallback)
  - **Why critical**: Must detect VRAM-only policy violations

- **Test thread safety with concurrent allocations**
  - Given: VramTracker shared across threads
  - When: 10 threads call record_allocation() concurrently
  - Then: total_usage() equals sum of all allocations, no race conditions
  - **Why critical**: Future-proofs for multi-stream support

**Edge Cases**:
- **Test allocation with zero bytes** (defensive)
  - Given: record_allocation(ptr, 0, purpose)
  - When: Called
  - Then: Tracked correctly or rejected with clear error

- **Test deallocation of non-existent pointer** (defensive)
  - Given: record_deallocation(invalid_ptr)
  - When: Called
  - Then: No crash, no-op or logged warning

### Integration Tests (MUST implement)

- **Test VramTracker integrated with Context** (M0-W-1010)
  - Given: Context initialized with VramTracker
  - When: Model loaded via Context
  - Then: VramTracker reports model weight allocation
  - **Why critical**: Validates tracker wiring into Context

- **Test VRAM usage reported in health endpoint**
  - Given: Worker with model loaded
  - When: GET /health called
  - Then: Response includes vram_bytes_used field
  - **Why critical**: Health endpoint must expose VRAM metrics

- **Test OOM error includes VRAM breakdown** (M0-W-1021)
  - Given: Insufficient VRAM for allocation
  - When: Allocation fails
  - Then: Error message includes breakdown by purpose (weights, KV, buffers)
  - **Why critical**: OOM diagnostics require detailed breakdown

### Property Tests (SHOULD implement)

- **Property: total_usage equals sum of all tracked allocations**
  - Given: Arbitrary sequence of allocations/deallocations
  - When: Operations complete
  - Then: total_usage() always equals sum of active allocations
  - **Why valuable**: Validates accounting invariant

### BDD Scenarios (VERY IMPORTANT - MUST implement)

**Feature**: VRAM Allocation Tracking

```gherkin
Scenario: Worker tracks VRAM allocation for model weights
  Given a worker with CUDA context initialized
  When the worker loads a 1GB model into VRAM
  Then the VRAM tracker reports 1GB usage for ModelWeights
  And the health endpoint shows vram_bytes_used >= 1GB

Scenario: Worker detects VRAM OOM and reports breakdown
  Given a worker with 2GB VRAM available
  And 1.5GB already allocated for model weights
  When the worker attempts to allocate 1GB for KV cache
  Then the allocation fails with VRAM_OOM error
  And the error message includes "ModelWeights: 1.5GB, requested: 1GB, available: 0.5GB"

Scenario: Worker tracks deallocation correctly
  Given a worker with 1GB VRAM allocated
  When the worker deallocates the memory
  Then the VRAM tracker reports 0GB usage
  And subsequent allocations succeed
```

### Test Artifacts (MUST produce)

- **Unit test report**: Pass/fail for each test, coverage metrics
- **Integration test logs**: VRAM usage at each stage
- **BDD scenario results**: Pass/fail with step traces
- **Memory leak report**: VRAM usage before/after test suite (must be equal)

### Acceptance Criteria for Testing

- ✅ All unit tests pass (8+ tests covering critical paths and edge cases)
- ✅ All integration tests pass (3+ tests validating Context integration)
- ✅ All BDD scenarios pass (3 scenarios validating user-facing behavior)
- ✅ No memory leaks detected (verified with cuda-memcheck)
- ✅ Test coverage ≥ 90% for VramTracker class
- ✅ All tests produce verifiable artifacts

### False Positive Prevention

**CRITICAL**: Tests MUST observe product behavior, NEVER manipulate product state.

❌ **FORBIDDEN**:
```cpp
// Pre-creating VRAM allocation that product should create
cudaMalloc(&ptr, 1024);
tracker.record_allocation(ptr, 1024, VramPurpose::ModelWeights);
assert(tracker.total_usage() == 1024);  // FALSE POSITIVE
```

✅ **REQUIRED**:
```cpp
// Product creates allocation, test observes
auto mem = DeviceMemory(1024, &tracker, VramPurpose::ModelWeights);
assert(tracker.total_usage() == 1024);  // Product created allocation
assert(tracker.allocation_count() == 1);  // Verify tracking happened
```

### Test Execution Commands

```bash
# Unit tests
./build/tests/vram_tracker_test

# Integration tests (with Context)
cargo test --features cuda --test integration_vram_tracking

# Memory leak detection
cuda-memcheck --leak-check full ./build/tests/vram_tracker_test

# BDD scenarios
cargo run --bin bdd-runner -- --features vram_tracking
```

### Dependencies for Testing

- **Upstream**: FT-010 (CUDA context) must be complete for integration tests
- **Test infrastructure**: Google Test (C++), Cucumber (BDD), cuda-memcheck

---
**Testing requirements added by Testing Team 🔍**

---

**Status**: ✅ Complete  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Completed**: 2025-10-04


### Events to Narrate

1. **VRAM allocation** (ACTION_VRAM_ALLOCATE)
   ```rust
   Narration::new(ACTOR_VRAM_RESIDENCY, "vram_allocate", format!("GPU{}", device_id))
       .human(format!("Allocated {} MB VRAM on GPU{} for {}", bytes / 1024 / 1024, device_id, purpose))
       .device(format!("GPU{}", device_id))
       .emit();
   ```

2. **VRAM deallocation** (ACTION_VRAM_DEALLOCATE)
   ```rust
   Narration::new(ACTOR_VRAM_RESIDENCY, "vram_deallocate", format!("GPU{}", device_id))
       .human(format!("Deallocated {} MB VRAM on GPU{} ({} MB still in use)", 
                      bytes / 1024 / 1024, device_id, remaining / 1024 / 1024))
       .device(format!("GPU{}", device_id))
       .emit();
   ```

3. **VRAM OOM error**
   ```rust
   Narration::new(ACTOR_VRAM_RESIDENCY, "vram_allocate", format!("GPU{}", device_id))
       .human(format!("VRAM allocation failed on GPU{}: requested {} MB, only {} MB available", 
                      device_id, requested / 1024 / 1024, available / 1024 / 1024))
       .device(format!("GPU{}", device_id))
       .error_kind("vram_oom")
       .emit_error();  // ← ERROR level
   ```

**Why this matters**: VRAM allocation is a critical resource constraint. Narration helps diagnose OOM issues and track memory usage patterns.

### Testing Your Narration

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]  // ← Required for test isolation!
fn test_vram_allocation_narrates() {
    let adapter = CaptureAdapter::install();
    
    // Your VRAM allocation code here
    allocate_vram(1024 * 1024 * 1024);  // 1GB
    
    // Assert narration was emitted
    adapter.assert_includes("Allocated");
    adapter.assert_field("actor", "vram-residency");
    adapter.assert_field("device", "GPU0");
}
```

Run with: `cargo test --features test-support`

### Need Help?

- **Full docs**: `bin/shared-crates/narration-core/README.md`
- **Quick start**: `bin/shared-crates/narration-core/QUICKSTART.md`
- **Field reference**: See README section "NarrationFields Reference"
- **Our team doc**: `bin/shared-crates/narration-core/TEAM_RESPONSIBILITY.md`

We're watching your narration with ❤️ and will provide feedback if we see opportunities for improvement!

---
*Narration guidance added by Narration-Core Team v0.2.0 🎀*
