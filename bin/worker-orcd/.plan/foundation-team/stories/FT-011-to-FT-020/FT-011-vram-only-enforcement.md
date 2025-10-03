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

- [ ] VRAM allocation tracker records all cudaMalloc calls with size and purpose
- [ ] Tracker reports total VRAM usage (model weights + KV cache + intermediate buffers)
- [ ] Validation function verifies no host pointers exist for model data
- [ ] Unit tests validate tracking accuracy
- [ ] Integration tests validate VRAM-only residency
- [ ] Error reporting includes VRAM usage breakdown on OOM
- [ ] Tracker thread-safe (if needed for future multi-stream support)
- [ ] Health endpoint exposes VRAM usage metrics

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

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (8+ tests)
- [ ] Integration tests passing (4+ tests)
- [ ] Documentation updated (VramTracker class docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §2.2 VRAM-Only Policy (M0-W-1010, M0-W-1011)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §2.2 VRAM-Only Enforcement (M0-SYS-2.2.1)
- Related Stories: FT-010 (context init), FT-014 (residency verification)
- CUDA Memory API: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
