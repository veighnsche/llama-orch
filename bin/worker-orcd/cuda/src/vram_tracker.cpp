/**
 * VRAM Allocation Tracker Implementation
 * 
 * Implements VRAM tracking and residency verification.
 * 
 * Spec: M0-W-1010, M0-W-1011, M0-SYS-2.2.1
 */

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
            // Failed to query pointer attributes
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
    oss << "  Total: " << std::fixed << std::setprecision(2) 
        << (total / 1024.0 / 1024.0) << " MB\n";
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
            << std::fixed << std::setprecision(2)
            << (bytes / 1024.0 / 1024.0) << " MB\n";
    }
    
    return oss.str();
}

} // namespace worker

// ---
// Built by Foundation-Alpha 🏗️
