/**
 * CUDA Health Monitoring Implementation
 * 
 * Implements VRAM residency verification and health checks.
 * 
 * Spec: M0-W-1012, CUDA-5421
 */

#include "../include/health.h"
#include "cuda_error.h"
#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>

namespace worker {

bool Health::check_vram_residency(const VramTracker& tracker) {
    // Delegate to VramTracker's built-in verification
    // This checks all tracked allocations via cudaPointerGetAttributes
    return tracker.verify_vram_residency();
}

bool Health::check_pointer_residency(const void* ptr) {
    if (ptr == nullptr) {
        return false;
    }
    
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    
    if (err != cudaSuccess) {
        // Failed to query pointer attributes - clear error and return false
        cudaGetLastError();
        return false;
    }
    
    // Verify pointer is device memory (not managed/host)
    if (attrs.type != cudaMemoryTypeDevice) {
        return false;
    }
    
    // Verify no host pointer exists (no UMA)
    if (attrs.hostPointer != nullptr) {
        return false;
    }
    
    return true;
}

uint64_t Health::get_process_vram_usage() {
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    
    if (err != cudaSuccess) {
        return 0;
    }
    
    // Return used VRAM (total - free)
    return total_bytes - free_bytes;
}

std::string Health::residency_report(const VramTracker& tracker) {
    std::ostringstream oss;
    
    bool resident = check_vram_residency(tracker);
    uint64_t process_usage = get_process_vram_usage();
    uint64_t tracked_usage = tracker.total_usage();
    size_t allocation_count = tracker.allocation_count();
    
    oss << "VRAM Residency Report:\n";
    oss << "  Status: " << (resident ? "RESIDENT" : "VIOLATION") << "\n";
    oss << "  Process VRAM Usage: " << std::fixed << std::setprecision(2)
        << (process_usage / 1024.0 / 1024.0) << " MB\n";
    oss << "  Tracked VRAM Usage: " << std::fixed << std::setprecision(2)
        << (tracked_usage / 1024.0 / 1024.0) << " MB\n";
    oss << "  Allocations: " << allocation_count << "\n";
    
    if (!resident) {
        oss << "  WARNING: RAM fallback or UMA detected!\n";
    }
    
    return oss.str();
}

} // namespace worker

// ---
// Built by Foundation-Alpha 🏗️
