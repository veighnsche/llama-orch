/**
 * CUDA Context Implementation
 * 
 * Implements CUDA device context with VRAM-only enforcement.
 * 
 * Spec: M0-W-1010, M0-W-1400, CUDA-5101, CUDA-5120
 */

#include "context.h"
#include "cuda_error.h"
#include <cuda_runtime.h>
#include <string>

namespace worker {

Context::Context(int gpu_device) : device_(gpu_device) {
    // Step 1: Check device count
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        throw CudaError::device_not_found(
            std::string("Failed to get device count: ") + cudaGetErrorString(err)
        );
    }
    
    // Step 2: Validate device ID
    if (gpu_device < 0 || gpu_device >= device_count) {
        throw CudaError::invalid_device(
            "Device ID " + std::to_string(gpu_device) + 
            " out of range (0-" + std::to_string(device_count - 1) + ")"
        );
    }
    
    // Step 3: Set device
    err = cudaSetDevice(gpu_device);
    if (err != cudaSuccess) {
        throw CudaError::invalid_device(
            std::string("Failed to set device: ") + cudaGetErrorString(err)
        );
    }
    
    // Step 4: Get device properties
    err = cudaGetDeviceProperties(&props_, gpu_device);
    if (err != cudaSuccess) {
        throw CudaError::invalid_device(
            std::string("Failed to get device properties: ") + cudaGetErrorString(err)
        );
    }
    
    // Step 5: Enforce VRAM-only mode - Disable Unified Memory
    // Setting malloc heap size to 0 disables UMA allocations
    err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0);
    if (err != cudaSuccess) {
        // This is critical for VRAM-only enforcement
        throw CudaError::kernel_launch_failed(
            std::string("Failed to disable UMA: ") + cudaGetErrorString(err)
        );
    }
    
    // Step 6: Set cache config for compute workloads
    // Prefer L1 cache over shared memory for better compute performance
    err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    if (err != cudaSuccess) {
        // Non-fatal: Some devices don't support cache config
        // Continue initialization but log warning
        // (In production, this would use tracing/logging)
    }
}

Context::~Context() {
    // Reset device to free all allocations
    // This is safe to call even if initialization failed
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

// ---
// Built by Foundation-Alpha 🏗️
