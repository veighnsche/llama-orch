/**
 * Device Memory RAII Wrapper Implementation
 * 
 * Implements RAII semantics for CUDA device memory with VramTracker integration.
 * 
 * Spec: M0-W-1220, CUDA-5222
 */

#include "device_memory.h"
#include <cuda_runtime.h>
#include <string>

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
    
    // Allocate device memory
    cudaError_t err = cudaMalloc(&ptr_, bytes);
    if (err != cudaSuccess) {
        throw CudaError::out_of_memory(
            std::string("Failed to allocate ") + 
            std::to_string(bytes) + " bytes: " + 
            cudaGetErrorString(err)
        );
    }
    
    // Zero-initialize if requested
    if (zero_init) {
        zero();
    }
    
    // Record allocation in tracker
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
    // Validate alignment is power of 2
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw CudaError::invalid_parameter("Alignment must be power of 2");
    }
    
    // Round up size to alignment
    size_t aligned_bytes = (bytes + alignment - 1) & ~(alignment - 1);
    
    // Allocate memory
    auto mem = std::make_unique<DeviceMemory>(aligned_bytes, tracker, purpose, zero_init);
    
    // Verify alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(mem->get());
    if ((addr & (alignment - 1)) != 0) {
        throw CudaError(
            CUDA_ERROR_UNKNOWN,
            "cudaMalloc did not return aligned pointer (expected " + 
            std::to_string(alignment) + "-byte alignment)"
        );
    }
    
    return mem;
}

DeviceMemory::~DeviceMemory() {
    if (ptr_) {
        // Record deallocation in tracker
        if (tracker_) {
            tracker_->record_deallocation(ptr_);
        }
        
        // Free device memory
        cudaFree(ptr_);
    }
}

DeviceMemory::DeviceMemory(DeviceMemory&& other) noexcept
    : ptr_(other.ptr_),
      size_(other.size_),
      tracker_(other.tracker_),
      purpose_(other.purpose_) {
    // Clear other
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
    // Record deallocation in tracker (ownership transferred)
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
            "Copy size (" + std::to_string(bytes) + 
            " bytes) exceeds allocated size (" + std::to_string(size_) + " bytes)"
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
            "Copy size (" + std::to_string(bytes) + 
            " bytes) exceeds allocated size (" + std::to_string(size_) + " bytes)"
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
    if (!ptr_) {
        return;
    }
    
    cudaError_t err = cudaMemset(ptr_, 0, size_);
    if (err != cudaSuccess) {
        throw CudaError::inference_failed(
            std::string("Failed to zero memory: ") + cudaGetErrorString(err)
        );
    }
}

} // namespace worker

// ---
// Built by Foundation-Alpha 🏗️
