/**
 * Device Memory RAII Wrapper
 * 
 * Provides RAII wrapper for CUDA device memory to ensure automatic cleanup
 * and prevent memory leaks. Integrates with VramTracker for usage monitoring.
 * 
 * Spec: M0-W-1220, CUDA-5222
 */

#ifndef WORKER_DEVICE_MEMORY_H
#define WORKER_DEVICE_MEMORY_H

#include <cuda_runtime.h>
#include <memory>
#include <cstddef>
#include "cuda_error.h"
#include "vram_tracker.h"

namespace worker {

/**
 * RAII wrapper for CUDA device memory.
 * 
 * Provides automatic cleanup, move semantics, and integration with VramTracker.
 * 
 * Features:
 * - Automatic cleanup in destructor (no manual cudaFree needed)
 * - Non-copyable (unique ownership)
 * - Movable (transfer ownership)
 * - Exception-safe (destructor always called)
 * - VramTracker integration for usage monitoring
 * - Aligned allocation support (256-byte boundaries)
 * - Zero-initialization option for KV cache
 * 
 * Example:
 * ```cpp
 * VramTracker tracker;
 * 
 * {
 *     DeviceMemory mem(1024 * 1024, &tracker, VramPurpose::ModelWeights);
 *     // Use mem.get() for CUDA operations
 * }  // Automatic cleanup here
 * 
 * // VRAM freed, tracker updated
 * ```
 */
class DeviceMemory {
public:
    /**
     * Allocate device memory.
     * 
     * @param bytes Size in bytes (must be > 0)
     * @param tracker Optional VRAM tracker for usage tracking
     * @param purpose Purpose of allocation (for tracking)
     * @param zero_init If true, initialize memory to zero
     * @throws CudaError::InvalidParameter if bytes == 0
     * @throws CudaError::OutOfMemory if allocation fails
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
     * Allocates memory aligned to specified boundary for optimal GPU performance.
     * 
     * @param bytes Size in bytes
     * @param alignment Alignment in bytes (must be power of 2)
     * @param tracker Optional VRAM tracker
     * @param purpose Purpose of allocation
     * @param zero_init If true, initialize memory to zero
     * @return Unique pointer to DeviceMemory
     * @throws CudaError::InvalidParameter if alignment not power of 2
     * @throws CudaError::OutOfMemory if allocation fails
     * @throws CudaError if cudaMalloc doesn't return aligned pointer
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
     * 
     * Automatically called when object goes out of scope.
     * Updates VramTracker if present.
     */
    ~DeviceMemory();
    
    // Non-copyable (unique ownership)
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    
    // Movable (transfer ownership)
    DeviceMemory(DeviceMemory&& other) noexcept;
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;
    
    /**
     * Get raw device pointer.
     * 
     * @return Device pointer, or nullptr if not allocated
     */
    void* get() const { return ptr_; }
    
    /**
     * Get typed device pointer.
     * 
     * @return Typed device pointer
     */
    template<typename T>
    T* get_as() const { return static_cast<T*>(ptr_); }
    
    /**
     * Get size in bytes.
     * 
     * @return Allocated size in bytes
     */
    size_t size() const { return size_; }
    
    /**
     * Check if memory is allocated.
     * 
     * @return true if memory allocated, false otherwise
     */
    bool is_allocated() const { return ptr_ != nullptr; }
    
    /**
     * Release ownership (caller responsible for freeing).
     * 
     * Transfers ownership to caller. Caller must call cudaFree manually.
     * Updates VramTracker to reflect deallocation.
     * 
     * @return Device pointer (caller must free)
     */
    void* release();
    
    /**
     * Copy data from host to device.
     * 
     * @param host_ptr Host pointer to copy from
     * @param bytes Number of bytes to copy (must be <= size())
     * @throws CudaError::InvalidParameter if bytes > size()
     * @throws CudaError if cudaMemcpy fails
     */
    void copy_from_host(const void* host_ptr, size_t bytes);
    
    /**
     * Copy data from device to host.
     * 
     * @param host_ptr Host pointer to copy to
     * @param bytes Number of bytes to copy (must be <= size())
     * @throws CudaError::InvalidParameter if bytes > size()
     * @throws CudaError if cudaMemcpy fails
     */
    void copy_to_host(void* host_ptr, size_t bytes) const;
    
    /**
     * Zero-initialize memory.
     * 
     * Sets all bytes to zero using cudaMemset.
     * 
     * @throws CudaError if cudaMemset fails
     */
    void zero();

private:
    void* ptr_ = nullptr;                    ///< Device pointer
    size_t size_ = 0;                        ///< Allocated size in bytes
    VramTracker* tracker_ = nullptr;         ///< Optional VRAM tracker
    VramPurpose purpose_ = VramPurpose::Unknown;  ///< Purpose for tracking
};

} // namespace worker

#endif // WORKER_DEVICE_MEMORY_H

// ---
// Built by Foundation-Alpha 🏗️
