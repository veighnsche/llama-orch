/**
 * CUDA Context Management
 * 
 * Provides RAII wrapper for CUDA device context with VRAM-only enforcement.
 * 
 * Spec: M0-W-1010, M0-W-1400, CUDA-5101, CUDA-5120
 */

#ifndef WORKER_CONTEXT_H
#define WORKER_CONTEXT_H

#include <cuda_runtime.h>
#include <cstddef>

namespace worker {

// Forward declaration
class CudaError;

/**
 * CUDA device context with VRAM-only enforcement.
 * 
 * This class manages CUDA device initialization and configuration.
 * Key features:
 * - Disables Unified Memory (VRAM-only mode)
 * - Sets cache config for compute workloads
 * - Provides device property queries
 * - Automatic cleanup via cudaDeviceReset()
 * 
 * Thread safety: Context is NOT thread-safe. Each thread should
 * have its own context or use external synchronization.
 * 
 * Example:
 * ```cpp
 * auto ctx = std::make_unique<Context>(0);  // GPU 0
 * std::cout << "Device: " << ctx->device_name() << std::endl;
 * std::cout << "VRAM: " << ctx->total_vram() / (1024*1024) << " MB" << std::endl;
 * ```
 */
class Context {
public:
    /**
     * Initialize CUDA context for specified device.
     * 
     * This constructor:
     * 1. Validates device ID against cudaGetDeviceCount()
     * 2. Sets device via cudaSetDevice()
     * 3. Retrieves device properties
     * 4. Disables Unified Memory (VRAM-only)
     * 5. Sets cache config to prefer L1
     * 
     * @param gpu_device Device ID (0, 1, 2, ...)
     * @throws CudaError if device invalid or initialization fails
     */
    explicit Context(int gpu_device);
    
    /**
     * Cleanup CUDA context.
     * 
     * Calls cudaDeviceReset() to free all device allocations.
     */
    ~Context();
    
    // Non-copyable, non-movable (owns CUDA device state)
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;
    
    // ========================================================================
    // Device Queries
    // ========================================================================
    
    /**
     * Get device ID.
     * 
     * @return Device ID (0, 1, 2, ...)
     */
    int device() const { return device_; }
    
    /**
     * Get device properties.
     * 
     * @return CUDA device properties structure
     */
    const cudaDeviceProp& properties() const { return props_; }
    
    /**
     * Get device name.
     * 
     * @return Device name string (e.g., "NVIDIA GeForce RTX 4090")
     */
    const char* device_name() const { return props_.name; }
    
    /**
     * Get compute capability.
     * 
     * @return Compute capability as integer (e.g., 86 for SM_86)
     */
    int compute_capability() const {
        return props_.major * 10 + props_.minor;
    }
    
    /**
     * Get total VRAM in bytes.
     * 
     * @return Total device memory in bytes
     */
    size_t total_vram() const { return props_.totalGlobalMem; }
    
    /**
     * Get free VRAM in bytes.
     * 
     * Queries current free memory via cudaMemGetInfo().
     * 
     * @return Free device memory in bytes (0 if query fails)
     */
    size_t free_vram() const;
    
    // ========================================================================
    // Static Helpers
    // ========================================================================
    
    /**
     * Get number of available CUDA devices.
     * 
     * @return Device count (0 if no devices or query fails)
     */
    static int device_count();

private:
    int device_;              ///< Device ID
    cudaDeviceProp props_;    ///< Cached device properties
};

} // namespace worker

#endif // WORKER_CONTEXT_H

// ---
// Built by Foundation-Alpha 🏗️
