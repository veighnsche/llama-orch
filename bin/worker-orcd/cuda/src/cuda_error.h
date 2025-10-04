/**
 * CUDA Error Exception Class
 * 
 * C++ exception class for CUDA errors. Used internally in C++ layer
 * and converted to error codes at FFI boundary.
 * 
 * Spec: M0-W-1501, CUDA-5040, CUDA-5041
 */

#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

#include <exception>
#include <string>
#include "../include/worker_errors.h"

namespace worker {

/**
 * CUDA error exception class.
 * 
 * Used internally in C++ layer for error handling.
 * Converted to error codes at FFI boundary (never crosses to Rust).
 * 
 * Thread safety: Exception objects are thread-safe (immutable after construction)
 */
class CudaError : public std::exception {
public:
    /**
     * Construct error with code and message.
     * 
     * @param code Error code from CudaErrorCode enum
     * @param message Human-readable error description
     */
    CudaError(int code, const std::string& message)
        : code_(code), message_(message) {}
    
    /**
     * Construct error with code and C string message.
     * 
     * @param code Error code from CudaErrorCode enum
     * @param message Human-readable error description (null-terminated)
     */
    CudaError(int code, const char* message)
        : code_(code), message_(message) {}
    
    /**
     * Get error code.
     * 
     * @return Error code from CudaErrorCode enum
     */
    int code() const noexcept { return code_; }
    
    /**
     * Get error message (std::exception interface).
     * 
     * @return Null-terminated error message
     */
    const char* what() const noexcept override { return message_.c_str(); }
    
    // ========================================================================
    // Factory Methods (Convenience)
    // ========================================================================
    
    /**
     * Create invalid device error.
     * 
     * @param details Additional context about the error
     * @return CudaError with CUDA_ERROR_INVALID_DEVICE code
     */
    static CudaError invalid_device(const std::string& details) {
        return CudaError(CUDA_ERROR_INVALID_DEVICE, "Invalid device: " + details);
    }
    
    /**
     * Create out of memory error.
     * 
     * @param details Additional context (e.g., "requested 16GB, available 8GB")
     * @return CudaError with CUDA_ERROR_OUT_OF_MEMORY code
     */
    static CudaError out_of_memory(const std::string& details) {
        return CudaError(CUDA_ERROR_OUT_OF_MEMORY, "Out of memory: " + details);
    }
    
    /**
     * Create model load failed error.
     * 
     * @param details Additional context (e.g., "file not found: /path/to/model.gguf")
     * @return CudaError with CUDA_ERROR_MODEL_LOAD_FAILED code
     */
    static CudaError model_load_failed(const std::string& details) {
        return CudaError(CUDA_ERROR_MODEL_LOAD_FAILED, "Model load failed: " + details);
    }
    
    /**
     * Create inference failed error.
     * 
     * @param details Additional context (e.g., "kernel launch failed at layer 5")
     * @return CudaError with CUDA_ERROR_INFERENCE_FAILED code
     */
    static CudaError inference_failed(const std::string& details) {
        return CudaError(CUDA_ERROR_INFERENCE_FAILED, "Inference failed: " + details);
    }
    
    /**
     * Create invalid parameter error.
     * 
     * @param details Additional context (e.g., "temperature must be 0.0-2.0, got 3.5")
     * @return CudaError with CUDA_ERROR_INVALID_PARAMETER code
     */
    static CudaError invalid_parameter(const std::string& details) {
        return CudaError(CUDA_ERROR_INVALID_PARAMETER, "Invalid parameter: " + details);
    }
    
    /**
     * Create kernel launch failed error.
     * 
     * @param details Additional context (e.g., "attention kernel failed")
     * @return CudaError with CUDA_ERROR_KERNEL_LAUNCH_FAILED code
     */
    static CudaError kernel_launch_failed(const std::string& details) {
        return CudaError(CUDA_ERROR_KERNEL_LAUNCH_FAILED, "Kernel launch failed: " + details);
    }
    
    /**
     * Create VRAM residency failed error.
     * 
     * @param details Additional context (e.g., "pointer not in device memory")
     * @return CudaError with CUDA_ERROR_VRAM_RESIDENCY_FAILED code
     */
    static CudaError vram_residency_failed(const std::string& details) {
        return CudaError(CUDA_ERROR_VRAM_RESIDENCY_FAILED, "VRAM residency failed: " + details);
    }
    
    /**
     * Create device not found error.
     * 
     * @param details Additional context (e.g., "no CUDA devices detected")
     * @return CudaError with CUDA_ERROR_DEVICE_NOT_FOUND code
     */
    static CudaError device_not_found(const std::string& details) {
        return CudaError(CUDA_ERROR_DEVICE_NOT_FOUND, "Device not found: " + details);
    }

private:
    int code_;
    std::string message_;
};

} // namespace worker

#endif // CUDA_ERROR_H

// ---
// Built by Foundation-Alpha 🏗️
