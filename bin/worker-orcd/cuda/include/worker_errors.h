/**
 * Worker CUDA - Error Codes
 * 
 * Defines all error codes used across the FFI boundary.
 * Error codes are positive integers (0 = success).
 */

#ifndef WORKER_ERRORS_H
#define WORKER_ERRORS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Error codes for FFI boundary.
 * All functions return error codes via out-parameters.
 */
typedef enum {
    CUDA_SUCCESS = 0,                      // Operation succeeded
    CUDA_ERROR_INVALID_DEVICE = 1,         // Invalid GPU device ID
    CUDA_ERROR_OUT_OF_MEMORY = 2,          // Insufficient VRAM
    CUDA_ERROR_MODEL_LOAD_FAILED = 3,      // Model loading failed
    CUDA_ERROR_INFERENCE_FAILED = 4,       // Inference execution failed
    CUDA_ERROR_INVALID_PARAMETER = 5,      // Invalid function parameter
    CUDA_ERROR_KERNEL_LAUNCH_FAILED = 6,   // CUDA kernel launch failed
    CUDA_ERROR_VRAM_RESIDENCY_FAILED = 7,  // VRAM residency check failed
    CUDA_ERROR_DEVICE_NOT_FOUND = 8,       // No CUDA devices found
    CUDA_ERROR_UNKNOWN = 99,               // Unknown error
} CudaErrorCode;

/**
 * Get human-readable error message for error code.
 * 
 * @param error_code Error code from any FFI function
 * @return Null-terminated error message string (static storage, never NULL)
 * 
 * Thread safety: Safe (returns static strings)
 * Memory ownership: Caller must NOT free the returned string
 */
const char* cuda_error_message(int error_code);

#ifdef __cplusplus
}
#endif

#endif // WORKER_ERRORS_H

// ---
// Built by Foundation-Alpha 🏗️
