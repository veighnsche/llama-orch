/**
 * Error Message Implementation
 * 
 * Implements cuda_error_message() function that returns human-readable
 * error messages for error codes.
 * 
 * Spec: M0-W-1501, CUDA-5040
 */

#include "../include/worker_errors.h"

extern "C" const char* cuda_error_message(int error_code) {
    switch (error_code) {
        case CUDA_SUCCESS:
            return "Operation completed successfully";
            
        case CUDA_ERROR_INVALID_DEVICE:
            return "Invalid CUDA device ID";
            
        case CUDA_ERROR_OUT_OF_MEMORY:
            return "Out of GPU memory (VRAM)";
            
        case CUDA_ERROR_MODEL_LOAD_FAILED:
            return "Failed to load model from GGUF file";
            
        case CUDA_ERROR_INFERENCE_FAILED:
            return "Inference execution failed";
            
        case CUDA_ERROR_INVALID_PARAMETER:
            return "Invalid parameter provided";
            
        case CUDA_ERROR_KERNEL_LAUNCH_FAILED:
            return "CUDA kernel launch failed";
            
        case CUDA_ERROR_VRAM_RESIDENCY_FAILED:
            return "VRAM residency check failed (RAM fallback detected)";
            
        case CUDA_ERROR_DEVICE_NOT_FOUND:
            return "No CUDA devices found";
            
        case CUDA_ERROR_UNKNOWN:
            return "Unknown error occurred";
            
        default:
            return "Unrecognized error code";
    }
}

// ---
// Built by Foundation-Alpha 🏗️
