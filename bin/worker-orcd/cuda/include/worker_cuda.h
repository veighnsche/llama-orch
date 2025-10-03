/**
 * Worker CUDA - C API for Rust FFI
 * 
 * This header defines the C API exposed to the Rust layer.
 * All functions use C linkage and error codes (no exceptions).
 */

#ifndef WORKER_CUDA_H
#define WORKER_CUDA_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Opaque Handle Types
// ============================================================================

typedef struct CudaContext CudaContext;
typedef struct CudaModel CudaModel;
typedef struct InferenceResult InferenceResult;

// ============================================================================
// Error Codes
// ============================================================================

enum CudaErrorCode {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_DEVICE_NOT_FOUND = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_INVALID_DEVICE = 3,
    CUDA_ERROR_MODEL_LOAD_FAILED = 4,
    CUDA_ERROR_INFERENCE_FAILED = 5,
    CUDA_ERROR_VRAM_RESIDENCY_FAILED = 6,
    CUDA_ERROR_KERNEL_LAUNCH_FAILED = 7,
    CUDA_ERROR_INVALID_PARAMETER = 8,
    CUDA_ERROR_UNKNOWN = 99,
};

// Get human-readable error message
const char* cuda_error_message(int error_code);

// ============================================================================
// Context Management
// ============================================================================

/**
 * Initialize CUDA context on specified device.
 * 
 * @param gpu_device CUDA device ID (0, 1, ...)
 * @param error_code Output: error code (0 = success)
 * @return Opaque context handle, or NULL on error
 */
CudaContext* cuda_init(int gpu_device, int* error_code);

/**
 * Destroy CUDA context and free resources.
 * 
 * @param ctx Context handle (may be NULL)
 */
void cuda_destroy(CudaContext* ctx);

/**
 * Get number of available CUDA devices.
 * 
 * @return Device count, or 0 if no devices
 */
int cuda_get_device_count(void);

// ============================================================================
// Model Loading
// ============================================================================

/**
 * Load model from disk/RAM to VRAM.
 * 
 * @param ctx Context handle
 * @param model_path Path to model file (GGUF format)
 * @param vram_bytes_used Output: actual VRAM bytes allocated
 * @param error_code Output: error code (0 = success)
 * @return Opaque model handle, or NULL on error
 */
CudaModel* cuda_load_model(
    CudaContext* ctx,
    const char* model_path,
    uint64_t* vram_bytes_used,
    int* error_code
);

/**
 * Unload model and free VRAM.
 * 
 * @param model Model handle (may be NULL)
 */
void cuda_unload_model(CudaModel* model);

/**
 * Get VRAM usage for model.
 * 
 * @param model Model handle
 * @return VRAM bytes used, or 0 if model is NULL
 */
uint64_t cuda_model_get_vram_usage(CudaModel* model);

// ============================================================================
// Inference Execution
// ============================================================================

/**
 * Start inference session.
 * 
 * @param model Model handle
 * @param prompt Input prompt (UTF-8 string)
 * @param max_tokens Maximum tokens to generate
 * @param temperature Sampling temperature (0.0 to 2.0)
 * @param seed RNG seed for deterministic sampling
 * @param error_code Output: error code (0 = success)
 * @return Opaque inference handle, or NULL on error
 */
InferenceResult* cuda_inference_start(
    CudaModel* model,
    const char* prompt,
    int max_tokens,
    float temperature,
    uint64_t seed,
    int* error_code
);

/**
 * Generate next token (blocking).
 * 
 * @param result Inference handle
 * @param token_out Output buffer for token (UTF-8 string)
 * @param token_buffer_size Size of token_out buffer
 * @param token_index Output: token position (0, 1, 2, ...)
 * @param error_code Output: error code (0 = success)
 * @return true if token generated, false if inference complete
 */
bool cuda_inference_next_token(
    InferenceResult* result,
    char* token_out,
    int token_buffer_size,
    int* token_index,
    int* error_code
);

/**
 * Free inference resources (KV cache, etc.).
 * 
 * @param result Inference handle (may be NULL)
 */
void cuda_inference_free(InferenceResult* result);

// ============================================================================
// Health Monitoring
// ============================================================================

/**
 * Check if model is still in VRAM (no RAM fallback).
 * 
 * @param model Model handle
 * @param error_code Output: error code (0 = success)
 * @return true if model in VRAM, false if corrupted/swapped
 */
bool cuda_check_vram_residency(CudaModel* model, int* error_code);

/**
 * Get current VRAM usage for model.
 * 
 * @param model Model handle
 * @return VRAM bytes used, or 0 if model is NULL
 */
uint64_t cuda_get_vram_usage(CudaModel* model);

/**
 * Get total VRAM allocated by process.
 * 
 * @param ctx Context handle
 * @return VRAM bytes used by process, or 0 if ctx is NULL
 */
uint64_t cuda_get_process_vram_usage(CudaContext* ctx);

/**
 * Check GPU device health.
 * 
 * @param ctx Context handle
 * @param error_code Output: error code (0 = success)
 * @return true if device healthy, false if errors detected
 */
bool cuda_check_device_health(CudaContext* ctx, int* error_code);

#ifdef __cplusplus
}
#endif

#endif // WORKER_CUDA_H
