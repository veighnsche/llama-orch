/**
 * Worker CUDA - FFI Interface (LOCKED)
 * 
 * This header defines the complete C API for the Rust-CUDA FFI boundary.
 * 
 * CRITICAL: This interface is LOCKED after review. No changes without team coordination.
 * 
 * Design Principles:
 * - All functions use C linkage (extern "C")
 * - All functions use out-parameters for error codes (no exceptions across FFI)
 * - All handle types are opaque (implementation hidden from Rust)
 * - All string parameters are null-terminated UTF-8
 * - All pointers can be NULL (functions must check)
 * - Error codes are positive integers (0 = success)
 * - Thread safety: Each context is single-threaded (no concurrent calls)
 * - Memory ownership: Rust owns nothing allocated by C++ (must call free functions)
 * 
 * Spec References: M0-W-1052, CUDA-4030, CUDA-4011
 */

#ifndef WORKER_FFI_H
#define WORKER_FFI_H

#include <stdint.h>
#include <stdbool.h>

#include "worker_types.h"
#include "worker_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Context Management
// ============================================================================

/**
 * Initialize CUDA context for specified GPU device.
 * 
 * Performs:
 * - Device validation and selection
 * - CUDA context initialization
 * - VRAM-only enforcement (disable UMA)
 * - Cache configuration for compute
 * 
 * @param gpu_device GPU device ID (0, 1, 2, ...)
 * @param error_code Output parameter for error code (must not be NULL)
 * @return Opaque context handle, or NULL on failure
 * 
 * Error codes:
 *   CUDA_ERROR_INVALID_DEVICE - Invalid device ID or device not found
 *   CUDA_ERROR_INVALID_PARAMETER - error_code is NULL
 *   CUDA_ERROR_UNKNOWN - CUDA initialization failed
 * 
 * Thread safety: Safe to call from multiple threads (creates independent contexts)
 * Memory ownership: Caller must call cuda_destroy() to free
 * 
 * Spec: M0-W-1110 (Startup Steps), M0-W-1010 (CUDA Context Configuration)
 */
CudaContext* cuda_init(int gpu_device, int* error_code);

/**
 * Destroy CUDA context and free all resources.
 * 
 * Performs:
 * - Free all CUDA resources
 * - Destroy CUDA context
 * - Clean up internal state
 * 
 * @param ctx Context handle from cuda_init (may be NULL)
 * 
 * Thread safety: NOT thread-safe. Caller must ensure no concurrent access.
 * Memory ownership: After this call, ctx is invalid and must not be used.
 * 
 * Note: Safe to call with NULL (no-op).
 * 
 * Spec: M0-W-1111 (Startup Failure Handling)
 */
void cuda_destroy(CudaContext* ctx);

/**
 * Get number of available CUDA devices.
 * 
 * @return Number of CUDA devices, or 0 if no devices or error
 * 
 * Thread safety: Safe to call from multiple threads
 * 
 * Spec: M0-W-1100 (CLI validation)
 */
int cuda_get_device_count(void);

// ============================================================================
// Model Loading
// ============================================================================

/**
 * Load model from GGUF file to VRAM.
 * 
 * Performs:
 * - GGUF header validation (magic bytes, version)
 * - Architecture detection (Llama vs GPT)
 * - VRAM allocation for model weights
 * - Memory-mapped I/O for efficient loading
 * - Chunked H2D transfer (1MB chunks)
 * - VRAM residency verification
 * 
 * @param ctx Context handle (must not be NULL)
 * @param model_path Absolute path to .gguf file (null-terminated UTF-8, must not be NULL)
 * @param vram_bytes_used Output parameter for VRAM bytes allocated (must not be NULL)
 * @param error_code Output parameter for error code (must not be NULL)
 * @return Opaque model handle, or NULL on failure
 * 
 * Error codes:
 *   CUDA_ERROR_MODEL_LOAD_FAILED - File not found, invalid format, or parsing failed
 *   CUDA_ERROR_OUT_OF_MEMORY - Insufficient VRAM for model weights
 *   CUDA_ERROR_INVALID_PARAMETER - ctx, model_path, vram_bytes_used, or error_code is NULL
 *   CUDA_ERROR_UNKNOWN - Unexpected error during loading
 * 
 * Thread safety: NOT thread-safe. Single-threaded access to ctx required.
 * Memory ownership: Caller must call cuda_unload_model() to free VRAM.
 * 
 * Spec: M0-W-1210 (Pre-Load Validation), M0-W-1211 (GGUF Parsing),
 *       M0-W-1220 (VRAM Allocation), M0-W-1221 (Memory-Mapped I/O),
 *       M0-W-1222 (Chunked H2D Transfer)
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
 * Performs:
 * - Free VRAM allocated for model weights
 * - Clean up internal state
 * 
 * @param model Model handle from cuda_load_model (may be NULL)
 * 
 * Thread safety: NOT thread-safe. Caller must ensure no concurrent access.
 * Memory ownership: After this call, model is invalid and must not be used.
 * 
 * Note: Safe to call with NULL (no-op).
 * 
 * Spec: M0-W-1002 (Model Immutability)
 */
void cuda_unload_model(CudaModel* model);

/**
 * Get current VRAM usage for model.
 * 
 * @param model Model handle (may be NULL)
 * @return VRAM bytes used, or 0 if model is NULL
 * 
 * Thread safety: Safe to call concurrently (read-only)
 * 
 * Spec: M0-W-1011 (VRAM Allocation Tracking)
 */
uint64_t cuda_model_get_vram_usage(CudaModel* model);

// ============================================================================
// Inference Execution
// ============================================================================

/**
 * Start inference job with given prompt and parameters.
 * 
 * Performs:
 * - Tokenize prompt
 * - Allocate KV cache in VRAM
 * - Initialize RNG with seed
 * - Prepare inference state
 * 
 * @param model Model handle (must not be NULL)
 * @param prompt Input prompt (null-terminated UTF-8, must not be NULL)
 * @param max_tokens Maximum tokens to generate (1-2048)
 * @param temperature Sampling temperature (0.0-2.0, 0.0 = greedy for testing)
 * @param seed Random seed for reproducibility
 * @param error_code Output parameter for error code (must not be NULL)
 * @return Opaque inference result handle, or NULL on failure
 * 
 * Error codes:
 *   CUDA_ERROR_OUT_OF_MEMORY - Insufficient VRAM for KV cache
 *   CUDA_ERROR_INVALID_PARAMETER - Invalid parameters (NULL pointers, out of range values)
 *   CUDA_ERROR_INFERENCE_FAILED - Inference initialization failed
 *   CUDA_ERROR_UNKNOWN - Unexpected error
 * 
 * Thread safety: NOT thread-safe. Single-threaded access to model required.
 * Memory ownership: Caller must call cuda_inference_free() to free resources.
 * 
 * Note: M0 supports single-threaded execution only (batch=1).
 * 
 * Spec: M0-W-1300 (POST /execute), M0-W-1301 (Single-Threaded Execution),
 *       M0-W-1030 (Seeded RNG), M0-W-1032 (Temperature Scaling)
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
 * Generate next token in inference sequence.
 * 
 * Performs:
 * - Execute forward pass (attention + FFN)
 * - Sample next token with temperature scaling
 * - Decode token to UTF-8 text
 * - Update KV cache
 * 
 * @param result Inference result handle (must not be NULL)
 * @param token_out Output buffer for token text (UTF-8, must not be NULL)
 * @param token_buffer_size Size of token_out buffer (recommended: 256 bytes)
 * @param token_index Output parameter for token index (0, 1, 2, ..., may be NULL)
 * @param error_code Output parameter for error code (must not be NULL)
 * @return true if token generated, false if sequence complete or error
 * 
 * Error codes:
 *   CUDA_ERROR_INFERENCE_FAILED - Kernel execution failed
 *   CUDA_ERROR_OUT_OF_MEMORY - VRAM exhausted during generation
 *   CUDA_ERROR_INVALID_PARAMETER - Invalid parameters (NULL pointers, buffer too small)
 *   CUDA_ERROR_KERNEL_LAUNCH_FAILED - CUDA kernel launch failed
 * 
 * Thread safety: NOT thread-safe. Single-threaded access to result required.
 * 
 * UTF-8 Safety:
 * - Function buffers partial multibyte sequences internally
 * - Never emits invalid UTF-8
 * - Handles token boundaries that split UTF-8 codepoints
 * 
 * Return value:
 * - true: Token generated successfully, check error_code for warnings
 * - false: Sequence complete (EOS token) or error occurred
 * 
 * Spec: M0-W-1310 (SSE Streaming), M0-W-1311 (Event Ordering),
 *       M0-W-1031 (Reproducible CUDA Kernels)
 */
bool cuda_inference_next_token(
    InferenceResult* result,
    char* token_out,
    int token_buffer_size,
    int* token_index,
    int* error_code
);

/**
 * Free inference result and associated resources.
 * 
 * Performs:
 * - Free KV cache VRAM
 * - Clean up inference state
 * 
 * @param result Inference result handle (may be NULL)
 * 
 * Thread safety: NOT thread-safe. Caller must ensure no concurrent access.
 * Memory ownership: After this call, result is invalid and must not be used.
 * 
 * Note: Safe to call with NULL (no-op).
 * 
 * Spec: M0-W-1021 (VRAM OOM During Inference)
 */
void cuda_inference_free(InferenceResult* result);

// ============================================================================
// Health & Monitoring
// ============================================================================

/**
 * Check VRAM residency for model weights.
 * 
 * Performs:
 * - Verify pointer type is cudaMemoryTypeDevice
 * - Verify no host pointer exists (no UMA)
 * - Check for RAM fallback
 * 
 * @param model Model handle (must not be NULL)
 * @param error_code Output parameter for error code (must not be NULL)
 * @return true if all weights resident in VRAM, false otherwise
 * 
 * Error codes:
 *   CUDA_ERROR_VRAM_RESIDENCY_FAILED - RAM fallback detected or pointer invalid
 *   CUDA_ERROR_INVALID_PARAMETER - model or error_code is NULL
 *   CUDA_ERROR_UNKNOWN - CUDA API call failed
 * 
 * Thread safety: Safe to call concurrently (read-only check)
 * 
 * Spec: M0-W-1012 (VRAM Residency Verification)
 */
bool cuda_check_vram_residency(CudaModel* model, int* error_code);

/**
 * Get current VRAM usage for model.
 * 
 * @param model Model handle (may be NULL)
 * @return VRAM bytes used, or 0 if model is NULL
 * 
 * Thread safety: Safe to call concurrently (read-only)
 * 
 * Spec: M0-W-1011 (VRAM Allocation Tracking)
 */
uint64_t cuda_get_vram_usage(CudaModel* model);

/**
 * Get process-wide VRAM usage.
 * 
 * Uses cudaMemGetInfo to query total VRAM allocated by this process.
 * 
 * @param ctx Context handle (must not be NULL)
 * @return Total VRAM bytes used by this process, or 0 if ctx is NULL
 * 
 * Thread safety: Safe to call concurrently (read-only)
 * 
 * Spec: M0-W-1012 (VRAM Residency Verification - Process VRAM Usage Query)
 */
uint64_t cuda_get_process_vram_usage(CudaContext* ctx);

/**
 * Check CUDA device health.
 * 
 * Performs:
 * - Query device status
 * - Check for CUDA errors
 * - Verify device is responsive
 * 
 * @param ctx Context handle (must not be NULL)
 * @param error_code Output parameter for error code (must not be NULL)
 * @return true if device is healthy, false otherwise
 * 
 * Error codes:
 *   CUDA_ERROR_INVALID_DEVICE - Device is not responsive or has errors
 *   CUDA_ERROR_INVALID_PARAMETER - ctx or error_code is NULL
 *   CUDA_ERROR_UNKNOWN - CUDA API call failed
 * 
 * Thread safety: Safe to call concurrently (read-only check)
 * 
 * Spec: M0-W-1012 (VRAM Residency Verification)
 */
bool cuda_check_device_health(CudaContext* ctx, int* error_code);

#ifdef __cplusplus
}
#endif

#endif // WORKER_FFI_H

// ---
// Built by Foundation-Alpha 🏗️
