/**
 * Worker CUDA - Type Definitions
 * 
 * Defines opaque handle types for FFI boundary.
 * Implementation details are hidden from Rust.
 */

#ifndef WORKER_TYPES_H
#define WORKER_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to CUDA context.
 * 
 * Represents a CUDA device context with initialized state.
 * Created by cuda_init(), destroyed by cuda_destroy().
 * 
 * Thread safety: NOT thread-safe. Each context is single-threaded.
 * Memory ownership: Rust must call cuda_destroy() to free.
 */
typedef struct CudaContext CudaContext;

/**
 * Opaque handle to loaded model.
 * 
 * Represents a model loaded into VRAM with all weights resident.
 * Created by cuda_load_model(), destroyed by cuda_unload_model().
 * 
 * Thread safety: NOT thread-safe. Single-threaded access only.
 * Memory ownership: Rust must call cuda_unload_model() to free VRAM.
 */
typedef struct CudaModel CudaModel;

/**
 * Opaque handle to inference session.
 * 
 * Represents an active inference job with KV cache and state.
 * Created by cuda_inference_start(), destroyed by cuda_inference_free().
 * 
 * Thread safety: NOT thread-safe. Single-threaded access only.
 * Memory ownership: Rust must call cuda_inference_free() to free resources.
 */
typedef struct InferenceResult InferenceResult;

#ifdef __cplusplus
}
#endif

#endif // WORKER_TYPES_H

// ---
// Built by Foundation-Alpha 🏗️
