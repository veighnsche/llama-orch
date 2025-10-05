//! Raw FFI declarations for CUDA C API
//!
//! This module contains unsafe extern "C" declarations that match
//! the C API defined in `cuda/include/worker_ffi.h`.
//!
//! # Safety
//!
//! All functions in this module are `unsafe` and require careful handling:
//! - Pointers must be valid and properly aligned
//! - Strings must be null-terminated UTF-8
//! - Error codes must be checked after every call
//! - Resources must be freed with corresponding free functions

use std::os::raw::{c_char, c_int};

// ============================================================================
// Opaque Handle Types
// ============================================================================

/// Opaque handle to CUDA context (matches C definition)
#[repr(C)]
pub struct CudaContext {
    _private: [u8; 0],
}

/// Opaque handle to loaded model (matches C definition)
#[repr(C)]
pub struct CudaModel {
    _private: [u8; 0],
}

/// Opaque handle to inference session (matches C definition)
#[repr(C)]
pub struct InferenceResult {
    _private: [u8; 0],
}

// ============================================================================
// Raw FFI Declarations (unsafe)
// ============================================================================

#[cfg(feature = "cuda")]
extern "C" {
    // ========================================================================
    // Context Management
    // ========================================================================

    /// Initialize CUDA context for specified GPU device.
    ///
    /// # Safety
    ///
    /// - `error_code` must be a valid pointer to writable i32
    /// - Returned pointer must be freed with `cuda_destroy`
    /// - Caller must check error_code and handle NULL return
    pub fn cuda_init(gpu_device: c_int, error_code: *mut c_int) -> *mut CudaContext;

    /// Destroy CUDA context and free all resources.
    ///
    /// # Safety
    ///
    /// - `ctx` must be a valid pointer from `cuda_init` or NULL
    /// - `ctx` must not be used after this call
    /// - Safe to call with NULL (no-op)
    pub fn cuda_destroy(ctx: *mut CudaContext);

    /// Get number of available CUDA devices.
    ///
    /// # Safety
    ///
    /// This function is safe to call (no pointer parameters).
    pub fn cuda_get_device_count() -> c_int;

    // ========================================================================
    // Model Loading
    // ========================================================================

    /// Load model from GGUF file to VRAM.
    ///
    /// # Safety
    ///
    /// - `ctx` must be a valid pointer from `cuda_init`
    /// - `model_path` must be a valid null-terminated UTF-8 string
    /// - `vram_bytes_used` must be a valid pointer to writable u64
    /// - `error_code` must be a valid pointer to writable i32
    /// - Returned pointer must be freed with `cuda_unload_model`
    /// - Caller must check error_code and handle NULL return
    pub fn cuda_load_model(
        ctx: *mut CudaContext,
        model_path: *const c_char,
        vram_bytes_used: *mut u64,
        error_code: *mut c_int,
    ) -> *mut CudaModel;

    /// Unload model and free VRAM.
    ///
    /// # Safety
    ///
    /// - `model` must be a valid pointer from `cuda_load_model` or NULL
    /// - `model` must not be used after this call
    /// - Safe to call with NULL (no-op)
    pub fn cuda_unload_model(model: *mut CudaModel);

    /// Get current VRAM usage for model.
    ///
    /// # Safety
    ///
    /// - `model` must be a valid pointer from `cuda_load_model` or NULL
    /// - Returns 0 if model is NULL
    pub fn cuda_model_get_vram_usage(model: *mut CudaModel) -> u64;

    // ========================================================================
    // Inference Execution
    // ========================================================================

    /// Start inference job with given prompt and parameters.
    ///
    /// # Safety
    ///
    /// - `model` must be a valid pointer from `cuda_load_model`
    /// - `prompt` must be a valid null-terminated UTF-8 string
    /// - `max_tokens` must be in range 1-2048
    /// - `temperature` must be in range 0.0-2.0
    /// - `error_code` must be a valid pointer to writable i32
    /// - Returned pointer must be freed with `cuda_inference_free`
    /// - Caller must check error_code and handle NULL return
    pub fn cuda_inference_start(
        model: *mut CudaModel,
        prompt: *const c_char,
        max_tokens: c_int,
        temperature: f32,
        seed: u64,
        error_code: *mut c_int,
    ) -> *mut InferenceResult;

    /// Generate next token in inference sequence.
    ///
    /// # Safety
    ///
    /// - `result` must be a valid pointer from `cuda_inference_start`
    /// - `token_out` must be a valid pointer to writable buffer
    /// - `token_buffer_size` must match size of `token_out` buffer
    /// - `token_index` may be NULL (optional output)
    /// - `error_code` must be a valid pointer to writable i32
    /// - Caller must check error_code
    /// - Returns true if token generated, false if sequence complete
    pub fn cuda_inference_next_token(
        result: *mut InferenceResult,
        token_out: *mut c_char,
        token_buffer_size: c_int,
        token_index: *mut c_int,
        error_code: *mut c_int,
    ) -> bool;

    /// Free inference result and associated resources.
    ///
    /// # Safety
    ///
    /// - `result` must be a valid pointer from `cuda_inference_start` or NULL
    /// - `result` must not be used after this call
    /// - Safe to call with NULL (no-op)
    pub fn cuda_inference_free(result: *mut InferenceResult);

    // ========================================================================
    // Health & Monitoring
    // ========================================================================

    /// Check VRAM residency for model weights.
    ///
    /// # Safety
    ///
    /// - `model` must be a valid pointer from `cuda_load_model`
    /// - `error_code` must be a valid pointer to writable i32
    /// - Caller must check error_code
    pub fn cuda_check_vram_residency(model: *mut CudaModel, error_code: *mut c_int) -> bool;

    /// Get current VRAM usage for model.
    ///
    /// # Safety
    ///
    /// - `model` must be a valid pointer from `cuda_load_model` or NULL
    /// - Returns 0 if model is NULL
    pub fn cuda_get_vram_usage(model: *mut CudaModel) -> u64;

    /// Get process-wide VRAM usage.
    ///
    /// # Safety
    ///
    /// - `ctx` must be a valid pointer from `cuda_init` or NULL
    /// - Returns 0 if ctx is NULL
    pub fn cuda_get_process_vram_usage(ctx: *mut CudaContext) -> u64;

    /// Check CUDA device health.
    ///
    /// # Safety
    ///
    /// - `ctx` must be a valid pointer from `cuda_init`
    /// - `error_code` must be a valid pointer to writable i32
    /// - Caller must check error_code
    pub fn cuda_check_device_health(ctx: *mut CudaContext, error_code: *mut c_int) -> bool;

    // ========================================================================
    // Error Handling
    // ========================================================================

    /// Get human-readable error message for error code.
    ///
    /// # Safety
    ///
    /// - Returns pointer to static string (never NULL)
    /// - Returned pointer is valid for program lifetime
    /// - Caller must NOT free the returned pointer
    pub fn cuda_error_message(error_code: c_int) -> *const c_char;
}

// ============================================================================
// Stub Implementations (when CUDA feature is disabled)
// ============================================================================

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_init(_gpu_device: c_int, error_code: *mut c_int) -> *mut CudaContext {
    *error_code = 8; // CUDA_ERROR_DEVICE_NOT_FOUND
    std::ptr::null_mut()
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_destroy(_ctx: *mut CudaContext) {}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_get_device_count() -> c_int {
    0
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_load_model(
    _ctx: *mut CudaContext,
    _model_path: *const c_char,
    _vram_bytes_used: *mut u64,
    error_code: *mut c_int,
) -> *mut CudaModel {
    *error_code = 3; // CUDA_ERROR_MODEL_LOAD_FAILED
    std::ptr::null_mut()
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_unload_model(_model: *mut CudaModel) {}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_model_get_vram_usage(_model: *mut CudaModel) -> u64 {
    0
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_inference_start(
    _model: *mut CudaModel,
    _prompt: *const c_char,
    _max_tokens: c_int,
    _temperature: f32,
    _seed: u64,
    error_code: *mut c_int,
) -> *mut InferenceResult {
    *error_code = 4; // CUDA_ERROR_INFERENCE_FAILED
    std::ptr::null_mut()
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_inference_next_token(
    _result: *mut InferenceResult,
    _token_out: *mut c_char,
    _token_buffer_size: c_int,
    _token_index: *mut c_int,
    error_code: *mut c_int,
) -> bool {
    *error_code = 4; // CUDA_ERROR_INFERENCE_FAILED
    false
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_inference_free(_result: *mut InferenceResult) {}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_check_vram_residency(_model: *mut CudaModel, error_code: *mut c_int) -> bool {
    *error_code = 7; // CUDA_ERROR_VRAM_RESIDENCY_FAILED
    false
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_get_vram_usage(_model: *mut CudaModel) -> u64 {
    0
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_get_process_vram_usage(_ctx: *mut CudaContext) -> u64 {
    0
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_check_device_health(_ctx: *mut CudaContext, error_code: *mut c_int) -> bool {
    *error_code = 1; // CUDA_ERROR_INVALID_DEVICE
    false
}

#[cfg(not(feature = "cuda"))]
pub unsafe fn cuda_error_message(error_code: c_int) -> *const c_char {
    use std::ffi::CString;

    let msg = match error_code {
        0 => "Success",
        1 => "Invalid device",
        2 => "Out of memory",
        3 => "Model load failed",
        4 => "Inference failed",
        5 => "Invalid parameter",
        6 => "Kernel launch failed",
        7 => "VRAM residency failed",
        8 => "Device not found",
        _ => "Unknown error",
    };

    // Leak the CString to get a static pointer (stub only)
    CString::new(msg).unwrap().into_raw()
}

// ---
// Built by Foundation-Alpha 🏗️
