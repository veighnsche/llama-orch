//! Safe wrapper for CUDA model
//!
//! This module provides a safe Rust wrapper around the C CUDA model API.
//! The `Model` type implements RAII pattern - VRAM is automatically
//! freed when the model is dropped.

use super::context::Context;
use super::error::CudaError;
use super::ffi;
use super::inference::Inference;
use std::ffi::CString;

/// Safe wrapper for loaded CUDA model
///
/// Represents a model loaded into VRAM. The model owns its VRAM allocation
/// and will free it when dropped.
///
/// # Thread Safety
///
/// `Model` is `Send` and `Sync`. Models are immutable after loading
/// and can be safely shared between threads via Arc.
///
/// # Example
///
/// ```no_run
/// use worker_orcd::cuda::Context;
///
/// let ctx = Context::new(0)?;
/// let model = ctx.load_model("/path/to/model.gguf")?;
/// println!("VRAM usage: {} bytes", model.vram_bytes());
/// # Ok::<(), worker_orcd::cuda::CudaError>(())
/// ```
#[derive(Debug)]
pub struct Model {
    ptr: *mut ffi::CudaModel,
    vram_bytes: u64,
}

impl Model {
    /// Load model from GGUF file to VRAM
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `model_path` - Absolute path to .gguf file
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model file not found
    /// - Invalid GGUF format
    /// - Insufficient VRAM
    /// - Model path contains null byte
    ///
    /// # Example
    ///
    /// ```no_run
    /// use worker_orcd::cuda::{Context, Model};
    ///
    /// let ctx = Context::new(0)?;
    /// let model = Model::load(&ctx, "/path/to/model.gguf")?;
    /// # Ok::<(), worker_orcd::cuda::CudaError>(())
    /// ```
    pub fn load(ctx: &Context, model_path: &str) -> Result<Self, CudaError> {
        let path_cstr = CString::new(model_path).map_err(|_| {
            CudaError::InvalidParameter("Model path contains null byte".to_string())
        })?;

        let mut vram_bytes = 0;
        let mut error_code = 0;

        // SAFETY: ctx.as_ptr() is valid, path_cstr is valid CString,
        // vram_bytes and error_code are valid pointers
        let ptr = unsafe {
            ffi::cuda_load_model(
                ctx.as_ptr(),
                path_cstr.as_ptr(),
                &mut vram_bytes,
                &mut error_code,
            )
        };

        if ptr.is_null() {
            return Err(CudaError::from_code(error_code));
        }

        Ok(Self { ptr, vram_bytes })
    }

    /// Get VRAM bytes used by this model
    ///
    /// Returns the total VRAM allocated for model weights.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use worker_orcd::cuda::Context;
    ///
    /// let ctx = Context::new(0)?;
    /// let model = ctx.load_model("/path/to/model.gguf")?;
    /// println!("Model uses {} bytes of VRAM", model.vram_bytes());
    /// # Ok::<(), worker_orcd::cuda::CudaError>(())
    /// ```
    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    /// Check VRAM residency for model weights
    ///
    /// Verifies that all model weights are still resident in VRAM
    /// (no RAM fallback or swapping).
    ///
    /// # Errors
    ///
    /// Returns error if residency check fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use worker_orcd::cuda::Context;
    ///
    /// let ctx = Context::new(0)?;
    /// let model = ctx.load_model("/path/to/model.gguf")?;
    /// let resident = model.check_vram_residency()?;
    /// if !resident {
    ///     eprintln!("Model weights not in VRAM!");
    /// }
    /// # Ok::<(), worker_orcd::cuda::CudaError>(())
    /// ```
    pub fn check_vram_residency(&self) -> Result<bool, CudaError> {
        let mut error_code = 0;

        // SAFETY: ptr is valid (checked in constructor), error_code is valid pointer
        let resident = unsafe { ffi::cuda_check_vram_residency(self.ptr, &mut error_code) };

        if error_code != 0 {
            return Err(CudaError::from_code(error_code));
        }

        Ok(resident)
    }

    /// Start inference with given prompt and parameters
    ///
    /// # Arguments
    ///
    /// * `prompt` - Input prompt (UTF-8 string)
    /// * `max_tokens` - Maximum tokens to generate (1-2048)
    /// * `temperature` - Sampling temperature (0.0-2.0, 0.0 = greedy)
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Prompt contains null byte
    /// - Parameters out of range
    /// - Insufficient VRAM for KV cache
    /// - Inference initialization fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use worker_orcd::cuda::Context;
    ///
    /// let ctx = Context::new(0)?;
    /// let model = ctx.load_model("/path/to/model.gguf")?;
    /// let mut inference = model.start_inference("Write a haiku", 100, 0.7, 42)?;
    ///
    /// while let Some((token, idx)) = inference.next_token()? {
    ///     print!("{}", token);
    /// }
    /// # Ok::<(), worker_orcd::cuda::CudaError>(())
    /// ```
    pub fn start_inference(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        seed: u64,
    ) -> Result<Inference, CudaError> {
        Inference::start(self, prompt, max_tokens, temperature, seed)
    }

    /// Get raw pointer (for internal use only)
    #[doc(hidden)]
    pub(crate) fn as_ptr(&self) -> *mut ffi::CudaModel {
        self.ptr
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr is valid and only dropped once
            unsafe { ffi::cuda_unload_model(self.ptr) };
        }
    }
}

// SAFETY: Model can be moved between threads (model is immutable after load)
// Model is also Sync because it's immutable and all operations are thread-safe
unsafe impl Send for Model {}
unsafe impl Sync for Model {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_model_load_without_cuda() {
        let ctx_result = Context::new(0);
        assert!(ctx_result.is_err());
    }

    #[test]
    fn test_model_load_invalid_path() {
        // Path with null byte should fail
        let ctx_result = Context::new(0);
        if let Ok(ctx) = ctx_result {
            let result = Model::load(&ctx, "path\0with\0null");
            assert!(result.is_err());
            if let Err(e) = result {
                assert!(matches!(e, CudaError::InvalidParameter(_)));
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_model_load_nonexistent_file() {
        let count = Context::device_count();
        if count > 0 {
            let ctx = Context::new(0).unwrap();
            let result = Model::load(&ctx, "/nonexistent/model.gguf");
            assert!(result.is_err());
            if let Err(e) = result {
                assert!(matches!(e, CudaError::ModelLoadFailed(_)));
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_model_vram_bytes() {
        let count = Context::device_count();
        if count > 0 {
            let ctx = Context::new(0).unwrap();
            // This will fail without a real model file, but tests the API
            let result = Model::load(&ctx, "/tmp/test.gguf");
            if let Ok(model) = result {
                let vram = model.vram_bytes();
                assert!(vram > 0);
            }
        }
    }
}

// ---
// Built by Foundation-Alpha 🏗️
