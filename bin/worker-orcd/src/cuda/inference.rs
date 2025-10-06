//! Safe wrapper for CUDA inference session
//!
//! This module provides a safe Rust wrapper around the C CUDA inference API.
//! The `Inference` type implements RAII pattern - KV cache and other resources
//! are automatically freed when the inference is dropped.

use super::error::CudaError;
use super::ffi;
use super::model::Model;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};

/// Safe wrapper for inference session
///
/// Represents an active inference job with allocated KV cache and state.
/// Resources are automatically freed when dropped.
///
/// # Thread Safety
///
/// `Inference` is NOT `Send` or `Sync`. Inference sessions are single-threaded
/// and must not be moved between threads or accessed concurrently.
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
/// println!();
/// # Ok::<(), worker_orcd::cuda::CudaError>(())
/// ```
#[derive(Debug)]
pub struct Inference {
    ptr: *mut ffi::InferenceResult,
}

impl Inference {
    /// Start inference with given prompt and parameters
    ///
    /// # Arguments
    ///
    /// * `model` - Loaded model
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
    /// use worker_orcd::cuda::{Context, Inference};
    ///
    /// let ctx = Context::new(0)?;
    /// let model = ctx.load_model("/path/to/model.gguf")?;
    /// let mut inference = Inference::start(&model, "Hello", 10, 0.0, 42)?;
    /// # Ok::<(), worker_orcd::cuda::CudaError>(())
    /// ```
    pub fn start(
        model: &Model,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        seed: u64,
    ) -> Result<Self, CudaError> {
        // Validate parameters
        if max_tokens == 0 || max_tokens > 2048 {
            return Err(CudaError::InvalidParameter(format!(
                "max_tokens must be 1-2048, got {}",
                max_tokens
            )));
        }

        if !(0.0..=2.0).contains(&temperature) {
            return Err(CudaError::InvalidParameter(format!(
                "temperature must be 0.0-2.0, got {}",
                temperature
            )));
        }

        let prompt_cstr = CString::new(prompt)
            .map_err(|_| CudaError::InvalidParameter("Prompt contains null byte".to_string()))?;

        let mut error_code = 0;

        // SAFETY: model.as_ptr() is valid, prompt_cstr is valid CString,
        // error_code is valid pointer
        let ptr = unsafe {
            ffi::cuda_inference_start(
                model.as_ptr(),
                prompt_cstr.as_ptr(),
                max_tokens as c_int,
                temperature,
                seed,
                &mut error_code,
            )
        };

        if ptr.is_null() {
            return Err(CudaError::from_code(error_code));
        }

        Ok(Self { ptr })
    }

    /// Generate next token in inference sequence
    ///
    /// Returns `Ok(Some((token, index)))` if token generated,
    /// `Ok(None)` if sequence complete (EOS token reached),
    /// `Err(...)` if error occurred.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - CUDA kernel execution fails
    /// - VRAM exhausted during generation
    /// - Token buffer overflow
    ///
    /// # Example
    ///
    /// ```no_run
    /// use worker_orcd::cuda::Context;
    ///
    /// let ctx = Context::new(0)?;
    /// let model = ctx.load_model("/path/to/model.gguf")?;
    /// let mut inference = model.start_inference("Hello", 10, 0.0, 42)?;
    ///
    /// while let Some((token, idx)) = inference.next_token()? {
    ///     println!("Token {}: {}", idx, token);
    /// }
    /// # Ok::<(), worker_orcd::cuda::CudaError>(())
    /// ```
    pub fn next_token(&mut self) -> Result<Option<(String, u32)>, CudaError> {
        const TOKEN_BUFFER_SIZE: usize = 256;
        let mut token_buffer = vec![0u8; TOKEN_BUFFER_SIZE];
        let mut token_index = 0;
        let mut error_code = 0;

        // SAFETY: ptr is valid (checked in constructor), token_buffer is valid,
        // token_index and error_code are valid pointers
        let has_token = unsafe {
            ffi::cuda_inference_next_token(
                self.ptr,
                token_buffer.as_mut_ptr() as *mut c_char,
                TOKEN_BUFFER_SIZE as c_int,
                &mut token_index,
                &mut error_code,
            )
        };

        if error_code != 0 {
            return Err(CudaError::from_code(error_code));
        }

        if !has_token {
            return Ok(None);
        }

        // Find null terminator
        let null_pos = token_buffer.iter().position(|&b| b == 0).unwrap_or(TOKEN_BUFFER_SIZE);

        // Convert to UTF-8 string (lossy to handle invalid UTF-8)
        let token_str = String::from_utf8_lossy(&token_buffer[..null_pos]).into_owned();

        Ok(Some((token_str, token_index as u32)))
    }
}

impl Drop for Inference {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr is valid and only dropped once
            unsafe { ffi::cuda_inference_free(self.ptr) };
        }
    }
}

// NOTE: Inference is NOT Send/Sync (single-threaded inference)
// Do not implement Send or Sync traits

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::Context;

    #[test]
    fn test_inference_parameter_validation() {
        // max_tokens validation
        let ctx_result = Context::new(0);
        if let Ok(ctx) = ctx_result {
            if let Ok(model) = ctx.load_model("/tmp/test.gguf") {
                // max_tokens = 0 should fail
                let result = Inference::start(&model, "test", 0, 0.5, 42);
                assert!(result.is_err());
                if let Err(e) = result {
                    assert!(matches!(e, CudaError::InvalidParameter(_)));
                }

                // max_tokens > 2048 should fail
                let result = Inference::start(&model, "test", 3000, 0.5, 42);
                assert!(result.is_err());
                if let Err(e) = result {
                    assert!(matches!(e, CudaError::InvalidParameter(_)));
                }

                // temperature < 0.0 should fail
                let result = Inference::start(&model, "test", 10, -0.1, 42);
                assert!(result.is_err());
                if let Err(e) = result {
                    assert!(matches!(e, CudaError::InvalidParameter(_)));
                }

                // temperature > 2.0 should fail
                let result = Inference::start(&model, "test", 10, 2.1, 42);
                assert!(result.is_err());
                if let Err(e) = result {
                    assert!(matches!(e, CudaError::InvalidParameter(_)));
                }
            }
        }
    }

    #[test]
    fn test_inference_null_byte_in_prompt() {
        let ctx_result = Context::new(0);
        if let Ok(ctx) = ctx_result {
            if let Ok(model) = ctx.load_model("/tmp/test.gguf") {
                let result = Inference::start(&model, "test\0prompt", 10, 0.5, 42);
                assert!(result.is_err());
                if let Err(e) = result {
                    assert!(matches!(e, CudaError::InvalidParameter(_)));
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_inference_start_without_model() {
        let count = Context::device_count();
        if count > 0 {
            let ctx = Context::new(0).unwrap();
            // This will fail without a real model file
            let model_result = ctx.load_model("/nonexistent/model.gguf");
            assert!(model_result.is_err());
        }
    }
}

// ---
// Built by Foundation-Alpha 🏗️
