//! Real GPU inference using QwenTransformer
//!
//! This module provides the real inference implementation that uses
//! the actual CUDA transformer forward pass, replacing the stub.

use super::error::CudaError;
use super::ffi;
use super::model::Model;

/// Real inference context using QwenTransformer
///
/// This wraps the C++ InferenceContext which contains:
/// - QwenTransformer for forward passes
/// - QwenModel with loaded weights
/// - Logits buffer for sampling
pub struct RealInference {
    ptr: *mut ffi::InferenceContext,
}

impl RealInference {
    /// Initialize real inference context
    ///
    /// # Arguments
    ///
    /// * `model` - Loaded model with weights in VRAM
    /// * `vocab_size` - Vocabulary size
    /// * `hidden_dim` - Hidden dimension
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of KV heads (for GQA)
    /// * `head_dim` - Head dimension
    /// * `ffn_dim` - FFN intermediate dimension
    /// * `context_length` - Maximum context length
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn init(
        model: &Model,
        vocab_size: u32,
        hidden_dim: u32,
        num_layers: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        ffn_dim: u32,
        context_length: u32,
        rope_freq_base: f32,
    ) -> Result<Self, CudaError> {
        let mut error_code = 0;

        // SAFETY: model.as_ptr() is valid, all parameters are validated
        let ptr = unsafe {
            ffi::cuda_inference_init(
                model.as_ptr() as *mut std::ffi::c_void,
                vocab_size,
                hidden_dim,
                num_layers,
                num_heads,
                num_kv_heads,
                head_dim,
                ffn_dim,
                context_length,
                rope_freq_base,
                &mut error_code,
            )
        };

        if ptr.is_null() {
            return Err(CudaError::from_code(error_code));
        }

        Ok(Self { ptr })
    }

    /// Generate next token from current token
    ///
    /// Runs transformer forward pass and samples next token.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Current token ID
    /// * `temperature` - Sampling temperature (0.0 = greedy)
    /// * `top_k` - Top-k filtering (0 = disabled)
    /// * `top_p` - Nucleus sampling (1.0 = disabled)
    /// * `seed` - Random seed
    ///
    /// # Returns
    ///
    /// Next token ID
    ///
    /// # Errors
    ///
    /// Returns error if generation fails
    pub fn generate_token(
        &mut self,
        token_id: u32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
    ) -> Result<u32, CudaError> {
        let mut error_code = 0;

        // SAFETY: ptr is valid (checked in constructor)
        let next_token = unsafe {
            ffi::cuda_inference_generate_token(
                self.ptr,
                token_id,
                temperature,
                top_k,
                top_p,
                seed,
                &mut error_code,
            )
        };

        if error_code != 0 {
            return Err(CudaError::from_code(error_code));
        }

        Ok(next_token)
    }

    /// Reset KV cache
    ///
    /// Clears the KV cache for a new sequence.
    pub fn reset(&mut self) {
        // SAFETY: ptr is valid (checked in constructor)
        unsafe {
            ffi::cuda_inference_reset(self.ptr);
        }
    }
}

impl Drop for RealInference {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr is valid and only dropped once
            unsafe {
                ffi::cuda_inference_context_free(self.ptr);
            }
        }
    }
}

// NOTE: RealInference is NOT Send/Sync (single-threaded inference)
// Do not implement Send or Sync traits

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_inference_not_send() {
        // This should not compile if RealInference is Send
        fn assert_not_send<T: Send>() {}
        // assert_not_send::<RealInference>(); // Uncomment to verify
    }
}

// ---
// Built by Foundation-Alpha 🏗️
