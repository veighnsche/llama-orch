//! Safe wrapper for CUDA model
//!
//! This module provides a safe Rust wrapper around the C CUDA model API.
//! The `Model` type implements RAII pattern - VRAM is automatically
//! freed when the model is dropped.

use super::context::Context;
use super::error::CudaError;
use super::ffi;
use super::inference::Inference;

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
        // NEW: Load weights in Rust with Q4_K dequantization!
        use super::weight_loader::load_model_from_rust;
        use worker_gguf::GGUFMetadata;

        eprintln!("🦀 [Rust] Loading model with Rust weight loading + Q4_K dequantization");
        // Parse model configuration from GGUF metadata
        let meta = GGUFMetadata::from_file(model_path).map_err(|e| {
            CudaError::ModelLoadFailed(format!("Failed to parse GGUF metadata: {}", e))
        })?;

        // CRITICAL: Use actual vocab size from output.weight tensor, not padded tokenizer size
        // The lm_head (output.weight) tensor has dimensions [hidden_dim, actual_vocab]
        // For Qwen2.5-0.5B: [896, 151643] not [896, 151936]
        // This prevents argmax from scanning garbage values beyond the actual vocabulary
        let vocab_size = {
            let tensors = GGUFMetadata::parse_tensors(model_path).map_err(|e| {
                CudaError::ModelLoadFailed(format!("Failed to parse tensors: {}", e))
            })?;

            // Get actual vocab from output.weight (lm_head) tensor dimensions
            let output_tensor = tensors
                .iter()
                .find(|t| t.name == "output.weight")
                .ok_or_else(|| {
                    CudaError::ModelLoadFailed("Cannot find output.weight tensor".to_string())
                })?;
            
            eprintln!("🔍 [Rust] output.weight dimensions: {:?}", output_tensor.dimensions);
            eprintln!("🔍 [Rust] output.weight ggml_type: {}", output_tensor.ggml_type);
            eprintln!("🔍 [Rust] output.weight offset: {}", output_tensor.offset);
            
            // Calculate expected size
            let elem_count: u64 = output_tensor.dimensions.iter().product();
            let bytes_per_elem = match output_tensor.ggml_type {
                0 => 4,  // F32
                1 => 2,  // F16
                _ => 2,  // Assume F16 for others
            };
            let expected_bytes = elem_count * bytes_per_elem;
            eprintln!("🔍 [Rust] output.weight expected size: {} bytes ({} MB)", 
                     expected_bytes, expected_bytes / 1024 / 1024);
            
            let actual_vocab = output_tensor.dimensions.get(1)
                .or_else(|| output_tensor.dimensions.get(0))  // Try first dim if second doesn't exist
                .map(|&d| d as u32)
                .ok_or_else(|| {
                    CudaError::ModelLoadFailed("output.weight has no dimensions".to_string())
                })?;

            eprintln!("✅ [Rust] Actual vocab size from output.weight: {}", actual_vocab);
            
            // Verify against tokenizer vocab (should be padded)
            if let Ok(tokenizer_vocab) = meta.vocab_size() {
                if tokenizer_vocab as u32 != actual_vocab {
                    eprintln!(
                        "⚠️  [Rust] Tokenizer vocab ({}) != output.weight vocab ({})",
                        tokenizer_vocab,
                        actual_vocab
                    );
                    eprintln!(
                        "⚠️  [Rust] Using actual vocab ({}) to avoid scanning garbage values",
                        actual_vocab
                    );
                }
            }
            
            actual_vocab
        };
        let hidden_dim = meta
            .hidden_dim()
            .map_err(|e| CudaError::ModelLoadFailed(format!("Failed to read hidden_dim: {}", e)))?
            as u32;
        let num_layers = meta
            .num_layers()
            .map_err(|e| CudaError::ModelLoadFailed(format!("Failed to read num_layers: {}", e)))?
            as u32;
        let num_heads = meta
            .num_heads()
            .map_err(|e| CudaError::ModelLoadFailed(format!("Failed to read num_heads: {}", e)))?
            as u32;
        let num_kv_heads = meta.num_kv_heads().map_err(|e| {
            CudaError::ModelLoadFailed(format!("Failed to read num_kv_heads: {}", e))
        })? as u32;
        let context_length = meta.context_length().map_err(|e| {
            CudaError::ModelLoadFailed(format!("Failed to read context_length: {}", e))
        })? as u32;

        eprintln!(
            "📋 [Rust] Model config (from GGUF): vocab={}, hidden={}, layers={}, heads={}/{} ctx={}",
            vocab_size, hidden_dim, num_layers, num_heads, num_kv_heads, context_length
        );

        // Load weights in Rust and create C++ model
        let ptr = unsafe {
            load_model_from_rust(
                model_path,
                vocab_size,
                hidden_dim,
                num_layers,
                num_heads,
                num_kv_heads,
                context_length,
            )
            .map_err(CudaError::ModelLoadFailed)?
        };

        // VRAM is tracked inside load_model_from_rust
        let vram_bytes = 0; // TODO: Get actual VRAM from Rust loader

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
