//! GPT Model Implementation
//!
//! Supports GPT-2 and GPT-3 architectures.
//! Integrates with Foundation layer via LlamaModelAdapter.
//!
//! # Architecture
//!
//! GPT models use:
//! - LayerNorm (instead of RMSNorm)
//! - GELU activation (instead of SiLU)
//! - Multi-Head Attention (MHA, no GQA)
//! - Absolute positional embeddings (no RoPE)
//!
//! # Spec References
//! - M0-W-1220: GPT model specification
//! - GT-XXX: GPT team implementation stories

use crate::cuda_ffi::{CudaContext, CudaError};
use thiserror::Error;

/// GPT model configuration
#[derive(Debug, Clone)]
pub struct GPTConfig {
    /// Vocabulary size (50257 for GPT-2, 50304 for GPT-3)
    pub vocab_size: usize,
    /// Hidden dimension (768 for GPT-2 small, 1600 for GPT-2 medium, etc.)
    pub hidden_dim: usize,
    /// Number of transformer layers (12 for GPT-2 small, 24 for medium, etc.)
    pub num_layers: usize,
    /// Number of attention heads (12 for GPT-2 small, 25 for medium, etc.)
    pub num_heads: usize,
    /// Maximum sequence length (1024 for GPT-2, 2048 for GPT-3)
    pub max_seq_len: usize,
    /// FFN intermediate dimension (4 * hidden_dim typically)
    pub ffn_dim: usize,
}

impl GPTConfig {
    /// GPT-2 Small (117M parameters)
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50257,
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            max_seq_len: 1024,
            ffn_dim: 3072,
        }
    }

    /// GPT-2 Medium (345M parameters)
    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size: 50257,
            hidden_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            max_seq_len: 1024,
            ffn_dim: 4096,
        }
    }

    /// GPT-2 Large (774M parameters)
    pub fn gpt2_large() -> Self {
        Self {
            vocab_size: 50257,
            hidden_dim: 1280,
            num_layers: 36,
            num_heads: 20,
            max_seq_len: 1024,
            ffn_dim: 5120,
        }
    }

    /// GPT-2 XL (1.5B parameters)
    pub fn gpt2_xl() -> Self {
        Self {
            vocab_size: 50257,
            hidden_dim: 1600,
            num_layers: 48,
            num_heads: 25,
            max_seq_len: 1024,
            ffn_dim: 6400,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), GPTError> {
        if self.vocab_size == 0 {
            return Err(GPTError::InvalidConfig("vocab_size must be > 0".to_string()));
        }
        if self.hidden_dim == 0 {
            return Err(GPTError::InvalidConfig("hidden_dim must be > 0".to_string()));
        }
        if self.num_layers == 0 {
            return Err(GPTError::InvalidConfig("num_layers must be > 0".to_string()));
        }
        if self.num_heads == 0 {
            return Err(GPTError::InvalidConfig("num_heads must be > 0".to_string()));
        }
        if self.hidden_dim % self.num_heads != 0 {
            return Err(GPTError::InvalidConfig(
                "hidden_dim must be divisible by num_heads".to_string(),
            ));
        }
        Ok(())
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }
}

/// GPT model instance
#[derive(Debug)]
pub struct GPTModel {
    /// Model configuration
    pub config: GPTConfig,
    /// Total VRAM usage in bytes
    pub total_vram_bytes: usize,
    // TODO(GPT-Gamma): Add weight tensor fields
    // - token_embeddings: SafeCudaPtr
    // - position_embeddings: SafeCudaPtr
    // - layers: Vec<GPTLayer>
    // - final_layernorm: SafeCudaPtr
    // - lm_head: SafeCudaPtr
}

/// GPT errors
#[derive(Debug, Error)]
pub enum GPTError {
    /// Invalid configuration
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Forward pass failed
    #[error("Forward pass failed: {0}")]
    ForwardPassFailed(String),

    /// CUDA error
    #[error("CUDA error: {0}")]
    CudaError(#[from] CudaError),

    /// GGUF parsing error
    #[error("GGUF parsing error: {0}")]
    GGUFError(String),

    /// Weight loading error
    #[error("Weight loading error: {0}")]
    WeightLoadingError(String),
}

/// GPT weight loader
pub struct GPTWeightLoader;

impl GPTWeightLoader {
    /// Load GPT model from GGUF file to VRAM
    ///
    /// # Arguments
    /// - `path`: Path to GGUF file
    /// - `config`: Model configuration
    ///
    /// # Returns
    /// Loaded model instance
    ///
    /// # Errors
    /// Returns error if:
    /// - File not found
    /// - Invalid GGUF format
    /// - VRAM allocation fails
    /// - Config validation fails
    pub fn load_to_vram(_path: &str, config: &GPTConfig) -> Result<GPTModel, GPTError> {
        // Validate config
        config.validate()?;

        // TODO(GPT-Gamma): Implement GGUF loading
        // 1. Parse GGUF file
        // 2. Extract metadata
        // 3. Validate tensor shapes
        // 4. Allocate VRAM
        // 5. Copy weights to GPU

        let total_vram_bytes = Self::calculate_vram_usage(config);

        Ok(GPTModel { config: config.clone(), total_vram_bytes })
    }

    /// Calculate total VRAM usage for model
    pub fn calculate_vram_usage(config: &GPTConfig) -> usize {
        // Weights
        let embedding_params = config.vocab_size * config.hidden_dim;
        let position_params = config.max_seq_len * config.hidden_dim;
        let layer_params = Self::calculate_layer_params(config);
        let total_params = embedding_params + position_params + (layer_params * config.num_layers);

        // FP16: 2 bytes per parameter
        let weights_bytes = total_params * 2;

        // KV cache (MHA: num_heads = num_kv_heads)
        let kv_cache_bytes =
            2 * config.num_layers * config.num_heads * config.head_dim() * config.max_seq_len * 2;

        // Activations (conservative estimate)
        let activation_bytes = config.max_seq_len * config.hidden_dim * 2 * 10;

        // Total with 10% overhead
        let total = weights_bytes + kv_cache_bytes + activation_bytes;
        (total as f64 * 1.1) as usize
    }

    fn calculate_layer_params(config: &GPTConfig) -> usize {
        // Attention: Q, K, V projections + output projection
        let attn_params = 4 * config.hidden_dim * config.hidden_dim;

        // FFN: up projection + down projection
        let ffn_params = config.hidden_dim * config.ffn_dim + config.ffn_dim * config.hidden_dim;

        // LayerNorm: 2 * hidden_dim (gamma + beta) for 2 LayerNorms
        let ln_params = 4 * config.hidden_dim;

        attn_params + ffn_params + ln_params
    }
}

/// GPT forward pass configuration
#[derive(Debug, Clone)]
pub struct GPTForwardConfig {
    /// True for prefill, false for decode
    pub is_prefill: bool,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length
    pub seq_len: usize,
    /// KV cache length
    pub cache_len: usize,
    /// Sampling temperature
    pub temperature: f32,
    /// Random seed
    pub seed: u32,
}

/// GPT forward pass implementation
pub struct GPTForward;

impl GPTForward {
    /// Prefill: process full prompt
    pub fn prefill(
        _model: &GPTModel,
        input_ids: &[u32],
        _config: &GPTForwardConfig,
    ) -> Result<Vec<u32>, GPTError> {
        // TODO(GPT-Gamma): Implement prefill
        // 1. Embed tokens + positions
        // 2. Run through transformer layers
        // 3. Apply final LayerNorm
        // 4. Project to vocabulary
        // 5. Sample tokens

        // Stub: return input
        Ok(input_ids.to_vec())
    }

    /// Decode: generate single token
    pub fn decode(
        _model: &GPTModel,
        input_id: u32,
        _config: &GPTForwardConfig,
    ) -> Result<u32, GPTError> {
        // TODO(GPT-Gamma): Implement decode
        // 1. Embed token + position
        // 2. Run through transformer layers (use KV cache)
        // 3. Apply final LayerNorm
        // 4. Project to vocabulary
        // 5. Sample token

        // Stub: return input
        Ok(input_id)
    }

    /// Generate: autoregressive generation
    pub fn generate(
        model: &GPTModel,
        input_ids: &[u32],
        max_tokens: usize,
        config: &GPTForwardConfig,
    ) -> Result<Vec<u32>, GPTError> {
        // TODO(GPT-Gamma): Implement generation loop
        // 1. Prefill with prompt
        // 2. Loop decode for max_tokens
        // 3. Handle EOS token
        // 4. Return generated sequence

        // Stub: return input + dummy tokens
        let mut output = input_ids.to_vec();
        for i in 0..max_tokens {
            output.push((input_ids.len() + i) as u32);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_small_config() {
        let config = GPTConfig::gpt2_small();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_config_validation() {
        let valid_config = GPTConfig::gpt2_small();
        assert!(valid_config.validate().is_ok());

        let invalid_config = GPTConfig { vocab_size: 0, ..GPTConfig::gpt2_small() };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_vram_calculation() {
        let config = GPTConfig::gpt2_small();
        let vram = GPTWeightLoader::calculate_vram_usage(&config);
        assert!(vram > 0);
        // GPT-2 small should be under 1GB
        assert!(vram < 1024 * 1024 * 1024);
    }

    #[test]
    fn test_model_loading() {
        let config = GPTConfig::gpt2_small();
        let model = GPTWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
        assert_eq!(model.config.vocab_size, 50257);
        assert!(model.total_vram_bytes > 0);
    }
}

// ---
// Built by Foundation-Alpha 🏗️
