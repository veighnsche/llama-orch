//! GPT model configuration
//!
//! Defines configuration structures for GPT-family models (GPT-2, GPT-OSS-20B).
//! Parsed from GGUF metadata during model loading.
//!
//! Spec: M0-W-1211, M0-W-1212

use serde::{Deserialize, Serialize};

/// GPT model configuration
///
/// Contains all hyperparameters needed for GPT inference.
/// Extracted from GGUF metadata keys prefixed with "gpt2." or "gpt.".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GPTConfig {
    /// Architecture name ("gpt2", "gpt", "gpt-neox")
    pub architecture: String,

    /// Context window size (max sequence length)
    pub context_length: u32,

    /// Hidden size / embedding dimension (d_model)
    pub embedding_length: u32,

    /// Number of transformer layers
    pub block_count: u32,

    /// Number of attention heads
    pub attention_head_count: u32,

    /// FFN intermediate size (typically 4 * embedding_length)
    pub ffn_length: u32,

    /// Vocabulary size
    pub vocab_size: u32,
}

impl GPTConfig {
    /// Create a new GPT configuration
    pub fn new(
        architecture: String,
        context_length: u32,
        embedding_length: u32,
        block_count: u32,
        attention_head_count: u32,
        ffn_length: u32,
        vocab_size: u32,
    ) -> Self {
        Self {
            architecture,
            context_length,
            embedding_length,
            block_count,
            attention_head_count,
            ffn_length,
            vocab_size,
        }
    }

    /// Calculate head dimension (d_head = d_model / num_heads)
    pub fn head_dim(&self) -> u32 {
        self.embedding_length / self.attention_head_count
    }

    /// Validate configuration parameters
    ///
    /// # Returns
    /// * `Ok(())` - Configuration is valid
    /// * `Err(String)` - Configuration is invalid with error message
    pub fn validate(&self) -> Result<(), String> {
        // Validate architecture
        if !["gpt2", "gpt", "gpt-neox"].contains(&self.architecture.as_str()) {
            return Err(format!(
                "Unsupported architecture: {}. Expected gpt2, gpt, or gpt-neox",
                self.architecture
            ));
        }

        // Validate dimensions
        if self.context_length == 0 {
            return Err("context_length must be > 0".to_string());
        }
        if self.embedding_length == 0 {
            return Err("embedding_length must be > 0".to_string());
        }
        if self.block_count == 0 {
            return Err("block_count must be > 0".to_string());
        }
        if self.attention_head_count == 0 {
            return Err("attention_head_count must be > 0".to_string());
        }
        if self.ffn_length == 0 {
            return Err("ffn_length must be > 0".to_string());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".to_string());
        }

        // Validate head dimension is integer
        if self.embedding_length % self.attention_head_count != 0 {
            return Err(format!(
                "embedding_length ({}) must be divisible by attention_head_count ({})",
                self.embedding_length, self.attention_head_count
            ));
        }

        // Validate reasonable ranges
        if self.context_length > 1_000_000 {
            return Err(format!(
                "context_length ({}) exceeds maximum (1,000,000)",
                self.context_length
            ));
        }
        if self.embedding_length > 100_000 {
            return Err(format!(
                "embedding_length ({}) exceeds maximum (100,000)",
                self.embedding_length
            ));
        }
        if self.block_count > 1000 {
            return Err(format!("block_count ({}) exceeds maximum (1000)", self.block_count));
        }

        Ok(())
    }

    /// Estimate VRAM usage in bytes (rough approximation)
    ///
    /// Calculates approximate VRAM needed for model weights only.
    /// Does not include KV cache or activation memory.
    pub fn estimate_vram_bytes(&self, bytes_per_param: f32) -> u64 {
        // Approximate parameter count for GPT-2 style models:
        // - Token embeddings: vocab_size * embedding_length
        // - Position embeddings: context_length * embedding_length
        // - Per layer:
        //   - Attention QKV: 3 * embedding_length^2
        //   - Attention output: embedding_length^2
        //   - FFN: 2 * embedding_length * ffn_length
        //   - LayerNorm: 2 * embedding_length (negligible)
        // - Final LayerNorm: embedding_length
        // - LM head: vocab_size * embedding_length (often tied to embeddings)

        let vocab_size = self.vocab_size as u64;
        let ctx_len = self.context_length as u64;
        let d_model = self.embedding_length as u64;
        let n_layers = self.block_count as u64;
        let ffn_size = self.ffn_length as u64;

        let embeddings = vocab_size * d_model + ctx_len * d_model;
        let attention_per_layer = 4 * d_model * d_model; // QKV + output
        let ffn_per_layer = 2 * d_model * ffn_size; // up + down
        let layer_params = (attention_per_layer + ffn_per_layer) * n_layers;
        let lm_head = vocab_size * d_model;

        let total_params = embeddings + layer_params + lm_head;
        (total_params as f64 * bytes_per_param as f64) as u64
    }
}

impl std::fmt::Display for GPTConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPTConfig {{ arch: {}, layers: {}, d_model: {}, heads: {}, ctx: {}, vocab: {} }}",
            self.architecture,
            self.block_count,
            self.embedding_length,
            self.attention_head_count,
            self.context_length,
            self.vocab_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> GPTConfig {
        GPTConfig::new(
            "gpt2".to_string(),
            2048,  // context_length
            2048,  // embedding_length
            24,    // block_count
            16,    // attention_head_count
            8192,  // ffn_length
            50257, // vocab_size (GPT-2)
        )
    }

    #[test]
    fn test_gpt_config_creation() {
        let config = create_test_config();
        assert_eq!(config.architecture, "gpt2");
        assert_eq!(config.context_length, 2048);
        assert_eq!(config.embedding_length, 2048);
        assert_eq!(config.block_count, 24);
        assert_eq!(config.attention_head_count, 16);
        assert_eq!(config.ffn_length, 8192);
        assert_eq!(config.vocab_size, 50257);
    }

    #[test]
    fn test_head_dim_calculation() {
        let config = create_test_config();
        assert_eq!(config.head_dim(), 128); // 2048 / 16
    }

    #[test]
    fn test_validation_success() {
        let config = create_test_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_invalid_architecture() {
        let mut config = create_test_config();
        config.architecture = "llama".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_zero_context_length() {
        let mut config = create_test_config();
        config.context_length = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_non_divisible_head_dim() {
        let mut config = create_test_config();
        config.embedding_length = 2049; // Not divisible by 16
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_excessive_context_length() {
        let mut config = create_test_config();
        config.context_length = 2_000_000;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vram_estimation() {
        let config = create_test_config();
        let vram_fp16 = config.estimate_vram_bytes(2.0); // 2 bytes per param (FP16)
        let vram_q4 = config.estimate_vram_bytes(0.5); // 0.5 bytes per param (Q4)

        // FP16 should use ~4x more VRAM than Q4
        assert!(vram_fp16 > vram_q4 * 3);
        assert!(vram_fp16 < vram_q4 * 5);
    }

    #[test]
    fn test_display_format() {
        let config = create_test_config();
        let display = format!("{}", config);
        assert!(display.contains("gpt2"));
        assert!(display.contains("24")); // layers
        assert!(display.contains("2048")); // d_model
    }

    #[test]
    fn test_gpt_oss_20b_config() {
        // Approximate config for GPT-OSS-20B
        let config = GPTConfig::new(
            "gpt2".to_string(),
            8192,  // context_length
            6144,  // embedding_length
            44,    // block_count
            64,    // attention_head_count
            24576, // ffn_length
            50257, // vocab_size
        );

        assert!(config.validate().is_ok());
        assert_eq!(config.head_dim(), 96); // 6144 / 64

        // Estimate VRAM for MXFP4 (~0.5 bytes per param)
        let vram_mxfp4 = config.estimate_vram_bytes(0.5);
        assert!(vram_mxfp4 < 25_000_000_000); // Should fit in 24GB
    }
}

// ---
// Crafted by GPT-Gamma 🤖
