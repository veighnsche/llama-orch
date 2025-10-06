// GPT Inference Adapter
//
// Implements GPTModelAdapter for GPT-family models (GPT-2, GPT-OSS-20B).
// Handles architecture-specific inference logic.
//
// Spec: M0-W-1215
// Story: GT-039

use crate::cuda_ffi;
use crate::model::GPTConfig;
use std::sync::Arc;

/// GPT inference adapter
///
/// Manages inference for GPT-family models with architecture-specific logic:
/// - LayerNorm (not RMSNorm)
/// - GELU activation (not SwiGLU)
/// - Absolute positional embeddings (not RoPE)
/// - Multi-Head Attention (not GQA)
pub struct GPTModelAdapter {
    config: GPTConfig,
    // CUDA resources would be managed here
}

impl GPTModelAdapter {
    /// Create new GPT inference adapter
    pub fn new(config: GPTConfig) -> Result<Self, String> {
        config.validate()?;

        Ok(Self { config })
    }

    /// Get model configuration
    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    /// Estimate VRAM usage for this model
    pub fn estimate_vram_bytes(&self, quantization: QuantizationType) -> u64 {
        let bytes_per_param = match quantization {
            QuantizationType::FP16 => 2.0,
            QuantizationType::Q4_K_M => 0.5,
            QuantizationType::MXFP4 => 0.53, // 17 bytes per 32 elements
        };

        self.config.estimate_vram_bytes(bytes_per_param)
    }

    /// Validate model can fit in available VRAM
    pub fn validate_vram(
        &self,
        available_vram: u64,
        quantization: QuantizationType,
    ) -> Result<(), String> {
        let required = self.estimate_vram_bytes(quantization);

        if required > available_vram {
            return Err(format!(
                "Model requires {} GB VRAM but only {} GB available",
                required / (1024 * 1024 * 1024),
                available_vram / (1024 * 1024 * 1024)
            ));
        }

        Ok(())
    }
}

/// Quantization type for GPT models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// FP16 (2 bytes per parameter)
    FP16,
    /// Q4_K_M quantization (0.5 bytes per parameter)
    Q4_K_M,
    /// MXFP4 quantization (0.53 bytes per parameter)
    MXFP4,
}

impl QuantizationType {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            QuantizationType::FP16 => "FP16",
            QuantizationType::Q4_K_M => "Q4_K_M",
            QuantizationType::MXFP4 => "MXFP4",
        }
    }

    /// Get compression ratio vs FP16
    pub fn compression_ratio(&self) -> f32 {
        match self {
            QuantizationType::FP16 => 1.0,
            QuantizationType::Q4_K_M => 4.0,
            QuantizationType::MXFP4 => 3.76,
        }
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
            50257, // vocab_size
        )
    }

    #[test]
    fn test_adapter_creation() {
        let config = create_test_config();
        let adapter = GPTModelAdapter::new(config);
        assert!(adapter.is_ok());
    }

    #[test]
    fn test_vram_estimation() {
        let config = create_test_config();
        let adapter = GPTModelAdapter::new(config).unwrap();

        let vram_fp16 = adapter.estimate_vram_bytes(QuantizationType::FP16);
        let vram_q4 = adapter.estimate_vram_bytes(QuantizationType::Q4_K_M);
        let vram_mxfp4 = adapter.estimate_vram_bytes(QuantizationType::MXFP4);

        // FP16 should use most VRAM
        assert!(vram_fp16 > vram_mxfp4);
        assert!(vram_mxfp4 > vram_q4);
    }

    #[test]
    fn test_vram_validation() {
        let config = create_test_config();
        let adapter = GPTModelAdapter::new(config).unwrap();

        // Should fit in 10GB
        assert!(adapter.validate_vram(10 * 1024 * 1024 * 1024, QuantizationType::Q4_K_M).is_ok());

        // Should not fit in 100MB
        assert!(adapter.validate_vram(100 * 1024 * 1024, QuantizationType::FP16).is_err());
    }

    #[test]
    fn test_quantization_types() {
        assert_eq!(QuantizationType::FP16.name(), "FP16");
        assert_eq!(QuantizationType::Q4_K_M.name(), "Q4_K_M");
        assert_eq!(QuantizationType::MXFP4.name(), "MXFP4");

        assert_eq!(QuantizationType::FP16.compression_ratio(), 1.0);
        assert_eq!(QuantizationType::Q4_K_M.compression_ratio(), 4.0);
        assert_eq!(QuantizationType::MXFP4.compression_ratio(), 3.76);
    }

    #[test]
    fn test_gpt_oss_20b() {
        let config = GPTConfig::new(
            "gpt2".to_string(),
            8192,  // context_length
            6144,  // embedding_length
            44,    // block_count
            64,    // attention_head_count
            24576, // ffn_length
            50257, // vocab_size
        );

        let adapter = GPTModelAdapter::new(config).unwrap();

        // Should fit in 24GB with MXFP4
        let vram_24gb = 24 * 1024 * 1024 * 1024u64;
        assert!(adapter.validate_vram(vram_24gb, QuantizationType::MXFP4).is_ok());
    }
}

// ---
// Crafted by GPT-Gamma 🤖
