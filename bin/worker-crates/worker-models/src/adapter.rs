//! LlamaModelAdapter - Unified Model Interface
//!
//! Provides a consistent interface for all Llama-family models (Qwen, Phi-3, Llama 2/3, GPT-2/3).
//! Enables model-agnostic code and easy model switching.
//!
//! # Architecture
//! The adapter uses enum dispatch to route calls to model-specific implementations:
//! - `ModelType` enum identifies which model is loaded
//! - `Option<ModelImpl>` fields store model instances
//! - Unified methods route to correct implementation
//!
//! # Example
//!
//! ```no_run
//! use worker_models::{LlamaModelAdapter, AdapterForwardConfig, QwenConfig};
//!
//! // Load model (stub implementation for testing)
//! let config = QwenConfig::qwen2_5_0_5b();
//! // In production: let model = QwenWeightLoader::load_to_vram("model.gguf", &config)?;
//! // let adapter = LlamaModelAdapter::new_qwen(model);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Spec Refs
//! - M0-W-1214: Adapter pattern specification
//! - FT-071: Foundation team adapter implementation
//! - LT-033: Llama team adapter integration

use super::phi3::{Phi3Forward, Phi3ForwardConfig, Phi3Model};
use super::qwen::{ForwardPassConfig as QwenForwardConfig, QwenForward, QwenModel};
use thiserror::Error;

/// Model type enumeration
///
/// Identifies which model architecture is loaded in the adapter.
/// Used for dispatch to model-specific implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Qwen2_5,
    Phi3,
    Llama2,
    Llama3,
    GPT2,
    GPT3,
}

/// Adapter errors
///
/// Errors that can occur during adapter operations.
#[derive(Debug, Error)]
pub enum AdapterError {
    /// Model not loaded (internal error - should never happen in production)
    #[error("Model not loaded")]
    ModelNotLoaded,

    /// Invalid model type for this operation
    #[error("Invalid model type: {0:?}")]
    InvalidModelType(ModelType),

    /// Forward pass failed with model-specific error
    #[error("Forward pass failed: {0}")]
    ForwardPassFailed(String),

    /// Operation not supported for this model type
    #[error("Unsupported operation for model type: {0:?}")]
    UnsupportedOperation(ModelType),
}

/// Unified forward pass configuration
///
/// Configuration for model forward passes. Works with all model types.
///
/// # Fields
///
/// - `is_prefill`: True for prefill (process full prompt), false for decode (single token)
/// - `batch_size`: Number of sequences to process (usually 1)
/// - `seq_len`: Sequence length (prompt length for prefill, 1 for decode)
/// - `cache_len`: KV cache length (0 for prefill, grows during decode)
/// - `temperature`: Sampling temperature (0.0 = greedy, 1.0 = normal, >1.0 = creative)
/// - `seed`: Random seed for reproducibility
#[derive(Debug, Clone)]
pub struct AdapterForwardConfig {
    /// True for prefill, false for decode
    pub is_prefill: bool,
    /// Batch size (usually 1)
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

impl AdapterForwardConfig {
    /// Convert to Qwen-specific config
    fn to_qwen_config(&self) -> QwenForwardConfig {
        QwenForwardConfig {
            is_prefill: self.is_prefill,
            batch_size: self.batch_size,
            seq_len: self.seq_len,
            cache_len: self.cache_len,
            temperature: self.temperature,
            seed: self.seed,
        }
    }

    /// Convert to Phi-3-specific config
    fn to_phi3_config(&self) -> Phi3ForwardConfig {
        Phi3ForwardConfig {
            is_prefill: self.is_prefill,
            batch_size: self.batch_size,
            seq_len: self.seq_len,
            cache_len: self.cache_len,
            temperature: self.temperature,
            seed: self.seed,
        }
    }

    /// Convert to GPT-specific config
    fn to_gpt_config(&self) -> super::gpt::GPTForwardConfig {
        super::gpt::GPTForwardConfig {
            is_prefill: self.is_prefill,
            batch_size: self.batch_size,
            seq_len: self.seq_len,
            cache_len: self.cache_len,
            temperature: self.temperature,
            seed: self.seed,
        }
    }
}

/// Unified inference adapter for all Llama-family models
///
/// Provides consistent interface regardless of specific model variant.
/// Handles model-specific differences internally via enum dispatch.
///
/// # Design
///
/// - **Zero-cost abstraction**: Compiles to direct calls (no vtable overhead)
/// - **Type-safe**: Enum dispatch ensures correct model type at compile time
/// - **Fail-fast**: Returns errors immediately, doesn't hide issues
///
/// # Supported Models
///
/// - Qwen 2.5 (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B)
/// - Phi-3 (mini-4k, mini-128k, medium)
/// - Llama 2 (7B, 13B, 70B) - future
/// - Llama 3 (8B, 70B) - future
/// - GPT-2 / GPT-3 - future
///
/// # Thread Safety
///
/// `LlamaModelAdapter` is `Send` but not `Sync`. Each adapter instance
/// should be used from a single thread. Create multiple adapters for
/// concurrent inference.
pub struct LlamaModelAdapter {
    model_type: ModelType,
    qwen_model: Option<QwenModel>,
    phi3_model: Option<Phi3Model>,
    gpt_model: Option<super::gpt::GPTModel>,
}

impl LlamaModelAdapter {
    /// Create adapter for Qwen model
    pub fn new_qwen(model: QwenModel) -> Self {
        Self {
            model_type: ModelType::Qwen2_5,
            qwen_model: Some(model),
            phi3_model: None,
            gpt_model: None,
        }
    }

    /// Create adapter for Phi-3 model
    pub fn new_phi3(model: Phi3Model) -> Self {
        Self {
            model_type: ModelType::Phi3,
            qwen_model: None,
            phi3_model: Some(model),
            gpt_model: None,
        }
    }

    /// Create adapter for GPT-2 model
    pub fn new_gpt2(model: super::gpt::GPTModel) -> Self {
        Self {
            model_type: ModelType::GPT2,
            qwen_model: None,
            phi3_model: None,
            gpt_model: Some(model),
        }
    }

    /// Create adapter for GPT-3 model
    pub fn new_gpt3(model: super::gpt::GPTModel) -> Self {
        Self {
            model_type: ModelType::GPT3,
            qwen_model: None,
            phi3_model: None,
            gpt_model: Some(model),
        }
    }

    /// Get model type
    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> Result<usize, AdapterError> {
        match self.model_type {
            ModelType::Qwen2_5 => self
                .qwen_model
                .as_ref()
                .map(|m| m.config.vocab_size)
                .ok_or(AdapterError::ModelNotLoaded),
            ModelType::Phi3 => self
                .phi3_model
                .as_ref()
                .map(|m| m.config.vocab_size)
                .ok_or(AdapterError::ModelNotLoaded),
            ModelType::GPT2 | ModelType::GPT3 => self
                .gpt_model
                .as_ref()
                .map(|m| m.config.vocab_size)
                .ok_or(AdapterError::ModelNotLoaded),
            _ => Err(AdapterError::UnsupportedOperation(self.model_type)),
        }
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> Result<usize, AdapterError> {
        match self.model_type {
            ModelType::Qwen2_5 => self
                .qwen_model
                .as_ref()
                .map(|m| m.config.hidden_dim)
                .ok_or(AdapterError::ModelNotLoaded),
            ModelType::Phi3 => self
                .phi3_model
                .as_ref()
                .map(|m| m.config.hidden_dim)
                .ok_or(AdapterError::ModelNotLoaded),
            ModelType::GPT2 | ModelType::GPT3 => self
                .gpt_model
                .as_ref()
                .map(|m| m.config.hidden_dim)
                .ok_or(AdapterError::ModelNotLoaded),
            _ => Err(AdapterError::UnsupportedOperation(self.model_type)),
        }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> Result<usize, AdapterError> {
        match self.model_type {
            ModelType::Qwen2_5 => self
                .qwen_model
                .as_ref()
                .map(|m| m.config.num_layers)
                .ok_or(AdapterError::ModelNotLoaded),
            ModelType::Phi3 => self
                .phi3_model
                .as_ref()
                .map(|m| m.config.num_layers)
                .ok_or(AdapterError::ModelNotLoaded),
            ModelType::GPT2 | ModelType::GPT3 => self
                .gpt_model
                .as_ref()
                .map(|m| m.config.num_layers)
                .ok_or(AdapterError::ModelNotLoaded),
            _ => Err(AdapterError::UnsupportedOperation(self.model_type)),
        }
    }

    /// Get total VRAM usage
    pub fn vram_usage(&self) -> Result<usize, AdapterError> {
        match self.model_type {
            ModelType::Qwen2_5 => self
                .qwen_model
                .as_ref()
                .map(|m| m.total_vram_bytes)
                .ok_or(AdapterError::ModelNotLoaded),
            ModelType::Phi3 => self
                .phi3_model
                .as_ref()
                .map(|m| m.total_vram_bytes)
                .ok_or(AdapterError::ModelNotLoaded),
            ModelType::GPT2 | ModelType::GPT3 => self
                .gpt_model
                .as_ref()
                .map(|m| m.total_vram_bytes)
                .ok_or(AdapterError::ModelNotLoaded),
            _ => Err(AdapterError::UnsupportedOperation(self.model_type)),
        }
    }

    /// Prefill: process full prompt
    pub fn prefill(
        &self,
        input_ids: &[u32],
        config: &AdapterForwardConfig,
    ) -> Result<Vec<u32>, AdapterError> {
        match self.model_type {
            ModelType::Qwen2_5 => {
                let model = self.qwen_model.as_ref().ok_or(AdapterError::ModelNotLoaded)?;

                QwenForward::prefill(model, input_ids, &config.to_qwen_config())
                    .map_err(|e| AdapterError::ForwardPassFailed(e.to_string()))
            }
            ModelType::Phi3 => {
                let model = self.phi3_model.as_ref().ok_or(AdapterError::ModelNotLoaded)?;

                Phi3Forward::prefill(model, input_ids, &config.to_phi3_config())
                    .map_err(|e| AdapterError::ForwardPassFailed(e.to_string()))
            }
            ModelType::GPT2 | ModelType::GPT3 => {
                let model = self.gpt_model.as_ref().ok_or(AdapterError::ModelNotLoaded)?;

                super::gpt::GPTForward::prefill(model, input_ids, &config.to_gpt_config())
                    .map_err(|e| AdapterError::ForwardPassFailed(e.to_string()))
            }
            _ => Err(AdapterError::UnsupportedOperation(self.model_type)),
        }
    }

    /// Decode: generate single token
    pub fn decode(
        &self,
        input_id: u32,
        config: &AdapterForwardConfig,
    ) -> Result<u32, AdapterError> {
        match self.model_type {
            ModelType::Qwen2_5 => {
                let model = self.qwen_model.as_ref().ok_or(AdapterError::ModelNotLoaded)?;

                QwenForward::decode(model, input_id, &config.to_qwen_config())
                    .map_err(|e| AdapterError::ForwardPassFailed(e.to_string()))
            }
            ModelType::Phi3 => {
                let model = self.phi3_model.as_ref().ok_or(AdapterError::ModelNotLoaded)?;

                Phi3Forward::decode(model, input_id, &config.to_phi3_config())
                    .map_err(|e| AdapterError::ForwardPassFailed(e.to_string()))
            }
            ModelType::GPT2 | ModelType::GPT3 => {
                let model = self.gpt_model.as_ref().ok_or(AdapterError::ModelNotLoaded)?;

                super::gpt::GPTForward::decode(model, input_id, &config.to_gpt_config())
                    .map_err(|e| AdapterError::ForwardPassFailed(e.to_string()))
            }
            _ => Err(AdapterError::UnsupportedOperation(self.model_type)),
        }
    }

    /// Generate tokens autoregressively
    pub fn generate(
        &self,
        input_ids: &[u32],
        max_tokens: usize,
        config: &AdapterForwardConfig,
    ) -> Result<Vec<u32>, AdapterError> {
        match self.model_type {
            ModelType::Qwen2_5 => {
                let model = self.qwen_model.as_ref().ok_or(AdapterError::ModelNotLoaded)?;

                QwenForward::generate(model, input_ids, max_tokens, &config.to_qwen_config())
                    .map_err(|e| AdapterError::ForwardPassFailed(e.to_string()))
            }
            ModelType::Phi3 => {
                let model = self.phi3_model.as_ref().ok_or(AdapterError::ModelNotLoaded)?;

                Phi3Forward::generate(model, input_ids, max_tokens, &config.to_phi3_config())
                    .map_err(|e| AdapterError::ForwardPassFailed(e.to_string()))
            }
            ModelType::GPT2 | ModelType::GPT3 => {
                let model = self.gpt_model.as_ref().ok_or(AdapterError::ModelNotLoaded)?;

                super::gpt::GPTForward::generate(
                    model,
                    input_ids,
                    max_tokens,
                    &config.to_gpt_config(),
                )
                .map_err(|e| AdapterError::ForwardPassFailed(e.to_string()))
            }
            _ => Err(AdapterError::UnsupportedOperation(self.model_type)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt::{GPTConfig, GPTWeightLoader};
    use crate::phi3::{Phi3Config, Phi3WeightLoader};
    use crate::qwen::{QwenConfig, QwenWeightLoader};

    #[test]
    fn test_adapter_qwen() {
        let config = QwenConfig::qwen2_5_0_5b();
        let model = QwenWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();

        let adapter = LlamaModelAdapter::new_qwen(model);

        assert_eq!(adapter.model_type(), ModelType::Qwen2_5);
        assert_eq!(adapter.vocab_size().unwrap(), 151936);
        assert_eq!(adapter.hidden_dim().unwrap(), 896);
        assert_eq!(adapter.num_layers().unwrap(), 24);
    }

    #[test]
    fn test_adapter_phi3() {
        let config = Phi3Config::phi3_mini_4k();
        let model = Phi3WeightLoader::load_to_vram("dummy.gguf", &config).unwrap();

        let adapter = LlamaModelAdapter::new_phi3(model);

        assert_eq!(adapter.model_type(), ModelType::Phi3);
        assert_eq!(adapter.vocab_size().unwrap(), 32064);
        assert_eq!(adapter.hidden_dim().unwrap(), 3072);
        assert_eq!(adapter.num_layers().unwrap(), 32);
    }

    #[test]
    fn test_adapter_prefill_qwen() {
        let config = QwenConfig::qwen2_5_0_5b();
        let model = QwenWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
        let adapter = LlamaModelAdapter::new_qwen(model);

        let input_ids = vec![1, 2, 3];
        let fwd_config = AdapterForwardConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len: 3,
            cache_len: 0,
            temperature: 1.0,
            seed: 42,
        };

        let result = adapter.prefill(&input_ids, &fwd_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adapter_prefill_phi3() {
        let config = Phi3Config::phi3_mini_4k();
        let model = Phi3WeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
        let adapter = LlamaModelAdapter::new_phi3(model);

        let input_ids = vec![1, 2, 3];
        let fwd_config = AdapterForwardConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len: 3,
            cache_len: 0,
            temperature: 1.0,
            seed: 42,
        };

        let result = adapter.prefill(&input_ids, &fwd_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adapter_generate_qwen() {
        let config = QwenConfig::qwen2_5_0_5b();
        let model = QwenWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
        let adapter = LlamaModelAdapter::new_qwen(model);

        let input_ids = vec![1, 2, 3];
        let fwd_config = AdapterForwardConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len: 3,
            cache_len: 0,
            temperature: 1.0,
            seed: 42,
        };

        let result = adapter.generate(&input_ids, 5, &fwd_config);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), input_ids.len() + 5);
    }

    #[test]
    fn test_adapter_generate_phi3() {
        let config = Phi3Config::phi3_mini_4k();
        let model = Phi3WeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
        let adapter = LlamaModelAdapter::new_phi3(model);

        let input_ids = vec![1, 2, 3];
        let fwd_config = AdapterForwardConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len: 3,
            cache_len: 0,
            temperature: 1.0,
            seed: 42,
        };

        let result = adapter.generate(&input_ids, 5, &fwd_config);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), input_ids.len() + 5);
    }

    #[test]
    fn test_adapter_vram_usage() {
        let qwen_config = QwenConfig::qwen2_5_0_5b();
        let qwen_model = QwenWeightLoader::load_to_vram("dummy.gguf", &qwen_config).unwrap();
        let qwen_adapter = LlamaModelAdapter::new_qwen(qwen_model);

        let phi3_config = Phi3Config::phi3_mini_4k();
        let phi3_model = Phi3WeightLoader::load_to_vram("dummy.gguf", &phi3_config).unwrap();
        let phi3_adapter = LlamaModelAdapter::new_phi3(phi3_model);

        // Both should report VRAM usage
        assert!(qwen_adapter.vram_usage().unwrap() > 0);
        assert!(phi3_adapter.vram_usage().unwrap() > 0);

        // Phi-3 should use more VRAM (larger model)
        assert!(phi3_adapter.vram_usage().unwrap() > qwen_adapter.vram_usage().unwrap());
    }

    #[test]
    fn test_adapter_model_not_loaded() {
        let adapter = LlamaModelAdapter {
            model_type: ModelType::Qwen2_5,
            qwen_model: None,
            phi3_model: None,
            gpt_model: None,
        };

        let result = adapter.vocab_size();
        assert!(result.is_err());

        match result {
            Err(AdapterError::ModelNotLoaded) => {}
            _ => panic!("Expected ModelNotLoaded error"),
        }
    }

    #[test]
    fn test_adapter_gpt2() {
        let config = GPTConfig::gpt2_small();
        let model = GPTWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
        let adapter = LlamaModelAdapter::new_gpt2(model);

        assert_eq!(adapter.model_type(), ModelType::GPT2);
        assert_eq!(adapter.vocab_size().unwrap(), 50257);
        assert_eq!(adapter.hidden_dim().unwrap(), 768);
        assert_eq!(adapter.num_layers().unwrap(), 12);
    }

    #[test]
    fn test_adapter_gpt2_generation() {
        let config = GPTConfig::gpt2_small();
        let model = GPTWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
        let adapter = LlamaModelAdapter::new_gpt2(model);

        let input_ids = vec![1, 2, 3];
        let fwd_config = AdapterForwardConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len: 3,
            cache_len: 0,
            temperature: 1.0,
            seed: 42,
        };

        let result = adapter.generate(&input_ids, 10, &fwd_config);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.len(), input_ids.len() + 10);
    }
}
