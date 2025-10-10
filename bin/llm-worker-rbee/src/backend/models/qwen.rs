//! Qwen model wrapper
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-017 (removed trait, using enum pattern)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{Config, ModelForCausalLM};
use std::path::Path;

/// Qwen model wrapper
///
/// TEAM-017: Wraps candle-transformers Qwen2 with its natural interface
#[derive(Debug)]
pub struct QwenModel {
    model: ModelForCausalLM,
    vocab_size: usize,
}

impl QwenModel {
    /// Load Qwen model from `SafeTensors`
    ///
    /// TEAM-017: Candle-idiomatic pattern
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let (parent, safetensor_files) = super::find_safetensors_files(path)?;

        // Parse config.json
        let config_path = parent.join("config.json");
        let config: Config = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("Failed to open config.json at {config_path:?}"))?,
        )
        .context("Failed to parse Qwen config.json")?;

        // Create VarBuilder and load model
        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)? };
        let model = ModelForCausalLM::new(&config, vb).context("Failed to load Qwen model")?;

        let vocab_size = config.vocab_size;

        tracing::info!(
            architecture = "qwen",
            hidden_size = config.hidden_size,
            num_layers = config.num_hidden_layers,
            vocab_size = vocab_size,
            "Loaded Qwen model"
        );

        Ok(Self { model, vocab_size })
    }

    /// Forward pass using Qwen's natural interface
    ///
    /// TEAM-017: Uses position parameter
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position).context("Qwen forward pass failed")
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        // Qwen typically uses token ID 151643 for EOS
        151643
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
