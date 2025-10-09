//! Phi model wrapper
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-017 (removed trait, using enum pattern)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::phi::{Config, Model};
use std::path::Path;

/// Phi model wrapper
///
/// TEAM-017: Wraps candle-transformers Phi with its natural interface
pub struct PhiModel {
    model: Model,
    vocab_size: usize,
}

impl PhiModel {
    /// Load Phi model from SafeTensors
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
        .context("Failed to parse Phi config.json")?;

        // Create VarBuilder and load model
        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)? };
        let model = Model::new(&config, vb).context("Failed to load Phi model")?;

        // TEAM-017: Extract vocab_size from JSON since fields are private
        let config_json: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("Failed to reopen config.json at {config_path:?}"))?,
        )?;

        let vocab_size = config_json["vocab_size"].as_u64().context("missing vocab_size")? as usize;
        let hidden_size =
            config_json["hidden_size"].as_u64().context("missing hidden_size")? as usize;
        let num_hidden_layers = config_json["num_hidden_layers"]
            .as_u64()
            .context("missing num_hidden_layers")? as usize;

        tracing::info!(
            architecture = "phi",
            hidden_size = hidden_size,
            num_layers = num_hidden_layers,
            vocab_size = vocab_size,
            "Loaded Phi model"
        );

        Ok(Self { model, vocab_size })
    }

    /// Forward pass using Phi's natural interface
    ///
    /// TEAM-017: Phi doesn't use position parameter, manages cache internally
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids).context("Phi forward pass failed")
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        // Phi typically uses token ID 50256 for EOS (GPT-2 tokenizer)
        50256
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
