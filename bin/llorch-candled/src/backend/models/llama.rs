//! Llama model wrapper
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-017 (removed trait, using enum pattern)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaEosToks};
use std::path::Path;

/// Llama model wrapper
///
/// TEAM-017: Wraps candle-transformers Llama with its natural interface
#[derive(Debug)]
pub struct LlamaModel {
    model: Llama,
    cache: Cache,
    config: Config,
    vocab_size: usize,
}

impl LlamaModel {
    /// Load Llama model from SafeTensors
    ///
    /// TEAM-017: Candle-idiomatic pattern
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let (parent, safetensor_files) = super::find_safetensors_files(path)?;

        // Parse config.json
        let config_path = parent.join("config.json");
        let config_json: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("Failed to open config.json at {config_path:?}"))?,
        )?;

        let hidden_size =
            config_json["hidden_size"].as_u64().context("config.json missing hidden_size")?;
        let intermediate_size = config_json["intermediate_size"]
            .as_u64()
            .context("config.json missing intermediate_size")?;
        let num_hidden_layers = config_json["num_hidden_layers"]
            .as_u64()
            .context("config.json missing num_hidden_layers")?;
        let num_attention_heads = config_json["num_attention_heads"]
            .as_u64()
            .context("config.json missing num_attention_heads")?;
        let num_key_value_heads =
            config_json["num_key_value_heads"].as_u64().unwrap_or(num_attention_heads);
        let vocab_size =
            config_json["vocab_size"].as_u64().context("config.json missing vocab_size")?;
        let rms_norm_eps = config_json["rms_norm_eps"].as_f64().unwrap_or(1e-5);
        let rope_theta = config_json["rope_theta"].as_f64().unwrap_or(10000.0);
        let max_position_embeddings =
            config_json["max_position_embeddings"].as_u64().unwrap_or(2048);
        let bos_token_id = config_json["bos_token_id"].as_u64().unwrap_or(1);
        let eos_token_id = config_json["eos_token_id"].as_u64().unwrap_or(2);
        let tie_word_embeddings = config_json["tie_word_embeddings"].as_bool().unwrap_or(false);

        let config = Config {
            hidden_size: hidden_size as usize,
            intermediate_size: intermediate_size as usize,
            vocab_size: vocab_size as usize,
            num_hidden_layers: num_hidden_layers as usize,
            num_attention_heads: num_attention_heads as usize,
            num_key_value_heads: num_key_value_heads as usize,
            rms_norm_eps,
            rope_theta: rope_theta as f32,
            max_position_embeddings: max_position_embeddings as usize,
            bos_token_id: Some(bos_token_id as u32),
            eos_token_id: Some(LlamaEosToks::Single(eos_token_id as u32)),
            rope_scaling: None,
            tie_word_embeddings,
            use_flash_attn: false,
        };

        // Create VarBuilder and load model
        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)? };
        let model = Llama::load(vb, &config).context("Failed to load Llama model")?;

        // Create cache
        let cache = Cache::new(true, DType::F32, &config, device)
            .context("Failed to create Llama cache")?;

        tracing::info!(
            architecture = "llama",
            hidden_size = hidden_size,
            num_layers = num_hidden_layers,
            vocab_size = vocab_size,
            "Loaded Llama model"
        );

        Ok(Self { model, cache, config, vocab_size: vocab_size as usize })
    }

    /// Forward pass using Llama's natural interface
    ///
    /// TEAM-017: Uses position and mutable cache
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        tracing::debug!(
            position = position,
            input_shape = ?input_ids.dims(),
            input_device = ?input_ids.device(),
            "Llama forward pass starting"
        );
        
        let result = self.model
            .forward(input_ids, position, &mut self.cache)
            .map_err(|e| {
                tracing::error!(
                    error = %e,
                    position = position,
                    input_shape = ?input_ids.dims(),
                    "Llama forward pass failed with Candle error"
                );
                e
            })
            .context("Llama forward pass failed");
        
        if let Ok(ref logits) = result {
            tracing::debug!(
                output_shape = ?logits.dims(),
                output_device = ?logits.device(),
                "Llama forward pass completed"
            );
        }
        
        result
    }

    /// Get EOS token ID from config
    pub fn eos_token_id(&self) -> u32 {
        match self.config.eos_token_id {
            Some(LlamaEosToks::Single(id)) => id,
            Some(LlamaEosToks::Multiple(ref ids)) => ids.first().copied().unwrap_or(2),
            None => 2, // Default EOS token
        }
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
