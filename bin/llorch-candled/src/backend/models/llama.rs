//! Llama model wrapper
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-017 (removed trait, using enum pattern)
//! Modified by: TEAM-019 (fixed Metal/CUDA F16 dtype bug)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaEosToks};
use std::path::Path;

/// Llama model wrapper
///
/// TEAM-017: Wraps candle-transformers Llama with its natural interface
/// TEAM-021: Added device field to support cache reset
#[derive(Debug)]
pub struct LlamaModel {
    model: Llama,
    cache: Cache,
    config: Config,
    vocab_size: usize,
    device: Device,
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
        // TEAM-019: Use F32 for all backends (Metal F16 causes forward pass failures)
        let dtype = DType::F32;
        tracing::info!(dtype = ?dtype, device = ?device, "Loading model with dtype");
        
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)? };
        let model = Llama::load(vb, &config).context("Failed to load Llama model")?;

        // Create cache with same dtype
        let cache = Cache::new(true, dtype, &config, device)
            .context("Failed to create Llama cache")?;

        tracing::info!(
            architecture = "llama",
            hidden_size = hidden_size,
            num_layers = num_hidden_layers,
            vocab_size = vocab_size,
            "Loaded Llama model"
        );

        Ok(Self { 
            model, 
            cache, 
            config, 
            vocab_size: vocab_size as usize,
            device: device.clone(),
        })
    }

    /// Forward pass using Llama's natural interface
    ///
    /// TEAM-017: Uses position and mutable cache
    /// TEAM-018: Added detailed error logging
    /// TEAM-019: Fixed Metal/CUDA mask broadcasting bug (workaround)
    /// TEAM-020: Using Candle fork with proper mask fix - workaround removed
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        // Log input details
        tracing::info!(
            position = position,
            input_shape = ?input_ids.dims(),
            input_device = ?input_ids.device(),
            input_dtype = ?input_ids.dtype(),
            "Llama forward pass starting"
        );
        
        // TEAM-020: Workaround removed - using Candle fork with proper mask broadcasting fix
        // The mask now correctly handles KV cache growth by accounting for seqlen_offset
        // See: reference/candle branch llorch/metal-bugfixes
        
        // Attempt forward pass with detailed error capture
        match self.model.forward(input_ids, position, &mut self.cache) {
            Ok(logits) => {
                tracing::info!(
                    output_shape = ?logits.dims(),
                    output_device = ?logits.device(),
                    output_dtype = ?logits.dtype(),
                    "Llama forward pass completed successfully"
                );
                Ok(logits)
            }
            Err(e) => {
                // Log the full error chain
                tracing::error!(
                    error = %e,
                    error_debug = ?e,
                    position = position,
                    input_shape = ?input_ids.dims(),
                    input_device = ?input_ids.device(),
                    "Llama forward pass failed - Candle error details"
                );
                
                // Check for common issues
                if format!("{:?}", e).contains("shape") {
                    tracing::error!("Shape mismatch detected in forward pass");
                } else if format!("{:?}", e).contains("device") {
                    tracing::error!("Device mismatch detected in forward pass");
                } else if format!("{:?}", e).contains("dtype") {
                    tracing::error!("DType mismatch detected in forward pass");
                }
                
                Err(e).context("Llama forward pass failed")
            }
        }
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

    /// Reset cache to clear KV history between requests
    ///
    /// TEAM-021: Required to clear state between HTTP requests
    /// Candle's Cache doesn't expose clear(), so we recreate it
    /// Uses F32 dtype per TEAM-019 (Metal F16 causes issues)
    /// 
    /// 🎯 TEAM-021 Victory: Proper cache lifecycle management!
    pub fn reset_cache(&mut self) -> Result<()> {
        let dtype = DType::F32; // TEAM-019: F32 for all backends
        
        self.cache = Cache::new(true, dtype, &self.config, &self.device)
            .context("Failed to recreate cache")?;
        
        tracing::debug!("Cache reset complete - ready for new request");
        Ok(())
    }
}
