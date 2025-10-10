//! Quantized Llama model wrapper for GGUF files
//!
//! Created by: TEAM-036
//! Purpose: Load and run GGUF quantized models (Q4_K_M, Q5_K_M, etc.)

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::path::Path;

/// Quantized Llama model wrapper for GGUF files
///
/// TEAM-036: Wraps candle-transformers quantized_llama with GGUF support
#[derive(Debug)]
pub struct QuantizedLlamaModel {
    model: ModelWeights,
    eos_token_id: u32,
    vocab_size: usize,
}

impl QuantizedLlamaModel {
    /// Load quantized Llama model from GGUF file
    ///
    /// TEAM-036: Loads GGUF files using candle's quantized model support
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        tracing::info!(path = ?path, "Loading GGUF model");

        // Open GGUF file
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file at {:?}", path))?;

        // Read GGUF content
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to read GGUF content from {:?}", path))?;

        // Extract metadata
        let vocab_size = content
            .metadata
            .get("llama.vocab_size")
            .and_then(|v| v.to_u32().ok())
            .context("Missing llama.vocab_size in GGUF metadata")?
            as usize;

        let eos_token_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(2); // Default EOS token for Llama

        tracing::info!(
            vocab_size = vocab_size,
            eos_token_id = eos_token_id,
            tensors = content.tensor_infos.len(),
            "GGUF metadata loaded"
        );

        // Load model weights from GGUF
        let model = ModelWeights::from_gguf(content, &mut file, device)
            .context("Failed to load model weights from GGUF")?;

        tracing::info!("GGUF model loaded successfully");

        Ok(Self { model, eos_token_id, vocab_size })
    }

    /// Forward pass through the model
    ///
    /// TEAM-036: Delegates to candle's quantized model
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position).map_err(|e| anyhow::anyhow!("{}", e))
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Reset KV cache to clear history
    ///
    /// TEAM-036: Quantized models manage cache internally per layer
    /// Cache is automatically cleared on position=0, so no explicit reset needed
    pub fn reset_cache(&mut self) -> Result<()> {
        // Quantized models in candle reset cache automatically when position=0
        // The kv_cache in each layer is set to None when index_pos == 0
        tracing::debug!("Quantized model cache will reset on next position=0 forward pass");
        Ok(())
    }
}
