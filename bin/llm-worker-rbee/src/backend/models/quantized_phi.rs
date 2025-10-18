// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Quantized Phi GGUF support

//! Quantized Phi model wrapper for GGUF files
//!
//! Created by: TEAM-090
//! Purpose: Load and run GGUF quantized Phi models

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_phi3::ModelWeights;
use observability_narration_core::{narrate, NarrationFields};
use std::path::Path;

/// Quantized Phi model wrapper for GGUF files
///
/// TEAM-090: Wraps candle-transformers `quantized_phi3` with GGUF support
pub struct QuantizedPhiModel {
    model: ModelWeights,
    eos_token_id: u32,
    vocab_size: usize,
}

impl QuantizedPhiModel {
    /// Load quantized Phi model from GGUF file
    ///
    /// TEAM-090: Loads GGUF files using candle's quantized model support
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        tracing::info!(path = ?path, "Loading GGUF Phi model");

        narrate(NarrationFields {
            actor: "model-loader",
            action: "gguf_load_start",
            target: path.display().to_string(),
            human: format!("Loading GGUF Phi model from {}", path.display()),
            cute: Some("Opening the GGUF Phi treasure chest! 📦🔍".to_string()),
            ..Default::default()
        });

        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file at {path:?}"))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to read GGUF content from {path:?}"))?;

        // Extract metadata
        let vocab_size = content
            .metadata
            .get("phi.vocab_size")
            .or_else(|| content.metadata.get("llama.vocab_size"))
            .and_then(|v| v.to_u32().ok())
            .or_else(|| {
                content.metadata.get("tokenizer.ggml.tokens").and_then(|v| match v {
                    candle_core::quantized::gguf_file::Value::Array(arr) => Some(arr.len() as u32),
                    _ => None,
                })
            })
            .with_context(|| "Cannot determine vocab_size from GGUF metadata")?
            as usize;

        let eos_token_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(32000);

        tracing::info!(
            vocab_size = vocab_size,
            eos_token_id = eos_token_id,
            tensors = content.tensor_infos.len(),
            "GGUF Phi metadata loaded"
        );

        let model = ModelWeights::from_gguf(false, content, &mut file, device)
            .with_context(|| "Failed to load Phi model weights from GGUF")?;

        narrate(NarrationFields {
            actor: "model-loader",
            action: "gguf_load_complete",
            target: path.display().to_string(),
            human: format!("GGUF Phi model loaded (vocab={}, eos={})", vocab_size, eos_token_id),
            cute: Some("GGUF Phi model loaded successfully! Ready to generate! 🎉✨".to_string()),
            ..Default::default()
        });

        Ok(Self { model, eos_token_id, vocab_size })
    }

    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position).map_err(|e| anyhow::anyhow!("{}", e))
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        tracing::debug!("Quantized Phi model cache will reset on next position=0 forward pass");
        Ok(())
    }
}
