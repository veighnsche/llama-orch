// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Quantized Qwen GGUF support

//! Quantized Qwen model wrapper for GGUF files
//!
//! Created by: TEAM-090
//! Purpose: Load and run GGUF quantized Qwen models

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use observability_narration_core::{narrate, NarrationFields};
use std::path::Path;

/// Quantized Qwen model wrapper for GGUF files
///
/// TEAM-090: Wraps candle-transformers `quantized_qwen2` with GGUF support
pub struct QuantizedQwenModel {
    model: ModelWeights,
    eos_token_id: u32,
    vocab_size: usize,
}

impl QuantizedQwenModel {
    /// Load quantized Qwen model from GGUF file
    ///
    /// TEAM-090: Loads GGUF files using candle's quantized model support
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        tracing::info!(path = ?path, "Loading GGUF Qwen model");

        narrate(NarrationFields {
            actor: "model-loader",
            action: "gguf_load_start",
            target: path.display().to_string(),
            human: format!("Loading GGUF Qwen model from {}", path.display()),
            cute: Some("Opening the GGUF Qwen treasure chest! 📦🔍".to_string()),
            ..Default::default()
        });

        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file at {path:?}"))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to read GGUF content from {path:?}"))?;

        // Extract metadata
        let vocab_size = content
            .metadata
            .get("qwen.vocab_size")
            .or_else(|| content.metadata.get("qwen2.vocab_size"))
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
            .unwrap_or(151643);

        tracing::info!(
            vocab_size = vocab_size,
            eos_token_id = eos_token_id,
            tensors = content.tensor_infos.len(),
            "GGUF Qwen metadata loaded"
        );

        let model = ModelWeights::from_gguf(content, &mut file, device)
            .with_context(|| "Failed to load Qwen model weights from GGUF")?;

        narrate(NarrationFields {
            actor: "model-loader",
            action: "gguf_load_complete",
            target: path.display().to_string(),
            human: format!("GGUF Qwen model loaded (vocab={}, eos={})", vocab_size, eos_token_id),
            cute: Some("GGUF Qwen model loaded successfully! Ready to generate! 🎉✨".to_string()),
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
        tracing::debug!("Quantized Qwen model cache will reset on next position=0 forward pass");
        Ok(())
    }
}
