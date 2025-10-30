// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Quantized Llama GGUF support

//! Quantized Llama model wrapper for GGUF files
//!
//! Created by: TEAM-036
//! Modified by: TEAM-088 (added comprehensive narration for debugging)
//! Purpose: Load and run GGUF quantized models (`Q4_K_M`, `Q5_K_M`, etc.)

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use observability_narration_core::n;
use std::path::Path;

/// Quantized Llama model wrapper for GGUF files
///
/// TEAM-036: Wraps candle-transformers `quantized_llama` with GGUF support
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
    /// TEAM-088: Added comprehensive narration for debugging
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        tracing::info!(path = ?path, "Loading GGUF model");

        // TEAM-088: Narrate GGUF loading start
        n!("gguf_load_start", "Loading GGUF model from {}", path.display());

        // Open GGUF file
        let mut file = std::fs::File::open(path).with_context(|| {
            // TEAM-088: Narrate file open failure
            n!("gguf_open_failed", "Failed to open GGUF file: {}", path.display());
            format!("Failed to open GGUF file at {path:?}")
        })?;

        // TEAM-088: Narrate file opened successfully
        n!("gguf_file_opened", "GGUF file opened, reading content");

        // Read GGUF content
        let content =
            candle_core::quantized::gguf_file::Content::read(&mut file).with_context(|| {
                // TEAM-088: Narrate GGUF parse failure
                n!("gguf_parse_failed", "Failed to parse GGUF content from {}", path.display());
                format!("Failed to read GGUF content from {path:?}")
            })?;

        // TEAM-088: Narrate metadata inspection
        n!("gguf_inspect_metadata", "Inspecting GGUF metadata ({} keys found)", content.metadata.len());

        // TEAM-088: List all available metadata keys for debugging
        let available_keys: Vec<String> =
            content.metadata.keys().map(std::string::ToString::to_string).collect();
        tracing::debug!(keys = ?available_keys, "Available GGUF metadata keys");

        // Extract metadata
        // TEAM-089: Make vocab_size optional - derive from tokenizer if missing
        let vocab_size = content
            .metadata
            .get("llama.vocab_size")
            .and_then(|v| v.to_u32().ok())
            .or_else(|| {
                // Fallback: count tokens in tokenizer array
                let derived_size = content.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(|v| match v {
                        candle_core::quantized::gguf_file::Value::Array(arr) => Some(arr.len() as u32),
                        _ => None,
                    });

                if let Some(size) = derived_size {
                    // TEAM-089: Narrate successful vocab_size derivation
                    n!("gguf_vocab_size_derived", "Derived vocab_size={} from tokenizer.ggml.tokens array", size);

                    tracing::info!(
                        vocab_size = size,
                        source = "tokenizer.ggml.tokens",
                        "Derived vocab_size from tokenizer array"
                    );
                }

                derived_size
            })
            .with_context(|| {
                // TEAM-088: Narrate missing vocab_size with helpful context
                let available_keys: Vec<String> = content.metadata.keys().map(std::string::ToString::to_string).collect();
                n!("gguf_metadata_missing", "Cannot determine vocab_size from GGUF metadata");

                tracing::error!(
                    required_key = "llama.vocab_size or tokenizer.ggml.tokens",
                    available_keys = ?available_keys,
                    "GGUF metadata missing required field"
                );

                format!(
                    "Cannot determine vocab_size: missing both llama.vocab_size and tokenizer.ggml.tokens. \
                     Available keys: [{}]. This GGUF file may be incomplete or corrupted. \
                     Try downloading a fresh copy from HuggingFace.",
                    available_keys.join(", ")
                )
            })?
            as usize;

        let eos_token_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(2); // Default EOS token for Llama

        // TEAM-088: Narrate successful metadata extraction
        n!("gguf_metadata_loaded", "GGUF metadata: vocab={}, eos={}, tensors={}", vocab_size, eos_token_id, content.tensor_infos.len());

        tracing::info!(
            vocab_size = vocab_size,
            eos_token_id = eos_token_id,
            tensors = content.tensor_infos.len(),
            "GGUF metadata loaded"
        );

        // TEAM-088: Narrate model weight loading
        n!("gguf_load_weights", "Loading {} tensors from GGUF", content.tensor_infos.len());

        // Load model weights from GGUF
        let model = ModelWeights::from_gguf(content, &mut file, device).with_context(|| {
            // TEAM-088: Narrate weight loading failure
            n!("gguf_weights_failed", "Failed to load model weights from GGUF");
            "Failed to load model weights from GGUF"
        })?;

        // TEAM-088: Narrate successful load
        n!("gguf_load_complete", "GGUF model loaded (vocab={}, eos={})", vocab_size, eos_token_id);

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
