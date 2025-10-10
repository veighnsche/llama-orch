//! Model factory - Auto-detect and load models
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-017 (switched to enum pattern for Candle idiomaticity)
//! Modified by: TEAM-036 (added GGUF support for quantized models)

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use serde_json::Value;
use std::path::Path;

pub mod llama;
pub mod mistral;
pub mod phi;
pub mod quantized_llama;
pub mod qwen;

/// Multi-model enum using Candle's idiomatic pattern
///
/// TEAM-017: Each variant wraps a specific model type with its natural interface
/// TEAM-036: Added QuantizedLlama for GGUF support
pub enum Model {
    Llama(llama::LlamaModel),
    Mistral(mistral::MistralModel),
    Phi(phi::PhiModel),
    Qwen(qwen::QwenModel),
    QuantizedLlama(quantized_llama::QuantizedLlamaModel),
}

impl Model {
    /// Forward pass - delegates to the specific model
    ///
    /// TEAM-017: Each model uses its natural interface
    /// TEAM-036: Added QuantizedLlama support
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        match self {
            Model::Llama(m) => m.forward(input_ids, position),
            Model::Mistral(m) => m.forward(input_ids, position),
            Model::Phi(m) => m.forward(input_ids), // Phi doesn't use position
            Model::Qwen(m) => m.forward(input_ids, position),
            Model::QuantizedLlama(m) => m.forward(input_ids, position),
        }
    }

    /// Get EOS token ID
    /// TEAM-036: Added QuantizedLlama support
    pub fn eos_token_id(&self) -> u32 {
        match self {
            Model::Llama(m) => m.eos_token_id(),
            Model::Mistral(m) => m.eos_token_id(),
            Model::Phi(m) => m.eos_token_id(),
            Model::Qwen(m) => m.eos_token_id(),
            Model::QuantizedLlama(m) => m.eos_token_id(),
        }
    }

    /// Get model architecture name
    /// TEAM-036: Added QuantizedLlama support
    pub fn architecture(&self) -> &str {
        match self {
            Model::Llama(_) => "llama",
            Model::Mistral(_) => "mistral",
            Model::Phi(_) => "phi",
            Model::Qwen(_) => "qwen",
            Model::QuantizedLlama(_) => "llama-quantized",
        }
    }

    /// Get vocab size
    /// TEAM-036: Added QuantizedLlama support
    pub fn vocab_size(&self) -> usize {
        match self {
            Model::Llama(m) => m.vocab_size(),
            Model::Mistral(m) => m.vocab_size(),
            Model::Phi(m) => m.vocab_size(),
            Model::Qwen(m) => m.vocab_size(),
            Model::QuantizedLlama(m) => m.vocab_size(),
        }
    }

    /// Reset KV cache to clear history
    ///
    /// TEAM-021: Required to clear warmup pollution before inference
    /// TEAM-036: Added QuantizedLlama support
    /// Not all models may support this (Phi manages cache internally)
    pub fn reset_cache(&mut self) -> Result<()> {
        match self {
            Model::Llama(m) => m.reset_cache(),
            Model::Mistral(_m) => {
                // TODO: Implement for Mistral when needed
                tracing::warn!("Cache reset not implemented for Mistral");
                Ok(())
            }
            Model::Phi(_m) => {
                // Phi manages cache internally, no reset needed
                tracing::debug!("Phi manages cache internally, no reset needed");
                Ok(())
            }
            Model::Qwen(_m) => {
                // TODO: Implement for Qwen when needed
                tracing::warn!("Cache reset not implemented for Qwen");
                Ok(())
            }
            Model::QuantizedLlama(m) => m.reset_cache(),
        }
    }
}

/// Detect model architecture from config.json
///
/// TEAM-017: Checks model_type and architectures fields
pub fn detect_architecture(config_json: &Value) -> Result<String> {
    // Check "model_type" field
    if let Some(model_type) = config_json.get("model_type").and_then(|v| v.as_str()) {
        return Ok(model_type.to_lowercase());
    }

    // Check "architectures" array
    if let Some(archs) = config_json.get("architectures").and_then(|v| v.as_array()) {
        if let Some(arch) = archs.first().and_then(|v| v.as_str()) {
            let arch_lower = arch.to_lowercase();
            // Normalize architecture names
            if arch_lower.contains("llama") {
                return Ok("llama".to_string());
            } else if arch_lower.contains("mistral") {
                return Ok("mistral".to_string());
            } else if arch_lower.contains("phi") {
                return Ok("phi".to_string());
            } else if arch_lower.contains("qwen") {
                return Ok("qwen".to_string());
            } else if arch_lower.contains("gemma") {
                return Ok("gemma".to_string());
            }
            return Ok(arch_lower);
        }
    }

    bail!("Could not detect model architecture from config.json");
}

/// Scan for safetensors files
///
/// TEAM-017: Candle-idiomatic helper to find safetensors files
pub(super) fn find_safetensors_files(
    path: &Path,
) -> Result<(std::path::PathBuf, Vec<std::path::PathBuf>)> {
    if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        Ok((parent.to_path_buf(), vec![path.to_path_buf()]))
    } else if path.is_dir() {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                files.push(entry_path);
            }
        }
        if files.is_empty() {
            bail!("No safetensors files found at {}", path.display());
        }
        Ok((path.to_path_buf(), files))
    } else {
        bail!("Path must be a .safetensors file or directory");
    }
}

/// Load config.json from model path
///
/// TEAM-017: Helper to load and parse config.json
fn load_config_json(model_path: &Path) -> Result<Value> {
    let parent = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or_else(|| Path::new("."))
    };

    let config_path = parent.join("config.json");
    let config_json: Value = serde_json::from_reader(
        std::fs::File::open(&config_path)
            .with_context(|| format!("Failed to open config.json at {config_path:?}"))?,
    )
    .context("Failed to parse config.json")?;

    Ok(config_json)
}

/// Load model based on detected architecture
///
/// TEAM-017: Factory function that returns Model enum (Candle-idiomatic pattern)
/// TEAM-036: Added GGUF support - detects .gguf files and loads quantized models
pub fn load_model(model_path: &str, device: &Device) -> Result<Model> {
    let path = Path::new(model_path);

    // TEAM-036: Check if this is a GGUF file (quantized model)
    if model_path.ends_with(".gguf") {
        tracing::info!(
            path = %model_path,
            "Detected GGUF file, loading quantized model"
        );
        let model = quantized_llama::QuantizedLlamaModel::load(path, device)?;
        return Ok(Model::QuantizedLlama(model));
    }

    // Otherwise, load from safetensors with config.json
    let config_json = load_config_json(path)?;
    let architecture = detect_architecture(&config_json)?;

    tracing::info!(
        architecture = %architecture,
        path = %model_path,
        "Detected model architecture"
    );

    match architecture.as_str() {
        "llama" => {
            let model = llama::LlamaModel::load(path, device)?;
            Ok(Model::Llama(model))
        }
        "mistral" => {
            let model = mistral::MistralModel::load(path, device)?;
            Ok(Model::Mistral(model))
        }
        "phi" => {
            let model = phi::PhiModel::load(path, device)?;
            Ok(Model::Phi(model))
        }
        "qwen" | "qwen2" => {
            let model = qwen::QwenModel::load(path, device)?;
            Ok(Model::Qwen(model))
        }
        _ => bail!("Unsupported model architecture: {}", architecture),
    }
}

/// Calculate model size in bytes from safetensors or GGUF files
///
/// TEAM-017: Helper to calculate total model size
/// TEAM-036: Added GGUF support
pub fn calculate_model_size(model_path: &str) -> Result<u64> {
    let path = Path::new(model_path);

    // TEAM-036: Handle GGUF files
    if model_path.ends_with(".gguf") {
        let metadata = std::fs::metadata(path)?;
        return Ok(metadata.len());
    }

    // Handle safetensors files
    let safetensor_files =
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            vec![path.to_path_buf()]
        } else if path.is_dir() {
            let mut files = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();
                if entry_path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    files.push(entry_path);
                }
            }
            files
        } else {
            bail!("Path must be a .safetensors, .gguf file or directory");
        };

    if safetensor_files.is_empty() {
        bail!("No safetensors files found at {}", path.display());
    }

    let model_size_bytes: u64 =
        safetensor_files.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum();

    Ok(model_size_bytes)
}
