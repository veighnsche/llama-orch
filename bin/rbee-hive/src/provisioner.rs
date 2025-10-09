//! Model provisioner - downloads models from HuggingFace
//!
//! Per test-001-mvp.md Phase 3: Model Provisioning
//! Downloads models and updates catalog
//!
//! Created by: TEAM-029

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{info, warn};

/// Model provisioner - handles model downloads
pub struct ModelProvisioner {
    base_dir: PathBuf,
}

/// Download progress information
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub stage: String,
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
    pub speed_mbps: f64,
}

impl ModelProvisioner {
    /// Create new model provisioner
    ///
    /// # Arguments
    /// * `base_dir` - Base directory for model storage
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    /// Check if model exists locally
    ///
    /// # Arguments
    /// * `reference` - Model reference (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    ///
    /// # Returns
    /// Local path if model exists
    pub fn find_local_model(&self, reference: &str) -> Option<PathBuf> {
        // TEAM-029: Simple heuristic - look for .gguf files in model directory
        let model_name = reference.split('/').last().unwrap_or(reference);
        let model_dir = self.base_dir.join(model_name.to_lowercase());

        if !model_dir.exists() {
            return None;
        }

        // Find first .gguf file
        std::fs::read_dir(&model_dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .find(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "gguf")
                    .unwrap_or(false)
            })
            .map(|entry| entry.path())
    }

    /// Download model from HuggingFace
    ///
    /// Per test-001-mvp.md Phase 3.1: Download Model
    ///
    /// # Arguments
    /// * `reference` - Model reference (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    /// * `provider` - Provider (currently only "hf" supported)
    ///
    /// # Returns
    /// Local path to downloaded model
    pub async fn download_model(&self, reference: &str, provider: &str) -> Result<PathBuf> {
        if provider != "hf" {
            anyhow::bail!("Only HuggingFace (hf) provider is currently supported");
        }

        info!("Downloading model from HuggingFace: {}", reference);

        // TEAM-029: Use llorch-models script for now
        // TODO: Implement native Rust download with hf_hub crate for better progress tracking
        let script_path = self.find_llorch_models_script()?;

        // Extract model name from reference
        let model_name = self.extract_model_name(reference)?;

        info!("Using llorch-models script to download: {}", model_name);

        // Run llorch-models download
        let output = Command::new(&script_path)
            .arg("download")
            .arg(&model_name)
            .env("LLORCH_MODEL_BASE_DIR", &self.base_dir)
            .output()
            .context("Failed to execute llorch-models script")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Model download failed: {}", stderr);
        }

        info!("Model download complete");

        // Find the downloaded model
        self.find_local_model(reference)
            .ok_or_else(|| anyhow::anyhow!("Model downloaded but not found in expected location"))
    }

    /// Find llorch-models script
    fn find_llorch_models_script(&self) -> Result<PathBuf> {
        // TEAM-029: Look for script in standard locations
        let candidates = vec![
            PathBuf::from("./scripts/llorch-models"),
            PathBuf::from("../scripts/llorch-models"),
            PathBuf::from("../../scripts/llorch-models"),
        ];

        for path in candidates {
            if path.exists() {
                return Ok(path);
            }
        }

        anyhow::bail!("llorch-models script not found. Please ensure scripts/llorch-models exists.")
    }

    /// Extract model name from reference
    ///
    /// Maps HuggingFace references to llorch-models names
    fn extract_model_name(&self, reference: &str) -> Result<String> {
        // TEAM-029: Map known references to llorch-models names
        let name = match reference {
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" => "tinyllama",
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF" => "qwen",
            "microsoft/Phi-3-mini-4k-instruct-gguf" => "phi3",
            "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF" => "llama3",
            "TheBloke/Llama-2-7B-GGUF" => "llama2",
            "TheBloke/Mistral-7B-Instruct-v0.2-GGUF" => "mistral",
            _ => {
                warn!("Unknown model reference: {}, using default tinyllama", reference);
                "tinyllama"
            }
        };

        Ok(name.to_string())
    }

    /// Get model size in bytes
    pub fn get_model_size(&self, path: &Path) -> Result<i64> {
        let metadata = std::fs::metadata(path)?;
        Ok(metadata.len() as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_model_name() {
        let provisioner = ModelProvisioner::new(PathBuf::from("/tmp"));

        assert_eq!(
            provisioner
                .extract_model_name("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
                .unwrap(),
            "tinyllama"
        );

        assert_eq!(
            provisioner
                .extract_model_name("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
                .unwrap(),
            "qwen"
        );
    }
}
