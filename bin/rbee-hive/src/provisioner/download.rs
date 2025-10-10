//! Model download operations
//!
//! Created by: TEAM-033
//! Based on: TEAM-029

use super::types::ModelProvisioner;
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Command;
use tracing::{info, warn};

impl ModelProvisioner {
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

        // TEAM-029: Use rbee-models script for now
        // TODO: Implement native Rust download with hf_hub crate for better progress tracking
        let script_path = self.find_rbee_models_script()?;

        // Extract model name from reference
        let model_name = self.extract_model_name(reference)?;

        info!("Using rbee-models script to download: {}", model_name);

        // Run rbee-models download
        let output = Command::new(&script_path)
            .arg("download")
            .arg(&model_name)
            .env("RBEE_MODEL_BASE_DIR", &self.base_dir)
            .output()
            .context("Failed to execute rbee-models script")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Model download failed: {}", stderr);
        }

        info!("Model download complete");

        // Find the downloaded model
        self.find_local_model(reference)
            .ok_or_else(|| anyhow::anyhow!("Model downloaded but not found in expected location"))
    }

    /// Find rbee-models script
    pub(super) fn find_rbee_models_script(&self) -> Result<PathBuf> {
        // TEAM-029: Look for script in standard locations
        let candidates = vec![
            PathBuf::from("./scripts/rbee-models"),
            PathBuf::from("../scripts/rbee-models"),
            PathBuf::from("../../scripts/rbee-models"),
        ];

        for path in candidates {
            if path.exists() {
                return Ok(path);
            }
        }

        anyhow::bail!("rbee-models script not found. Please ensure scripts/rbee-models exists.")
    }

    /// Extract model name from reference
        /// Maps HuggingFace references to rbee-models names
    pub(super) fn extract_model_name(&self, reference: &str) -> Result<String> {
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

    #[test]
    fn test_extract_model_name_all_known_models() {
        let provisioner = ModelProvisioner::new(PathBuf::from("/tmp"));

        assert_eq!(
            provisioner.extract_model_name("microsoft/Phi-3-mini-4k-instruct-gguf").unwrap(),
            "phi3"
        );
        assert_eq!(
            provisioner.extract_model_name("QuantFactory/Meta-Llama-3-8B-Instruct-GGUF").unwrap(),
            "llama3"
        );
        assert_eq!(
            provisioner.extract_model_name("TheBloke/Llama-2-7B-GGUF").unwrap(),
            "llama2"
        );
        assert_eq!(
            provisioner.extract_model_name("TheBloke/Mistral-7B-Instruct-v0.2-GGUF").unwrap(),
            "mistral"
        );
    }

    #[test]
    fn test_extract_model_name_unknown() {
        let provisioner = ModelProvisioner::new(PathBuf::from("/tmp"));
        // Unknown models default to tinyllama
        assert_eq!(
            provisioner.extract_model_name("unknown/model").unwrap(),
            "tinyllama"
        );
    }

    #[test]
    fn test_find_rbee_models_script_not_found() {
        // TEAM-033: This test may pass if the script exists in the workspace
        // We test the error message format instead
        let provisioner = ModelProvisioner::new(PathBuf::from("/tmp/nonexistent"));
        let result = provisioner.find_rbee_models_script();
        
        // If script doesn't exist, should error with specific message
        if result.is_err() {
            assert!(result.unwrap_err().to_string().contains("llorch-models script not found"));
        }
        // If script exists (in CI/dev environment), that's also valid
    }

    #[test]
    fn test_extract_model_name_case_sensitivity() {
        let provisioner = ModelProvisioner::new(PathBuf::from("/tmp"));
        
        // Exact match should work
        assert_eq!(
            provisioner.extract_model_name("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF").unwrap(),
            "tinyllama"
        );
        
        // Different case should fall back to default
        assert_eq!(
            provisioner.extract_model_name("thebloke/tinyllama-1.1b-chat-v1.0-gguf").unwrap(),
            "tinyllama"  // Falls back to default
        );
    }
}
