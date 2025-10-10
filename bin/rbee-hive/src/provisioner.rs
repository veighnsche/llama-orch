//! Model provisioner - downloads models from HuggingFace
//!
//! Per test-001-mvp.md Phase 3: Model Provisioning
//! Downloads models using llorch-models script
//!
//! TEAM-030: Simplified to filesystem-based cache (no SQLite)
//! Source of truth is the filesystem - just scans for .gguf files
//!
//! Created by: TEAM-029
//! Modified by: TEAM-030

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

    /// List all available models (TEAM-030: Filesystem-based catalog)
    ///
    /// Scans base directory for .gguf files
    ///
    /// # Returns
    /// Vector of (model_name, path) tuples
    pub fn list_models(&self) -> Result<Vec<(String, PathBuf)>> {
        let mut models = Vec::new();

        if !self.base_dir.exists() {
            return Ok(models);
        }

        // Scan subdirectories for .gguf files
        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let model_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Find .gguf files in this directory
                if let Ok(files) = std::fs::read_dir(&path) {
                    for file_entry in files.flatten() {
                        let file_path = file_entry.path();
                        if file_path
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .map(|ext| ext == "gguf")
                            .unwrap_or(false)
                        {
                            models.push((model_name.clone(), file_path));
                        }
                    }
                }
            }
        }

        Ok(models)
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

    // TEAM-031: Additional comprehensive tests
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
    fn test_provisioner_creation() {
        let base_dir = PathBuf::from("/tmp/models");
        let provisioner = ModelProvisioner::new(base_dir.clone());
        assert_eq!(provisioner.base_dir, base_dir);
    }

    #[test]
    fn test_find_local_model_nonexistent() {
        let temp_dir = std::env::temp_dir().join("test_provisioner_nonexistent");
        let provisioner = ModelProvisioner::new(temp_dir);
        
        let result = provisioner.find_local_model("nonexistent/model");
        assert!(result.is_none());
    }

    #[test]
    fn test_list_models_empty_dir() {
        let temp_dir = std::env::temp_dir().join("test_provisioner_empty");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let provisioner = ModelProvisioner::new(temp_dir);
        
        let models = provisioner.list_models().unwrap();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_list_models_with_files() {
        use std::fs;
        
        let temp_dir = std::env::temp_dir().join("test_provisioner_list");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        
        // Create model directory with .gguf file
        let model_dir = temp_dir.join("testmodel");
        fs::create_dir_all(&model_dir).unwrap();
        let model_file = model_dir.join("model.gguf");
        fs::write(&model_file, b"test").unwrap();
        
        let provisioner = ModelProvisioner::new(temp_dir.clone());
        let models = provisioner.list_models().unwrap();
        
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].0, "testmodel");
        assert!(models[0].1.ends_with("model.gguf"));
        
        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_get_model_size() {
        use std::fs;
        
        let temp_dir = std::env::temp_dir().join("test_provisioner_size");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        
        let test_file = temp_dir.join("test.gguf");
        let test_data = b"test data content";
        fs::write(&test_file, test_data).unwrap();
        
        let provisioner = ModelProvisioner::new(temp_dir.clone());
        let size = provisioner.get_model_size(&test_file).unwrap();
        
        assert_eq!(size, test_data.len() as i64);
        
        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_get_model_size_nonexistent() {
        let provisioner = ModelProvisioner::new(PathBuf::from("/tmp"));
        let result = provisioner.get_model_size(&PathBuf::from("/nonexistent/file.gguf"));
        assert!(result.is_err());
    }

    #[test]
    fn test_find_local_model_with_file() {
        use std::fs;
        
        let temp_dir = std::env::temp_dir().join("test_provisioner_find");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        
        // Create model directory with .gguf file
        let model_dir = temp_dir.join("tinyllama-1.1b-chat-v1.0-gguf");
        fs::create_dir_all(&model_dir).unwrap();
        let model_file = model_dir.join("model.gguf");
        fs::write(&model_file, b"test").unwrap();
        
        let provisioner = ModelProvisioner::new(temp_dir.clone());
        let found = provisioner.find_local_model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
        
        assert!(found.is_some());
        assert!(found.unwrap().ends_with("model.gguf"));
        
        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
