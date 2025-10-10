//! Model operations - delete, info, verify, disk usage
//!
//! Created by: TEAM-033
//! Based on: TEAM-032

use super::types::{ModelInfo, ModelProvisioner};
use anyhow::Result;
use std::path::Path;
use tracing::info;

impl ModelProvisioner {
    /// Get model size in bytes
    pub fn get_model_size(&self, path: &Path) -> Result<i64> {
        let metadata = std::fs::metadata(path)?;
        Ok(metadata.len() as i64)
    }

    /// Delete a model directory
    ///
    /// TEAM-032: Equivalent to `llorch-models delete`
    pub fn delete_model(&self, reference: &str) -> Result<()> {
        let model_name = reference.split('/').last().unwrap_or(reference);
        let model_dir = self.base_dir.join(model_name.to_lowercase());

        if !model_dir.exists() {
            anyhow::bail!("Model not found: {}", reference);
        }

        info!("Deleting model directory: {:?}", model_dir);
        std::fs::remove_dir_all(&model_dir)?;

        Ok(())
    }

    /// Get detailed model information
    ///
    /// TEAM-032: Equivalent to `llorch-models info`
    pub fn get_model_info(&self, reference: &str) -> Result<ModelInfo> {
        let model_name = reference.split('/').last().unwrap_or(reference);
        let model_dir = self.base_dir.join(model_name.to_lowercase());

        if !model_dir.exists() {
            anyhow::bail!("Model not found: {}", reference);
        }

        let mut total_size = 0i64;
        let mut file_count = 0usize;
        let mut gguf_files = Vec::new();

        for entry in std::fs::read_dir(&model_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                file_count += 1;
                let metadata = std::fs::metadata(&path)?;
                total_size += metadata.len() as i64;

                if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    gguf_files.push(path);
                }
            }
        }

        Ok(ModelInfo {
            reference: reference.to_string(),
            model_dir,
            total_size,
            file_count,
            gguf_files,
        })
    }

    /// Verify model integrity
    ///
    /// TEAM-032: Equivalent to `llorch-models verify`
    pub fn verify_model(&self, reference: &str) -> Result<()> {
        let info = self.get_model_info(reference)?;

        if info.gguf_files.is_empty() {
            anyhow::bail!("No .gguf files found in model directory");
        }

        for gguf_path in &info.gguf_files {
            let size = self.get_model_size(gguf_path)?;
            if size == 0 {
                anyhow::bail!("Empty .gguf file: {:?}", gguf_path);
            }
        }

        info!(
            "Model verified: {} .gguf files, {} total bytes",
            info.gguf_files.len(),
            info.total_size
        );

        Ok(())
    }

    /// Get total disk usage for all models
    ///
    /// TEAM-032: Equivalent to `llorch-models disk-usage`
    pub fn get_total_disk_usage(&self) -> Result<i64> {
        let mut total = 0i64;

        if !self.base_dir.exists() {
            return Ok(0);
        }

        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Ok(files) = std::fs::read_dir(&path) {
                    for file_entry in files.flatten() {
                        if let Ok(metadata) = file_entry.metadata() {
                            if metadata.is_file() {
                                total += metadata.len() as i64;
                            }
                        }
                    }
                }
            }
        }

        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_get_model_size() {
        let temp_dir = std::env::temp_dir().join("test_ops_size");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let test_file = temp_dir.join("test.gguf");
        fs::write(&test_file, b"test data content").unwrap();

        let provisioner = ModelProvisioner::new(temp_dir.clone());
        let size = provisioner.get_model_size(&test_file).unwrap();

        assert_eq!(size, 17);
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_delete_model() {
        let temp_dir = std::env::temp_dir().join("test_ops_delete");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let model_dir = temp_dir.join("testmodel");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.gguf"), b"test").unwrap();

        let provisioner = ModelProvisioner::new(temp_dir.clone());
        assert!(provisioner.delete_model("testmodel").is_ok());
        assert!(!model_dir.exists());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_get_model_info() {
        let temp_dir = std::env::temp_dir().join("test_ops_info");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let model_dir = temp_dir.join("testmodel");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.gguf"), b"test data").unwrap();

        let provisioner = ModelProvisioner::new(temp_dir.clone());
        let info = provisioner.get_model_info("testmodel").unwrap();

        assert_eq!(info.file_count, 1);
        assert_eq!(info.gguf_files.len(), 1);
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_verify_model() {
        let temp_dir = std::env::temp_dir().join("test_ops_verify");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let model_dir = temp_dir.join("testmodel");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.gguf"), b"test data").unwrap();

        let provisioner = ModelProvisioner::new(temp_dir.clone());
        assert!(provisioner.verify_model("testmodel").is_ok());
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_get_total_disk_usage() {
        let temp_dir = std::env::temp_dir().join("test_ops_disk");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let model_dir = temp_dir.join("model1");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("file.gguf"), b"12345").unwrap();

        let provisioner = ModelProvisioner::new(temp_dir.clone());
        let usage = provisioner.get_total_disk_usage().unwrap();
        assert_eq!(usage, 5);

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
