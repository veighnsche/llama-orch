//! Model catalog operations - listing and finding models
//!
//! Created by: TEAM-033
//! Based on: TEAM-029, TEAM-030

use super::types::ModelProvisioner;
use anyhow::Result;
use std::path::PathBuf;

impl ModelProvisioner {
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
    use std::fs;

    #[test]
    fn test_find_local_model_nonexistent() {
        let temp_dir = std::env::temp_dir().join("test_catalog_nonexistent");
        let provisioner = ModelProvisioner::new(temp_dir);
        
        let result = provisioner.find_local_model("nonexistent/model");
        assert!(result.is_none());
    }

    #[test]
    fn test_find_local_model_with_file() {
        let temp_dir = std::env::temp_dir().join("test_catalog_find");
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

    #[test]
    fn test_list_models_empty_dir() {
        let temp_dir = std::env::temp_dir().join("test_catalog_empty");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let provisioner = ModelProvisioner::new(temp_dir);
        
        let models = provisioner.list_models().unwrap();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_list_models_with_files() {
        let temp_dir = std::env::temp_dir().join("test_catalog_list");
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
    fn test_list_models_multiple_gguf_files() {
        let temp_dir = std::env::temp_dir().join("test_catalog_multiple");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        
        // Create model directory with multiple .gguf files
        let model_dir = temp_dir.join("multimodel");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model1.gguf"), b"test1").unwrap();
        fs::write(model_dir.join("model2.gguf"), b"test2").unwrap();
        
        let provisioner = ModelProvisioner::new(temp_dir.clone());
        let models = provisioner.list_models().unwrap();
        
        assert_eq!(models.len(), 2);
        
        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_list_models_ignores_non_gguf() {
        let temp_dir = std::env::temp_dir().join("test_catalog_ignore");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        
        let model_dir = temp_dir.join("testmodel");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.gguf"), b"test").unwrap();
        fs::write(model_dir.join("config.json"), b"{}").unwrap();
        fs::write(model_dir.join("readme.txt"), b"readme").unwrap();
        
        let provisioner = ModelProvisioner::new(temp_dir.clone());
        let models = provisioner.list_models().unwrap();
        
        // Should only find the .gguf file
        assert_eq!(models.len(), 1);
        assert!(models[0].1.to_str().unwrap().ends_with(".gguf"));
        
        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
