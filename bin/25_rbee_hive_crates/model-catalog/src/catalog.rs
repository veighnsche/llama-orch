// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-267: Implemented filesystem-based model catalog

use crate::types::{ModelEntry, ModelStatus};
use anyhow::{anyhow, Result};
use std::fs;
use std::path::PathBuf;

/// Filesystem-based model catalog
///
/// Models are stored in platform-specific cache directory:
/// - Linux/Mac: ~/.cache/rbee/models/
/// - Windows: %LOCALAPPDATA%\rbee\models\
///
/// Each model has:
/// - Directory: {cache}/rbee/models/{model-id}/
/// - Metadata: {model-id}/metadata.yaml
#[derive(Clone)]
pub struct ModelCatalog {
    models_dir: PathBuf,
}

impl ModelCatalog {
    /// Create a new catalog pointing to the models directory
    pub fn new() -> Result<Self> {
        let models_dir = Self::get_models_dir()?;
        fs::create_dir_all(&models_dir)?;

        Ok(Self { models_dir })
    }

    /// Get the platform-specific models directory
    fn get_models_dir() -> Result<PathBuf> {
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow!("Cannot determine cache directory"))?;

        Ok(cache_dir.join("rbee").join("models"))
    }

    /// Get path for a specific model
    ///
    /// TEAM-269: Made public for provisioner access
    pub fn model_path(&self, id: &str) -> PathBuf {
        self.models_dir.join(id)
    }

    /// Get metadata file path for a model
    fn metadata_path(&self, id: &str) -> PathBuf {
        self.model_path(id).join("metadata.yaml")
    }

    /// Read metadata from YAML file
    fn read_metadata(&self, id: &str) -> Result<ModelEntry> {
        let metadata_path = self.metadata_path(id);

        if !metadata_path.exists() {
            return Err(anyhow!("Metadata file not found for model '{}'", id));
        }

        let content = fs::read_to_string(&metadata_path)?;
        let entry: ModelEntry = serde_yaml::from_str(&content)?;
        Ok(entry)
    }

    /// Write metadata to YAML file
    fn write_metadata(&self, entry: &ModelEntry) -> Result<()> {
        let metadata_path = self.metadata_path(&entry.id);
        let model_dir = self.model_path(&entry.id);

        fs::create_dir_all(&model_dir)?;

        let content = serde_yaml::to_string(entry)?;
        fs::write(&metadata_path, content)?;
        Ok(())
    }

    /// Add a model to the catalog (creates directory and metadata file)
    pub fn add(&self, model: ModelEntry) -> Result<()> {
        let model_dir = self.model_path(&model.id);

        if model_dir.exists() {
            return Err(anyhow!("Model '{}' already exists in catalog", model.id));
        }

        self.write_metadata(&model)?;
        Ok(())
    }

    /// Get a model by ID (reads from filesystem)
    pub fn get(&self, id: &str) -> Result<ModelEntry> {
        self.read_metadata(id)
    }

    /// Remove a model from the catalog (deletes directory)
    pub fn remove(&self, id: &str) -> Result<ModelEntry> {
        let entry = self.read_metadata(id)?;
        let model_dir = self.model_path(id);

        if model_dir.exists() {
            fs::remove_dir_all(&model_dir)?;
        }

        Ok(entry)
    }

    /// List all models (scans filesystem)
    pub fn list(&self) -> Vec<ModelEntry> {
        let Ok(entries) = fs::read_dir(&self.models_dir) else {
            return Vec::new();
        };

        entries
            .flatten()
            .filter_map(|entry| {
                // Only process directories
                let file_type = entry.file_type().ok()?;
                if !file_type.is_dir() {
                    return None;
                }

                // Get model ID from directory name
                let model_id = entry.file_name().to_str()?.to_string();

                // Read and return metadata (skip if invalid)
                self.read_metadata(&model_id).ok()
            })
            .collect()
    }

    /// List models by status
    pub fn list_by_status(&self, status_filter: fn(&ModelStatus) -> bool) -> Vec<ModelEntry> {
        self.list()
            .into_iter()
            .filter(|m| status_filter(&m.status))
            .collect()
    }

    /// Update model status (rewrites metadata file)
    pub fn update_status(&self, id: &str, status: ModelStatus) -> Result<()> {
        let mut entry = self.read_metadata(id)?;
        entry.status = status;
        self.write_metadata(&entry)?;
        Ok(())
    }

    /// Check if model exists (checks filesystem)
    pub fn contains(&self, id: &str) -> bool {
        self.model_path(id).exists()
    }

    /// Get catalog size (number of models)
    pub fn len(&self) -> usize {
        self.list().len()
    }

    /// Check if catalog is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::new().expect("Failed to create default ModelCatalog")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // Helper to create a test catalog in a temporary directory
    fn create_test_catalog() -> (ModelCatalog, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let catalog = ModelCatalog {
            models_dir: temp_dir.path().to_path_buf(),
        };
        (catalog, temp_dir)
    }

    #[test]
    fn test_catalog_add_get() {
        let (catalog, _temp) = create_test_catalog();
        let model = ModelEntry::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            PathBuf::from("/tmp/models/test"),
            1024 * 1024 * 100, // 100 MB
        );

        catalog.add(model.clone()).unwrap();
        let retrieved = catalog.get("test-model").unwrap();

        assert_eq!(retrieved.id, "test-model");
        assert_eq!(retrieved.name, "Test Model");
        assert_eq!(retrieved.size_bytes, 1024 * 1024 * 100);
    }

    #[test]
    fn test_catalog_duplicate_add() {
        let (catalog, _temp) = create_test_catalog();
        let model = ModelEntry::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            PathBuf::from("/tmp/models/test"),
            1024,
        );

        catalog.add(model.clone()).unwrap();
        let result = catalog.add(model);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already exists"));
    }

    #[test]
    fn test_catalog_remove() {
        let (catalog, _temp) = create_test_catalog();
        let model = ModelEntry::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            PathBuf::from("/tmp/models/test"),
            1024,
        );

        catalog.add(model).unwrap();
        assert_eq!(catalog.len(), 1);

        let removed = catalog.remove("test-model").unwrap();
        assert_eq!(removed.id, "test-model");
        assert_eq!(catalog.len(), 0);
    }

    #[test]
    fn test_catalog_list() {
        let (catalog, _temp) = create_test_catalog();

        for i in 0..3 {
            let model = ModelEntry::new(
                format!("model-{}", i),
                format!("Model {}", i),
                PathBuf::from(format!("/tmp/models/model-{}", i)),
                1024,
            );
            catalog.add(model).unwrap();
        }

        let models = catalog.list();
        assert_eq!(models.len(), 3);
    }

    #[test]
    fn test_catalog_update_status() {
        let (catalog, _temp) = create_test_catalog();
        let model = ModelEntry::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            PathBuf::from("/tmp/models/test"),
            1024,
        );

        catalog.add(model).unwrap();

        catalog
            .update_status("test-model", ModelStatus::Downloading { progress: 0.5 })
            .unwrap();

        let updated = catalog.get("test-model").unwrap();
        assert!(matches!(
            updated.status,
            ModelStatus::Downloading { .. }
        ));
    }

    #[test]
    fn test_catalog_contains() {
        let (catalog, _temp) = create_test_catalog();
        let model = ModelEntry::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            PathBuf::from("/tmp/models/test"),
            1024,
        );

        assert!(!catalog.contains("test-model"));
        catalog.add(model).unwrap();
        assert!(catalog.contains("test-model"));
    }

    #[test]
    fn test_catalog_list_by_status() {
        let (catalog, _temp) = create_test_catalog();

        // Add ready model
        let model1 = ModelEntry::new(
            "model-1".to_string(),
            "Model 1".to_string(),
            PathBuf::from("/tmp/models/model-1"),
            1024,
        );
        catalog.add(model1).unwrap();

        // Add downloading model
        let mut model2 = ModelEntry::new(
            "model-2".to_string(),
            "Model 2".to_string(),
            PathBuf::from("/tmp/models/model-2"),
            2048,
        );
        model2.status = ModelStatus::Downloading { progress: 0.5 };
        catalog.add(model2).unwrap();

        let ready_models = catalog.list_by_status(|s| matches!(s, ModelStatus::Ready));
        assert_eq!(ready_models.len(), 1);
        assert_eq!(ready_models[0].id, "model-1");
    }

    #[test]
    fn test_model_entry_is_ready() {
        let model = ModelEntry::new(
            "test".to_string(),
            "Test".to_string(),
            PathBuf::from("/tmp"),
            1024,
        );
        assert!(model.is_ready());

        let mut downloading = model.clone();
        downloading.status = ModelStatus::Downloading { progress: 0.5 };
        assert!(!downloading.is_ready());
    }
}
