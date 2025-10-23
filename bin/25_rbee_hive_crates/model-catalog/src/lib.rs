// TEAM-267: Model catalog using artifact-catalog abstraction
#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-hive-model-catalog
//!
//! Model catalog for managing LLM model files.
//! Built on top of artifact-catalog for consistency.

mod types;

pub use types::{ModelEntry, ModelStatus};

use anyhow::Result;
use rbee_hive_artifact_catalog::{ArtifactCatalog, FilesystemCatalog};
use std::path::PathBuf;

/// Model catalog for managing model files
pub struct ModelCatalog {
    inner: FilesystemCatalog<ModelEntry>,
}

impl ModelCatalog {
    /// Create a new model catalog
    ///
    /// Models are stored in:
    /// - Linux/Mac: ~/.cache/rbee/models/
    /// - Windows: %LOCALAPPDATA%\rbee\models\
    pub fn new() -> Result<Self> {
        let catalog_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("rbee")
            .join("models");

        let inner = FilesystemCatalog::new(catalog_dir)?;

        Ok(Self { inner })
    }

    /// Create catalog with custom directory (for testing)
    pub fn with_dir(catalog_dir: PathBuf) -> Result<Self> {
        let inner = FilesystemCatalog::new(catalog_dir)?;
        Ok(Self { inner })
    }

    /// Get path where a model would be stored
    pub fn model_path(&self, model_id: &str) -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("rbee")
            .join("models")
            .join(model_id)
    }
}

// Delegate to FilesystemCatalog
impl ArtifactCatalog<ModelEntry> for ModelCatalog {
    fn add(&self, model: ModelEntry) -> Result<()> {
        self.inner.add(model)
    }

    fn get(&self, id: &str) -> Result<ModelEntry> {
        self.inner.get(id)
    }

    fn list(&self) -> Vec<ModelEntry> {
        self.inner.list()
    }

    fn remove(&self, id: &str) -> Result<()> {
        self.inner.remove(id)
    }

    fn contains(&self, id: &str) -> bool {
        self.inner.contains(id)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::new().expect("Failed to create model catalog")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rbee_hive_artifact_catalog::ArtifactStatus;
    use tempfile::TempDir;

    #[test]
    fn test_model_catalog_crud() {
        let temp_dir = TempDir::new().unwrap();
        let catalog = ModelCatalog::with_dir(temp_dir.path().to_path_buf()).unwrap();

        let model = ModelEntry::new(
            "test-model".to_string(),
            "Test Model".to_string(),
            temp_dir.path().join("test-model"),
            1024,
        );

        // Add
        catalog.add(model.clone()).unwrap();
        assert_eq!(catalog.len(), 1);

        // Get
        let retrieved = catalog.get("test-model").unwrap();
        assert_eq!(retrieved.id(), "test-model");

        // List
        let models = catalog.list();
        assert_eq!(models.len(), 1);

        // Remove
        catalog.remove("test-model").unwrap();
        assert_eq!(catalog.len(), 0);
    }
}
