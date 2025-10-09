//! Model catalog types and management
//!
//! Created by: TEAM-022

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCatalog {
    pub version: String,
    pub pool_id: String,
    pub updated_at: String,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub format: String,
    pub size_gb: f64,
    pub architecture: String,
    pub downloaded: bool,
    pub backends: Vec<String>,
    pub metadata: serde_json::Value,
}

impl ModelCatalog {
    /// Create a new empty catalog
    pub fn new(pool_id: String) -> Self {
        Self {
            version: "1.0".to_string(),
            pool_id,
            updated_at: chrono::Utc::now().to_rfc3339(),
            models: Vec::new(),
        }
    }

    /// Load catalog from file
    pub fn load(path: &Path) -> crate::Result<Self> {
        if !path.exists() {
            return Err(crate::PoolError::CatalogNotFound(path.display().to_string()));
        }
        let content = std::fs::read_to_string(path)?;
        let catalog: Self = serde_json::from_str(&content)?;
        Ok(catalog)
    }

    /// Save catalog to file
    pub fn save(&self, path: &Path) -> crate::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Add a model to the catalog
    pub fn add_model(&mut self, entry: ModelEntry) -> crate::Result<()> {
        if self.find_model(&entry.id).is_some() {
            return Err(crate::PoolError::ModelAlreadyExists(entry.id.clone()));
        }
        self.models.push(entry);
        self.updated_at = chrono::Utc::now().to_rfc3339();
        Ok(())
    }

    /// Remove a model from the catalog
    pub fn remove_model(&mut self, id: &str) -> crate::Result<()> {
        let index = self
            .models
            .iter()
            .position(|m| m.id == id)
            .ok_or_else(|| crate::PoolError::ModelNotFound(id.to_string()))?;
        self.models.remove(index);
        self.updated_at = chrono::Utc::now().to_rfc3339();
        Ok(())
    }

    /// Find a model by ID
    pub fn find_model(&self, id: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Find a model by ID (mutable)
    pub fn find_model_mut(&mut self, id: &str) -> Option<&mut ModelEntry> {
        self.models.iter_mut().find(|m| m.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_create() {
        let catalog = ModelCatalog::new("test-pool".to_string());
        assert_eq!(catalog.pool_id, "test-pool");
        assert_eq!(catalog.version, "1.0");
        assert_eq!(catalog.models.len(), 0);
    }

    #[test]
    fn test_catalog_add_model() {
        let mut catalog = ModelCatalog::new("test-pool".to_string());
        let entry = ModelEntry {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            path: PathBuf::from(".test-models/test"),
            format: "safetensors".to_string(),
            size_gb: 1.0,
            architecture: "llama".to_string(),
            downloaded: false,
            backends: vec!["cpu".to_string()],
            metadata: serde_json::json!({"repo": "test/repo"}),
        };
        catalog.add_model(entry).unwrap();
        assert_eq!(catalog.models.len(), 1);
    }

    #[test]
    fn test_catalog_add_duplicate() {
        let mut catalog = ModelCatalog::new("test-pool".to_string());
        let entry = ModelEntry {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            path: PathBuf::from(".test-models/test"),
            format: "safetensors".to_string(),
            size_gb: 1.0,
            architecture: "llama".to_string(),
            downloaded: false,
            backends: vec!["cpu".to_string()],
            metadata: serde_json::json!({"repo": "test/repo"}),
        };
        catalog.add_model(entry.clone()).unwrap();
        assert!(catalog.add_model(entry).is_err());
    }

    #[test]
    fn test_catalog_remove_model() {
        let mut catalog = ModelCatalog::new("test-pool".to_string());
        let entry = ModelEntry {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            path: PathBuf::from(".test-models/test"),
            format: "safetensors".to_string(),
            size_gb: 1.0,
            architecture: "llama".to_string(),
            downloaded: false,
            backends: vec!["cpu".to_string()],
            metadata: serde_json::json!({"repo": "test/repo"}),
        };
        catalog.add_model(entry).unwrap();
        catalog.remove_model("test-model").unwrap();
        assert_eq!(catalog.models.len(), 0);
    }

    #[test]
    fn test_catalog_save_load() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("catalog.json");

        let mut catalog = ModelCatalog::new("test-pool".to_string());
        let entry = ModelEntry {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            path: PathBuf::from(".test-models/test"),
            format: "safetensors".to_string(),
            size_gb: 1.0,
            architecture: "llama".to_string(),
            downloaded: false,
            backends: vec!["cpu".to_string()],
            metadata: serde_json::json!({"repo": "test/repo"}),
        };
        catalog.add_model(entry).unwrap();
        catalog.save(&path).unwrap();

        let loaded = ModelCatalog::load(&path).unwrap();
        assert_eq!(loaded.pool_id, catalog.pool_id);
        assert_eq!(loaded.models.len(), 1);
        assert_eq!(loaded.models[0].id, "test-model");
    }
}
