// TEAM-273: Generic catalog implementation
use crate::types::{Artifact, ArtifactMetadata};
use anyhow::{anyhow, Result};
use std::path::PathBuf;

/// Generic artifact catalog trait
pub trait ArtifactCatalog<T: Artifact> {
    /// Add an artifact to the catalog
    fn add(&self, artifact: T) -> Result<()>;

    /// Get an artifact by ID
    fn get(&self, id: &str) -> Result<T>;

    /// List all artifacts
    fn list(&self) -> Vec<T>;

    /// Remove an artifact from catalog
    fn remove(&self, id: &str) -> Result<()>;

    /// Check if artifact exists
    fn contains(&self, id: &str) -> bool;

    /// Get artifact count
    fn len(&self) -> usize;

    /// Check if catalog is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Filesystem-based catalog implementation
///
/// Stores artifacts as JSON files in a directory.
/// Each artifact gets its own subdirectory with metadata.json.
pub struct FilesystemCatalog<T: Artifact> {
    catalog_dir: PathBuf,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Artifact> FilesystemCatalog<T> {
    /// Create a new filesystem catalog
    ///
    /// # Arguments
    /// * `catalog_dir` - Base directory for catalog (e.g., ~/.cache/rbee/models/)
    pub fn new(catalog_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&catalog_dir)?;

        Ok(Self { catalog_dir, _phantom: std::marker::PhantomData })
    }

    /// Get path to artifact's metadata file
    fn metadata_path(&self, id: &str) -> PathBuf {
        self.catalog_dir.join(id).join("metadata.json")
    }

    /// Load metadata from disk
    fn load_metadata(&self, id: &str) -> Result<ArtifactMetadata<T>> {
        let path = self.metadata_path(id);

        if !path.exists() {
            return Err(anyhow!("Artifact '{}' not found in catalog", id));
        }

        let contents = std::fs::read_to_string(&path)?;
        let metadata: ArtifactMetadata<T> = serde_json::from_str(&contents)?;

        Ok(metadata)
    }

    /// Save metadata to disk
    fn save_metadata(&self, id: &str, metadata: &ArtifactMetadata<T>) -> Result<()> {
        let dir = self.catalog_dir.join(id);
        std::fs::create_dir_all(&dir)?;

        let path = self.metadata_path(id);
        let contents = serde_json::to_string_pretty(metadata)?;
        std::fs::write(&path, contents)?;

        Ok(())
    }

    /// Get all artifact IDs
    fn list_ids(&self) -> Vec<String> {
        let mut ids = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&self.catalog_dir) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        ids.push(name.to_string());
                    }
                }
            }
        }

        ids
    }
}

impl<T: Artifact> ArtifactCatalog<T> for FilesystemCatalog<T> {
    fn add(&self, artifact: T) -> Result<()> {
        let id = artifact.id().to_string();

        // Check if already exists
        if self.contains(&id) {
            return Err(anyhow!("Artifact '{}' already exists in catalog", id));
        }

        // Create metadata
        let metadata = ArtifactMetadata::new(artifact);

        // Save to disk
        self.save_metadata(&id, &metadata)?;

        Ok(())
    }

    fn get(&self, id: &str) -> Result<T> {
        let mut metadata = self.load_metadata(id)?;
        metadata.touch();

        // Update last accessed time
        self.save_metadata(id, &metadata)?;

        Ok(metadata.artifact)
    }

    fn list(&self) -> Vec<T> {
        let mut artifacts = Vec::new();

        for id in self.list_ids() {
            if let Ok(metadata) = self.load_metadata(&id) {
                artifacts.push(metadata.artifact);
            }
        }

        artifacts
    }

    fn remove(&self, id: &str) -> Result<()> {
        let dir = self.catalog_dir.join(id);

        if !dir.exists() {
            return Err(anyhow!("Artifact '{}' not found in catalog", id));
        }

        std::fs::remove_dir_all(&dir)?;

        Ok(())
    }

    fn contains(&self, id: &str) -> bool {
        self.metadata_path(id).exists()
    }

    fn len(&self) -> usize {
        self.list_ids().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ArtifactStatus;
    use serde::{Deserialize, Serialize};
    use tempfile::TempDir;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestArtifact {
        id: String,
        path: PathBuf,
        size: u64,
        status: ArtifactStatus,
    }

    impl Artifact for TestArtifact {
        fn id(&self) -> &str {
            &self.id
        }

        fn path(&self) -> &std::path::Path {
            &self.path
        }

        fn size(&self) -> u64 {
            self.size
        }

        fn status(&self) -> &ArtifactStatus {
            &self.status
        }

        fn set_status(&mut self, status: ArtifactStatus) {
            self.status = status;
        }
    }

    fn create_test_artifact(id: &str) -> TestArtifact {
        TestArtifact {
            id: id.to_string(),
            path: PathBuf::from(format!("/tmp/{}", id)),
            size: 1024,
            status: ArtifactStatus::Available,
        }
    }

    #[test]
    fn test_add_and_get() {
        let temp_dir = TempDir::new().unwrap();
        let catalog = FilesystemCatalog::new(temp_dir.path().to_path_buf()).unwrap();

        let artifact = create_test_artifact("test-1");
        catalog.add(artifact.clone()).unwrap();

        let retrieved = catalog.get("test-1").unwrap();
        assert_eq!(retrieved.id(), "test-1");
        assert_eq!(retrieved.size(), 1024);
    }

    #[test]
    fn test_add_duplicate() {
        let temp_dir = TempDir::new().unwrap();
        let catalog = FilesystemCatalog::new(temp_dir.path().to_path_buf()).unwrap();

        let artifact = create_test_artifact("test-1");
        catalog.add(artifact.clone()).unwrap();

        let result = catalog.add(artifact);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_list() {
        let temp_dir = TempDir::new().unwrap();
        let catalog = FilesystemCatalog::new(temp_dir.path().to_path_buf()).unwrap();

        for i in 0..3 {
            let artifact = create_test_artifact(&format!("test-{}", i));
            catalog.add(artifact).unwrap();
        }

        let artifacts = catalog.list();
        assert_eq!(artifacts.len(), 3);
    }

    #[test]
    fn test_remove() {
        let temp_dir = TempDir::new().unwrap();
        let catalog = FilesystemCatalog::new(temp_dir.path().to_path_buf()).unwrap();

        let artifact = create_test_artifact("test-1");
        catalog.add(artifact).unwrap();

        assert!(catalog.contains("test-1"));
        catalog.remove("test-1").unwrap();
        assert!(!catalog.contains("test-1"));
    }

    #[test]
    fn test_contains() {
        let temp_dir = TempDir::new().unwrap();
        let catalog = FilesystemCatalog::new(temp_dir.path().to_path_buf()).unwrap();

        assert!(!catalog.contains("test-1"));

        let artifact = create_test_artifact("test-1");
        catalog.add(artifact).unwrap();

        assert!(catalog.contains("test-1"));
    }
}
