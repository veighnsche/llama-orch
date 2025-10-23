// TEAM-273: Worker catalog using artifact-catalog abstraction
#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-hive-worker-catalog
//!
//! Worker catalog for managing worker binaries.
//! Built on top of artifact-catalog for consistency with model-catalog.

mod types;

pub use types::{Platform, WorkerBinary, WorkerStatus, WorkerType};

use anyhow::Result;
use rbee_hive_artifact_catalog::{ArtifactCatalog, FilesystemCatalog};
use std::path::PathBuf;

/// Worker catalog for managing worker binaries
pub struct WorkerCatalog {
    inner: FilesystemCatalog<WorkerBinary>,
}

impl WorkerCatalog {
    /// Create a new worker catalog
    ///
    /// Workers are stored in:
    /// - Linux/Mac: ~/.cache/rbee/workers/
    /// - Windows: %LOCALAPPDATA%\rbee\workers\
    pub fn new() -> Result<Self> {
        let catalog_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
            .join("rbee")
            .join("workers");
        
        let inner = FilesystemCatalog::new(catalog_dir)?;
        
        Ok(Self { inner })
    }
    
    /// Create catalog with custom directory (for testing)
    pub fn with_dir(catalog_dir: PathBuf) -> Result<Self> {
        let inner = FilesystemCatalog::new(catalog_dir)?;
        Ok(Self { inner })
    }
    
    /// Get path where a worker binary would be stored
    pub fn worker_path(&self, worker_id: &str) -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("rbee")
            .join("workers")
            .join(worker_id)
    }
    
    /// Find worker binary by type and platform
    pub fn find_by_type_and_platform(
        &self,
        worker_type: WorkerType,
        platform: Platform,
    ) -> Option<WorkerBinary> {
        self.list()
            .into_iter()
            .find(|w| w.worker_type() == &worker_type && w.platform() == &platform)
    }
}

// Delegate to FilesystemCatalog
impl ArtifactCatalog<WorkerBinary> for WorkerCatalog {
    fn add(&self, worker: WorkerBinary) -> Result<()> {
        self.inner.add(worker)
    }
    
    fn get(&self, id: &str) -> Result<WorkerBinary> {
        self.inner.get(id)
    }
    
    fn list(&self) -> Vec<WorkerBinary> {
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

impl Default for WorkerCatalog {
    fn default() -> Self {
        Self::new().expect("Failed to create worker catalog")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_worker_catalog_crud() {
        let temp_dir = TempDir::new().unwrap();
        let catalog = WorkerCatalog::with_dir(temp_dir.path().to_path_buf()).unwrap();
        
        let worker = WorkerBinary::new(
            "cpu-llm-worker-rbee-v0.1.0-linux".to_string(),
            WorkerType::CpuLlm,
            Platform::Linux,
            temp_dir.path().join("cpu-llm-worker-rbee"),
            1024 * 1024, // 1 MB
            "0.1.0".to_string(),
        );
        
        // Add
        catalog.add(worker.clone()).unwrap();
        assert_eq!(catalog.len(), 1);
        
        // Get
        let retrieved = catalog.get("cpu-llm-worker-rbee-v0.1.0-linux").unwrap();
        assert_eq!(retrieved.worker_type(), &WorkerType::CpuLlm);
        
        // List
        let workers = catalog.list();
        assert_eq!(workers.len(), 1);
        
        // Find by type and platform
        let found = catalog.find_by_type_and_platform(WorkerType::CpuLlm, Platform::Linux);
        assert!(found.is_some());
        
        // Remove
        catalog.remove("cpu-llm-worker-rbee-v0.1.0-linux").unwrap();
        assert_eq!(catalog.len(), 0);
    }
}
