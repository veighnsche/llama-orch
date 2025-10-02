//! Metadata cache
//!
//! Stores extracted metadata to avoid re-parsing source files when unchanged.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use crate::core::TestMetadata;
use crate::Result;

/// Metadata cache for performance optimization
pub struct MetadataCache {
    cache_file: PathBuf,
}

impl MetadataCache {
    /// Create a cache co-located with the crate manifest directory
    pub fn new(manifest_dir: &Path) -> Self {
        let cache_dir = manifest_dir.join(".proof_bundle");
        let _ = fs::create_dir_all(&cache_dir);
        let cache_file = cache_dir.join("metadata_index.json");
        Self { cache_file }
    }

    /// Attempt to load cached metadata
    pub fn load(&self) -> Option<HashMap<String, TestMetadata>> {
        let bytes = fs::read(&self.cache_file).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Save extracted metadata
    pub fn save(&self, map: &HashMap<String, TestMetadata>) -> Result<()> {
        let data = serde_json::to_vec_pretty(map)?;
        fs::write(&self.cache_file, data).map_err(|e| crate::core::ProofBundleError::Io {
            operation: format!("write {}", self.cache_file.display()),
            source: e,
        })?;
        Ok(())
    }

    /// Check if cache is fresh vs a set of source files
    pub fn is_fresh(&self, sources: &[PathBuf]) -> bool {
        let Ok(cache_meta) = fs::metadata(&self.cache_file) else { return false; };
        let Ok(cache_mtime) = cache_meta.modified() else { return false; };
        for p in sources {
            if let Ok(meta) = fs::metadata(p) {
                if let Ok(mtime) = meta.modified() {
                    if mtime > cache_mtime { return false; }
                }
            }
        }
        true
    }
}
