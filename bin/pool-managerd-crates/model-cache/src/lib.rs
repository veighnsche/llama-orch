//! model-cache — Model caching
//!
//! TODO(ARCH-CHANGE): This crate is a stub. Needs full implementation:
//! - Implement LRU cache for frequently used models
//! - Add cache hit/miss tracking and metrics
//! - Implement cache warming (preload popular models)
//! - Add cache eviction policy (coordinate with model-eviction crate)
//! - Implement cache persistence across restarts
//! - Add cache size limits (disk space, VRAM)
//! - Integrate with model-catalog for metadata
//! See: ARCHITECTURE_CHANGE_PLAN.md §2.11 (Model Provisioner evaluation)

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

pub struct ModelCache;

impl ModelCache {
    pub fn new() -> Self {
        Self
    }

    // TODO(ARCH-CHANGE): Add cache methods:
    // - pub fn get(&self, model_ref: &str) -> Option<PathBuf>
    // - pub fn put(&mut self, model_ref: &str, path: PathBuf) -> Result<()>
    // - pub fn evict(&mut self, model_ref: &str) -> Result<()>
    // - pub fn warm(&mut self, models: Vec<String>) -> Result<()>
    // - pub fn stats(&self) -> CacheStats (hits, misses, size)
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}
