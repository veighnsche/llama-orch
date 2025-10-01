//! model-eviction — Model eviction policy for VRAM management
//!
//! TODO(ARCH-CHANGE): This crate is minimal. Needs full implementation:
//! - Implement LRU (Least Recently Used) eviction
//! - Implement LFU (Least Frequently Used) eviction
//! - Implement cost-based eviction (evict cheapest to reload)
//! - Add VRAM pressure detection
//! - Implement eviction scoring algorithm
//! - Add metrics for eviction events
//! - Coordinate with model-cache and worker-orcd
//! See: ARCHITECTURE_CHANGE_PLAN.md Phase 4 (Production Hardening)

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

pub enum EvictionPolicy {
    Lru,
    Lfu,
    CostBased,
}

pub struct EvictionManager {
    policy: EvictionPolicy,
}

impl EvictionManager {
    pub fn new(policy: EvictionPolicy) -> Self {
        Self { policy }
    }
    
    // TODO(ARCH-CHANGE): Add eviction methods:
    // - pub fn select_victim(&self, models: &[Model]) -> Option<String>
    // - pub fn record_access(&mut self, model_ref: &str)
    // - pub fn calculate_score(&self, model: &Model) -> f64
    // - pub fn should_evict(&self, vram_pressure: f64) -> bool
    // - pub fn evict_until_free(&mut self, required_vram: u64) -> Result<Vec<String>>
}
