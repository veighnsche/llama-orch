//! model-eviction — Model eviction policy for VRAM management

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
}
