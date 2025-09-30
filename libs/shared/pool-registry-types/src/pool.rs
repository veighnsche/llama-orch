//! Pool types

use serde::{Deserialize, Serialize};

/// Unique identifier for a pool
pub type PoolId = String;

/// Pool metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolMetadata {
    pub pool_id: PoolId,
    pub node_id: Option<String>,
    pub engine: Option<String>,
    pub engine_version: Option<String>,
    pub device_mask: Option<String>,
}

/// Snapshot of pool state for placement decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolSnapshot {
    pub pool_id: PoolId,
    pub node_id: Option<String>,
    pub ready: bool,
    pub draining: bool,
    pub slots_free: u32,
    pub slots_total: u32,
    pub vram_free_bytes: u64,
    pub engine: Option<String>,
    pub models_available: Vec<String>,
}

impl PoolSnapshot {
    pub fn is_available(&self) -> bool {
        self.ready && !self.draining && self.slots_free > 0
    }

    pub fn can_fit_model(&self, required_vram: u64) -> bool {
        self.vram_free_bytes >= required_vram
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_available_pool() {
        let snapshot = PoolSnapshot {
            pool_id: "pool-0".to_string(),
            node_id: Some("node-1".to_string()),
            ready: true,
            draining: false,
            slots_free: 2,
            slots_total: 4,
            vram_free_bytes: 20_000_000_000,
            engine: Some("llamacpp".to_string()),
            models_available: vec![],
        };

        assert!(snapshot.is_available());
        assert!(snapshot.can_fit_model(10_000_000_000));
        assert!(!snapshot.can_fit_model(30_000_000_000));
    }
}
