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
    fn test_pool_snapshot_available() {
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

    #[test]
    fn test_pool_snapshot_not_ready() {
        let snapshot = PoolSnapshot {
            pool_id: "pool-0".to_string(),
            node_id: Some("node-1".to_string()),
            ready: false,
            draining: false,
            slots_free: 2,
            slots_total: 4,
            vram_free_bytes: 20_000_000_000,
            engine: Some("llamacpp".to_string()),
            models_available: vec![],
        };

        assert!(!snapshot.is_available());
    }

    #[test]
    fn test_pool_snapshot_draining() {
        let snapshot = PoolSnapshot {
            pool_id: "pool-0".to_string(),
            node_id: Some("node-1".to_string()),
            ready: true,
            draining: true,
            slots_free: 2,
            slots_total: 4,
            vram_free_bytes: 20_000_000_000,
            engine: Some("llamacpp".to_string()),
            models_available: vec![],
        };

        assert!(!snapshot.is_available());
    }

    #[test]
    fn test_pool_snapshot_no_slots() {
        let snapshot = PoolSnapshot {
            pool_id: "pool-0".to_string(),
            node_id: Some("node-1".to_string()),
            ready: true,
            draining: false,
            slots_free: 0,
            slots_total: 4,
            vram_free_bytes: 20_000_000_000,
            engine: Some("llamacpp".to_string()),
            models_available: vec![],
        };

        assert!(!snapshot.is_available());
    }

    #[test]
    fn test_pool_metadata_serialization() {
        let metadata = PoolMetadata {
            pool_id: "pool-0".to_string(),
            node_id: Some("node-1".to_string()),
            engine: Some("llamacpp".to_string()),
            engine_version: Some("v1.0".to_string()),
            device_mask: Some("GPU0".to_string()),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: PoolMetadata = serde_json::from_str(&json).unwrap();
        
        assert_eq!(deserialized.pool_id, "pool-0");
        assert_eq!(deserialized.node_id, Some("node-1".to_string()));
    }

    #[test]
    fn test_pool_snapshot_vram_check_exact() {
        let snapshot = PoolSnapshot {
            pool_id: "pool-0".to_string(),
            node_id: None,
            ready: true,
            draining: false,
            slots_free: 1,
            slots_total: 1,
            vram_free_bytes: 10_000_000_000,
            engine: Some("llamacpp".to_string()),
            models_available: vec![],
        };

        assert!(snapshot.can_fit_model(10_000_000_000));
        assert!(!snapshot.can_fit_model(10_000_000_001));
    }

    #[test]
    fn test_pool_snapshot_models_available() {
        let snapshot = PoolSnapshot {
            pool_id: "pool-0".to_string(),
            node_id: Some("node-1".to_string()),
            ready: true,
            draining: false,
            slots_free: 2,
            slots_total: 4,
            vram_free_bytes: 20_000_000_000,
            engine: Some("llamacpp".to_string()),
            models_available: vec!["model-a".to_string(), "model-b".to_string()],
        };

        assert_eq!(snapshot.models_available.len(), 2);
        assert!(snapshot.models_available.contains(&"model-a".to_string()));
    }
}
