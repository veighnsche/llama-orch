//! Node types for service discovery

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique identifier for a node
pub type NodeId = String;

/// Node status in the cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is registering
    Registering,
    /// Node is online and accepting tasks
    Online,
    /// Node is draining (no new tasks)
    Draining,
    /// Node is offline (missed heartbeat)
    Offline,
    /// Node has failed
    Failed,
}

/// GPU information for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub device_id: u32,
    pub name: String,
    pub vram_total_bytes: u64,
    pub compute_capability: String,
}

/// Node capabilities reported during registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub gpus: Vec<GpuInfo>,
    pub cpu_cores: Option<u32>,
    pub ram_total_bytes: Option<u64>,
}

/// Complete information about a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: NodeId,
    pub machine_id: String,
    pub address: String,
    pub pools: Vec<String>,
    pub capabilities: NodeCapabilities,
    pub status: NodeStatus,
    pub registered_at_ms: u64,
    pub last_heartbeat_ms: u64,
    pub version: Option<String>,
}

impl NodeInfo {
    pub fn new(
        node_id: NodeId,
        machine_id: String,
        address: String,
        pools: Vec<String>,
        capabilities: NodeCapabilities,
    ) -> Self {
        let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;

        Self {
            node_id,
            machine_id,
            address,
            pools,
            capabilities,
            status: NodeStatus::Registering,
            registered_at_ms: now_ms,
            last_heartbeat_ms: now_ms,
            version: None,
        }
    }

    pub fn is_online(&self) -> bool {
        self.status == NodeStatus::Online
    }

    pub fn is_available(&self) -> bool {
        self.status == NodeStatus::Online && !self.is_stale(30_000)
    }

    pub fn is_stale(&self, timeout_ms: u64) -> bool {
        let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
        now_ms - self.last_heartbeat_ms > timeout_ms
    }

    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat_ms =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = NodeInfo::new(
            "node-1".to_string(),
            "machine-1".to_string(),
            "http://localhost:9200".to_string(),
            vec!["pool-0".to_string()],
            NodeCapabilities {
                gpus: vec![],
                cpu_cores: Some(8),
                ram_total_bytes: Some(32_000_000_000),
            },
        );

        assert_eq!(node.node_id, "node-1");
        assert_eq!(node.status, NodeStatus::Registering);
        assert!(!node.is_online());
    }

    #[test]
    fn test_node_heartbeat() {
        let mut node = NodeInfo::new(
            "node-1".to_string(),
            "machine-1".to_string(),
            "http://localhost:9200".to_string(),
            vec!["pool-0".to_string()],
            NodeCapabilities {
                gpus: vec![],
                cpu_cores: Some(8),
                ram_total_bytes: Some(32_000_000_000),
            },
        );

        let before = node.last_heartbeat_ms;
        std::thread::sleep(std::time::Duration::from_millis(10));
        node.update_heartbeat();
        let after = node.last_heartbeat_ms;

        assert!(after > before);
    }

    #[test]
    fn test_node_status_transitions() {
        let mut node = NodeInfo::new(
            "node-1".to_string(),
            "machine-1".to_string(),
            "http://localhost:9200".to_string(),
            vec!["pool-0".to_string()],
            NodeCapabilities { gpus: vec![], cpu_cores: Some(8), ram_total_bytes: None },
        );

        assert_eq!(node.status, NodeStatus::Registering);
        assert!(!node.is_online());

        node.status = NodeStatus::Online;
        assert!(node.is_online());
        assert!(node.is_available());

        node.status = NodeStatus::Draining;
        assert!(!node.is_online());

        node.status = NodeStatus::Offline;
        assert!(!node.is_online());
        assert!(!node.is_available());
    }

    #[test]
    fn test_node_stale_detection() {
        let mut node = NodeInfo::new(
            "node-1".to_string(),
            "machine-1".to_string(),
            "http://localhost:9200".to_string(),
            vec!["pool-0".to_string()],
            NodeCapabilities {
                gpus: vec![],
                cpu_cores: Some(8),
                ram_total_bytes: Some(32_000_000_000),
            },
        );

        // Fresh node should not be stale
        assert!(!node.is_stale(30_000));

        // Simulate old heartbeat
        node.last_heartbeat_ms = node.last_heartbeat_ms - 40_000;
        assert!(node.is_stale(30_000));
    }

    #[test]
    fn test_node_with_gpus() {
        let node = NodeInfo::new(
            "gpu-node-1".to_string(),
            "machine-gpu".to_string(),
            "http://localhost:9200".to_string(),
            vec!["pool-0".to_string(), "pool-1".to_string()],
            NodeCapabilities {
                gpus: vec![
                    GpuInfo {
                        device_id: 0,
                        name: "RTX 3090".to_string(),
                        vram_total_bytes: 24_000_000_000,
                        compute_capability: "8.6".to_string(),
                    },
                    GpuInfo {
                        device_id: 1,
                        name: "RTX 3090".to_string(),
                        vram_total_bytes: 24_000_000_000,
                        compute_capability: "8.6".to_string(),
                    },
                ],
                cpu_cores: Some(16),
                ram_total_bytes: Some(64_000_000_000),
            },
        );

        assert_eq!(node.capabilities.gpus.len(), 2);
        assert_eq!(node.pools.len(), 2);
        assert_eq!(node.capabilities.cpu_cores, Some(16));
    }

    #[test]
    fn test_node_serialization() {
        let node = NodeInfo::new(
            "node-1".to_string(),
            "machine-1".to_string(),
            "http://localhost:9200".to_string(),
            vec!["pool-0".to_string()],
            NodeCapabilities {
                gpus: vec![],
                cpu_cores: Some(8),
                ram_total_bytes: Some(32_000_000_000),
            },
        );

        let json = serde_json::to_string(&node).unwrap();
        let deserialized: NodeInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.node_id, "node-1");
        assert_eq!(deserialized.machine_id, "machine-1");
        assert_eq!(deserialized.status, NodeStatus::Registering);
    }

    #[test]
    fn test_node_available_requires_online_and_fresh() {
        let mut node = NodeInfo::new(
            "node-1".to_string(),
            "machine-1".to_string(),
            "http://localhost:9200".to_string(),
            vec!["pool-0".to_string()],
            NodeCapabilities {
                gpus: vec![],
                cpu_cores: Some(8),
                ram_total_bytes: Some(32_000_000_000),
            },
        );

        // Registering node is not available
        assert!(!node.is_available());

        // Online node is available
        node.status = NodeStatus::Online;
        assert!(node.is_available());

        // Stale online node is not available
        node.last_heartbeat_ms = node.last_heartbeat_ms - 40_000;
        assert!(!node.is_available());
    }

    #[test]
    fn test_gpu_info_fields() {
        let gpu = GpuInfo {
            device_id: 0,
            name: "RTX 4090".to_string(),
            vram_total_bytes: 24_000_000_000,
            compute_capability: "8.9".to_string(),
        };

        assert_eq!(gpu.device_id, 0);
        assert_eq!(gpu.name, "RTX 4090");
        assert_eq!(gpu.vram_total_bytes, 24_000_000_000);
        assert_eq!(gpu.compute_capability, "8.9");
    }
}
