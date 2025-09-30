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
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

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
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        now_ms - self.last_heartbeat_ms > timeout_ms
    }

    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node_registering() {
        let node = NodeInfo::new(
            "node-1".to_string(),
            "machine-1".to_string(),
            "http://localhost:9200".to_string(),
            vec!["pool-0".to_string()],
            NodeCapabilities {
                gpus: vec![],
                cpu_cores: Some(8),
                ram_total_bytes: Some(16_000_000_000),
            },
        );

        assert_eq!(node.status, NodeStatus::Registering);
        assert!(!node.is_online());
    }

    #[test]
    fn test_online_node_available() {
        let mut node = NodeInfo::new(
            "node-1".to_string(),
            "machine-1".to_string(),
            "http://localhost:9200".to_string(),
            vec![],
            NodeCapabilities {
                gpus: vec![],
                cpu_cores: None,
                ram_total_bytes: None,
            },
        );
        node.status = NodeStatus::Online;
        assert!(node.is_available());
    }
}
