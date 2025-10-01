//! Service registry for tracking GPU nodes in CLOUD_PROFILE
//!
//! Maintains authoritative state of which GPU nodes are online,
//! their capabilities, and pool availability for placement decisions.
//!
//! # ⚠️ AUDIT LOGGING REQUIRED
//!
//! **IMPORTANT**: Node registration/deregistration MUST be logged to `audit-logging`:
//!
//! ```rust,ignore
//! use audit_logging::{AuditLogger, AuditEvent, ActorInfo};
//!
//! // ✅ Node registration
//! audit_logger.emit(AuditEvent::NodeRegistered {
//!     timestamp: Utc::now(),
//!     actor: ActorInfo { user_id: "system", ip, auth_method, session_id },
//!     node_id: node.id.clone(),
//!     capabilities: node.capabilities.clone(),
//! }).await?;
//!
//! // ✅ Node deregistration
//! audit_logger.emit(AuditEvent::NodeDeregistered {
//!     timestamp: Utc::now(),
//!     actor: ActorInfo { user_id: "system", ip, auth_method, session_id },
//!     node_id: node.id.clone(),
//!     reason: "heartbeat_timeout".to_string(),
//! }).await?;
//! ```
//!
//! See: `bin/shared-crates/AUDIT_LOGGING_REMINDER.md`

use pool_registry_types::{NodeId, NodeInfo, NodeStatus};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub mod api;
pub mod heartbeat;

pub use api::{
    HeartbeatPoolStatus, HeartbeatRequest, HeartbeatResponse, RegisterRequest, RegisterResponse,
};

/// Service registry for managing GPU nodes
#[derive(Debug, Clone)]
pub struct ServiceRegistry {
    inner: Arc<Mutex<RegistryInner>>,
}

#[derive(Debug)]
struct RegistryInner {
    nodes: HashMap<NodeId, NodeInfo>,
    pool_to_node: HashMap<String, NodeId>,
    pool_status: HashMap<(NodeId, String), pool_registry_types::PoolSnapshot>,
    heartbeat_timeout_ms: u64,
}

impl ServiceRegistry {
    /// Create a new service registry
    pub fn new(heartbeat_timeout_ms: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner {
                nodes: HashMap::new(),
                pool_to_node: HashMap::new(),
                pool_status: HashMap::new(),
                heartbeat_timeout_ms,
            })),
        }
    }

    /// Register a new node
    pub fn register(&self, node: NodeInfo) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap();

        tracing::info!(
            node_id = %node.node_id,
            machine_id = %node.machine_id,
            address = %node.address,
            pools = ?node.pools,
            "Registering GPU node"
        );

        // Update pool -> node mapping
        for pool_id in &node.pools {
            inner.pool_to_node.insert(pool_id.clone(), node.node_id.clone());
        }

        inner.nodes.insert(node.node_id.clone(), node);
        Ok(())
    }

    /// Process heartbeat from a node
    pub fn heartbeat(&self, node_id: &NodeId) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap();

        let node = inner
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| format!("Node {} not registered", node_id))?;

        node.update_heartbeat();
        if node.status == NodeStatus::Registering {
            node.status = NodeStatus::Online;
        }

        tracing::debug!(
            node_id = %node_id,
            "Heartbeat received"
        );

        Ok(())
    }

    /// Update pool status from heartbeat
    pub fn update_pool_status(
        &self,
        node_id: &NodeId,
        pools: Vec<pool_registry_types::PoolSnapshot>,
    ) {
        let mut inner = self.inner.lock().unwrap();

        for pool in pools {
            let key = (node_id.clone(), pool.pool_id.clone());
            inner.pool_status.insert(key, pool);
        }
    }

    /// Get pool status for a specific node+pool
    pub fn get_pool_status(
        &self,
        node_id: &NodeId,
        pool_id: &str,
    ) -> Option<pool_registry_types::PoolSnapshot> {
        let inner = self.inner.lock().unwrap();
        inner.pool_status.get(&(node_id.clone(), pool_id.to_string())).cloned()
    }

    /// Get all pool statuses for a node
    pub fn get_node_pools(&self, node_id: &NodeId) -> Vec<pool_registry_types::PoolSnapshot> {
        let inner = self.inner.lock().unwrap();
        inner
            .pool_status
            .iter()
            .filter(|((nid, _), _)| nid == node_id)
            .map(|(_, status)| status.clone())
            .collect()
    }

    /// Deregister a node
    pub fn deregister(&self, node_id: &NodeId) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap();

        if let Some(node) = inner.nodes.remove(node_id) {
            tracing::info!(
                node_id = %node_id,
                "Deregistering GPU node"
            );

            // Remove pool mappings
            for pool_id in &node.pools {
                inner.pool_to_node.remove(pool_id);
            }
            Ok(())
        } else {
            Err(format!("Node {} not found", node_id))
        }
    }

    /// Get all online nodes
    pub fn get_online_nodes(&self) -> Vec<NodeInfo> {
        let inner = self.inner.lock().unwrap();
        inner.nodes.values().filter(|n| n.is_available()).cloned().collect()
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: &NodeId) -> Option<NodeInfo> {
        let inner = self.inner.lock().unwrap();
        inner.nodes.get(node_id).cloned()
    }

    /// Get node for a specific pool
    pub fn get_node_for_pool(&self, pool_id: &str) -> Option<NodeInfo> {
        let inner = self.inner.lock().unwrap();
        inner.pool_to_node.get(pool_id).and_then(|node_id| inner.nodes.get(node_id)).cloned()
    }

    /// Check for stale nodes and mark offline
    pub fn check_stale_nodes(&self) -> Vec<NodeId> {
        let mut inner = self.inner.lock().unwrap();
        let mut stale_nodes = Vec::new();
        let timeout_ms = inner.heartbeat_timeout_ms;

        for (node_id, node) in inner.nodes.iter_mut() {
            if node.status == NodeStatus::Online && node.is_stale(timeout_ms) {
                tracing::warn!(
                    node_id = %node_id,
                    last_heartbeat_ms = node.last_heartbeat_ms,
                    "Node marked offline due to stale heartbeat"
                );
                node.status = NodeStatus::Offline;
                stale_nodes.push(node_id.clone());
            }
        }

        stale_nodes
    }

    /// Get count of registered nodes
    pub fn node_count(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.nodes.len()
    }

    /// Get count of online nodes
    pub fn online_node_count(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.nodes.values().filter(|n| n.is_online()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pool_registry_types::{GpuInfo, NodeCapabilities};

    fn test_node(node_id: &str) -> NodeInfo {
        NodeInfo::new(
            node_id.to_string(),
            format!("machine-{}", node_id),
            format!("http://{}:9200", node_id),
            vec![format!("pool-{}", node_id)],
            NodeCapabilities {
                gpus: vec![GpuInfo {
                    device_id: 0,
                    name: "RTX 3090".to_string(),
                    vram_total_bytes: 24_000_000_000,
                    compute_capability: "8.6".to_string(),
                }],
                cpu_cores: Some(8),
                ram_total_bytes: Some(32_000_000_000),
            },
        )
    }

    #[test]
    fn test_register_node() {
        let registry = ServiceRegistry::new(30_000);
        let node = test_node("node-1");

        registry.register(node.clone()).unwrap();
        assert_eq!(registry.node_count(), 1);

        let retrieved = registry.get_node(&node.node_id).unwrap();
        assert_eq!(retrieved.node_id, "node-1");
    }

    #[test]
    fn test_heartbeat_transitions_to_online() {
        let registry = ServiceRegistry::new(30_000);
        let node = test_node("node-1");

        registry.register(node.clone()).unwrap();

        let before = registry.get_node(&node.node_id).unwrap();
        assert_eq!(before.status, NodeStatus::Registering);

        registry.heartbeat(&node.node_id).unwrap();

        let after = registry.get_node(&node.node_id).unwrap();
        assert_eq!(after.status, NodeStatus::Online);
    }

    #[test]
    fn test_deregister_node() {
        let registry = ServiceRegistry::new(30_000);
        let node = test_node("node-1");

        registry.register(node.clone()).unwrap();
        assert_eq!(registry.node_count(), 1);

        registry.deregister(&node.node_id).unwrap();
        assert_eq!(registry.node_count(), 0);
    }

    #[test]
    fn test_get_node_for_pool() {
        let registry = ServiceRegistry::new(30_000);
        let node = test_node("node-1");

        registry.register(node.clone()).unwrap();

        let retrieved = registry.get_node_for_pool("pool-node-1").unwrap();
        assert_eq!(retrieved.node_id, "node-1");
    }

    #[test]
    fn test_online_nodes_only_available() {
        let registry = ServiceRegistry::new(30_000);

        let mut node1 = test_node("node-1");
        let mut node2 = test_node("node-2");

        node1.status = NodeStatus::Online;
        node2.status = NodeStatus::Offline;

        registry.register(node1).unwrap();
        registry.register(node2).unwrap();

        let online = registry.get_online_nodes();
        assert_eq!(online.len(), 1);
        assert_eq!(online[0].node_id, "node-1");
    }
}
