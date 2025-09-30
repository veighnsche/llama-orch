//! API types for service registry

use pool_registry_types::NodeCapabilities;
use serde::{Deserialize, Serialize};

/// Request to register a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub node_id: String,
    pub machine_id: String,
    pub address: String,
    pub pools: Vec<String>,
    pub capabilities: NodeCapabilities,
    pub version: Option<String>,
}

/// Pool status in heartbeat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatPoolStatus {
    pub pool_id: String,
    pub ready: bool,
    pub draining: bool,
    pub slots_free: u32,
    pub slots_total: u32,
    pub vram_free_bytes: u64,
    pub engine: Option<String>,
}

/// Request to send heartbeat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub timestamp: String,
    pub pools: Vec<HeartbeatPoolStatus>,
}

/// Response to registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub success: bool,
    pub message: String,
    pub node_id: String,
}

/// Response to heartbeat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub success: bool,
    pub next_heartbeat_ms: u64,
}
