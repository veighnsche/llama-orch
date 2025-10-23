// TEAM-271: Worker lifecycle types
use serde::{Deserialize, Serialize};

/// Worker spawn configuration
#[derive(Debug, Clone)]
pub struct WorkerSpawnConfig {
    /// Worker ID
    pub worker_id: String,

    /// Model ID to load
    pub model_id: String,

    /// Device (e.g., "cpu", "cuda:0", "metal")
    pub device: String,

    /// Port for worker HTTP server
    pub port: u16,

    /// Queen URL for heartbeat
    pub queen_url: String,

    /// Job ID for narration routing
    pub job_id: String,
}

/// Result of worker spawn operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnResult {
    /// Worker ID
    pub worker_id: String,

    /// Process ID
    pub pid: u32,

    /// HTTP port
    pub port: u16,

    /// Worker binary path
    pub binary_path: String,
}
