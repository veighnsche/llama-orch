// TEAM-271: Worker lifecycle types
// TEAM-276: Renamed types for consistency (SpawnConfig → StartConfig, SpawnResult → StartResult)
use serde::{Deserialize, Serialize};

/// Worker start configuration
///
/// TEAM-276: Renamed from WorkerSpawnConfig for consistency
#[derive(Debug, Clone)]
pub struct WorkerStartConfig {
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

/// Result of worker start operation
///
/// TEAM-276: Renamed from SpawnResult for consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartResult {
    /// Worker ID
    pub worker_id: String,

    /// Process ID
    pub pid: u32,

    /// HTTP port
    pub port: u16,

    /// Worker binary path
    pub binary_path: String,
}

/// Worker information
///
/// TEAM-276: Renamed from WorkerProcessInfo for consistency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Process ID
    pub pid: u32,

    /// Command line
    pub command: String,

    /// Arguments
    pub args: Vec<String>,
}
