//! Telemetry types for hive monitoring
//!
//! TEAM-381: Process statistics for worker telemetry
//! Moved from rbee-hive-monitor to contract crate (types only, no runtime deps)

use serde::{Deserialize, Serialize};

// TEAM-381: Optional WASM support for TypeScript type generation
#[cfg(feature = "wasm")]
use tsify::Tsify;

/// Process statistics
/// 
/// TEAM-381: This type is auto-generated for TypeScript via tsify.
/// DO NOT manually define this type in TypeScript - it will be generated automatically.
/// Import from SDK: `import type { ProcessStats } from '@rbee/queen-rbee-sdk'`
/// 
/// Collected from cgroup + GPU monitoring + /proc/pid/cmdline
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ProcessStats {
    /// Process ID
    pub pid: u32,
    /// Service group (e.g., "llm")
    pub group: String,
    /// Instance identifier (e.g., "8080")
    pub instance: String,
    /// CPU usage percentage
    pub cpu_pct: f64,
    /// Resident memory in MB
    pub rss_mb: u64,
    /// I/O read rate in MB/s
    pub io_r_mb_s: f64,
    /// I/O write rate in MB/s
    pub io_w_mb_s: f64,
    /// Process uptime in seconds
    pub uptime_s: u64,
    
    // GPU telemetry
    /// GPU utilization percentage (0.0 = idle, >0 = busy)
    pub gpu_util_pct: f64,
    /// GPU VRAM used in MB
    pub vram_mb: u64,
    /// Total GPU VRAM available in MB
    pub total_vram_mb: u64,
    
    // Model detection
    /// Model name from command line args (e.g., "llama-3.2-1b")
    pub model: Option<String>,
}

/// TEAM-381: Hive telemetry event with worker details
/// 
/// This type is auto-generated for TypeScript via tsify.
/// Source: bin/10_queen_rbee/src/http/heartbeat.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct HiveTelemetry {
    /// Hive identifier
    pub hive_id: String,
    /// Timestamp of telemetry
    pub timestamp: String,
    /// Worker process stats
    pub workers: Vec<ProcessStats>,
}

/// TEAM-381: Queen's own heartbeat
/// 
/// This type is auto-generated for TypeScript via tsify.
/// Source: bin/10_queen_rbee/src/http/heartbeat.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct QueenHeartbeat {
    /// Number of workers online
    pub workers_online: usize,
    /// Number of workers available for work
    pub workers_available: usize,
    /// Number of hives online
    pub hives_online: usize,
    /// Number of hives available
    pub hives_available: usize,
    /// List of worker IDs
    pub worker_ids: Vec<String>,
    /// List of hive IDs
    pub hive_ids: Vec<String>,
    /// Timestamp of heartbeat
    pub timestamp: String,
}

/// TEAM-381: Heartbeat snapshot for SSE stream
/// 
/// This type is auto-generated for TypeScript via tsify.
/// Aggregated view of all workers across all hives
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct HeartbeatSnapshot {
    /// Number of workers online
    pub workers_online: usize,
    /// Number of hives online
    pub hives_online: usize,
    /// Timestamp of snapshot
    pub timestamp: String,
    /// All worker process stats
    pub workers: Vec<ProcessStats>,
}

/// TEAM-381: Hive heartbeat event for SSE stream
/// 
/// This type is auto-generated for TypeScript via tsify.
/// Sent from rbee-hive to UI every 1 second via SSE
/// Source: bin/20_rbee_hive/src/http/heartbeat.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct HiveHeartbeatEvent {
    /// Event type discriminator
    #[serde(rename = "type")]
    pub event_type: String,
    /// Hive identifier
    pub hive_id: String,
    /// Hive information
    pub hive_info: crate::types::HiveInfo,
    /// Timestamp of event
    pub timestamp: String,
    /// Worker process stats
    pub workers: Vec<ProcessStats>,
}
