// TEAM-135: Created by TEAM-135 (scaffolding)
// Purpose: System monitoring for rbee-hive
// Status: STUB - Awaiting implementation
// TEAM-XXX: RULE ZERO - This is THE process monitoring crate (not process-monitor)

#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-hive-monitor
//!
//! Cross-platform process monitoring and resource management for rbee-hive workers.
//!
//! Provides:
//! - Starting processes with resource limits (cgroup on Linux)
//! - Monitoring process resources (CPU, memory, I/O)
//! - Grouping processes for management
//! - Cross-platform abstraction (Linux, macOS, Windows)

use serde::{Deserialize, Serialize};

// TEAM-359: Platform-specific imports

/// Configuration for process monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Service group (e.g., "llm", "embedding")
    pub group: String,
    
    /// Instance identifier (usually port number)
    pub instance: String,
    
    /// CPU limit (e.g., "200%" = 2 cores, "100%" = 1 core)
    /// Linux: Enforced via cgroup
    /// macOS/Windows: Best-effort or not supported
    pub cpu_limit: Option<String>,
    
    /// Memory limit (e.g., "4G", "512M")
    /// Linux: Enforced via cgroup
    /// macOS: Enforced via launchd (if used)
    /// Windows: Enforced via Job Objects (if used)
    pub memory_limit: Option<String>,
}

// TEAM-381: Optional WASM support for TypeScript type generation
#[cfg(feature = "wasm")]
use tsify::Tsify;

/// Process statistics
/// 
/// TEAM-381: This type is auto-generated for TypeScript via tsify.
/// DO NOT manually define this type in TypeScript - it will be generated automatically.
/// Import from SDK: `import type { ProcessStats } from '@rbee/queen-rbee-sdk'`
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
    
    // TEAM-360: GPU telemetry
    /// GPU utilization percentage (0.0 = idle, >0 = busy)
    pub gpu_util_pct: f64,
    /// GPU VRAM used in MB
    pub vram_mb: u64,
    /// Total GPU VRAM available in MB (TEAM-364: Critical Issue #5)
    pub total_vram_mb: u64,
    
    // TEAM-360: Model detection
    /// Model name from command line args (e.g., "llama-3.2-1b")
    pub model: Option<String>,
}

// TEAM-359: Process monitoring implementation
mod monitor;
mod telemetry;

pub use monitor::ProcessMonitor;
pub use telemetry::{collect_all_workers, collect_group, collect_instance};
