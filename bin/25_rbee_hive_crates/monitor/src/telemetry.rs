//! TEAM-359: Telemetry collection for heartbeat integration
//!
//! This module provides high-level APIs for collecting worker telemetry
//! to be included in Hive → Queen heartbeats.

use crate::{ProcessMonitor, ProcessStats};
use anyhow::Result;

/// Collect telemetry for all monitored workers
///
/// This is the main API used by rbee-hive for heartbeat generation.
/// It walks the cgroup tree and collects stats for all workers.
///
/// # Returns
/// Vector of ProcessStats for all monitored processes
///
/// # Platform Support
/// - Linux: Reads from /sys/fs/cgroup/rbee.slice/
/// - macOS/Windows: Returns empty list
pub async fn collect_all_workers() -> Result<Vec<ProcessStats>> {
    ProcessMonitor::enumerate_all().await
}

/// Collect telemetry for a specific worker group
///
/// # Arguments
/// - `group`: Service group (e.g., "llm", "vllm", "comfy")
///
/// # Returns
/// Vector of ProcessStats for all instances in the group
pub async fn collect_group(group: &str) -> Result<Vec<ProcessStats>> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        
        let group_path = format!("/sys/fs/cgroup/rbee.slice/{}", group);
        if !std::path::Path::new(&group_path).exists() {
            return Ok(Vec::new());
        }

        let mut stats = Vec::new();
        for instance_entry in fs::read_dir(&group_path)? {
            let instance_entry = instance_entry?;
            let instance_name = instance_entry.file_name().to_string_lossy().to_string();

            if !instance_entry.path().is_dir() {
                continue;
            }

            if let Ok(stat) = ProcessMonitor::collect_stats(group, &instance_name).await {
                stats.push(stat);
            }
        }

        Ok(stats)
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = group; // Suppress unused warning
        Ok(Vec::new())
    }
}

/// Collect telemetry for a specific worker instance
///
/// # Arguments
/// - `group`: Service group (e.g., "llm")
/// - `instance`: Instance identifier (e.g., "8080")
///
/// # Returns
/// ProcessStats for the specific instance
pub async fn collect_instance(group: &str, instance: &str) -> Result<ProcessStats> {
    ProcessMonitor::collect_stats(group, instance).await
}
