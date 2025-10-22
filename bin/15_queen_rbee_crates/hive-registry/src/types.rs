//! Type definitions for hive registry
//!
//! Runtime state structures for tracking hive information in memory.
//!
//! TEAM-221: Investigated 2025-10-22 - All types documented

use serde::{Deserialize, Serialize};

/// Runtime state for a single hive (in-memory only)
///
/// This tracks the current state of a hive based on heartbeats.
/// Lost on restart - rebuilt from incoming heartbeats.
#[derive(Debug, Clone)]
pub struct HiveRuntimeState {
    /// Hive ID
    pub hive_id: String,

    /// All workers currently running on this hive
    pub workers: Vec<WorkerInfo>,

    /// Last heartbeat timestamp (milliseconds since epoch)
    pub last_heartbeat_ms: i64,

    /// Total VRAM used by all workers (GB)
    pub vram_used_gb: f32,

    /// Total RAM used by all workers (GB)
    pub ram_used_gb: f32,

    /// Number of active workers
    pub worker_count: usize,
}

/// Worker information from heartbeat
///
/// Complete worker info for queen's scheduling decisions.
/// This serves as the worker registry - all worker info is here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Worker ID
    pub worker_id: String,

    /// Worker state ("Idle", "Busy", "Loading")
    pub state: String,

    /// Last heartbeat from worker
    pub last_heartbeat: String,

    /// Health status ("healthy", "degraded")
    pub health_status: String,

    /// Worker URL for direct inference (e.g., "http://localhost:9300")
    /// This is the ONLY place queen stores worker URLs
    pub url: String,

    /// Model loaded on this worker
    pub model_id: Option<String>,

    /// Backend type (e.g., "cpu", "cuda", "metal")
    pub backend: Option<String>,

    /// Device ID (e.g., GPU index)
    pub device_id: Option<u32>,

    /// VRAM used by this worker (bytes)
    pub vram_bytes: Option<u64>,

    /// RAM used by this worker (bytes)
    pub ram_bytes: Option<u64>,

    /// CPU usage percentage (0-100)
    pub cpu_percent: Option<f32>,

    /// GPU usage percentage (0-100)
    pub gpu_percent: Option<f32>,
}

/// Resource information for a hive
///
/// Used for scheduling decisions.
#[derive(Debug, Clone)]
pub struct ResourceInfo {
    /// Total VRAM used by all workers (GB)
    pub vram_used_gb: f32,

    /// Total RAM used by all workers (GB)
    pub ram_used_gb: f32,

    /// Number of active workers
    pub worker_count: usize,
}

impl HiveRuntimeState {
    /// Create a new runtime state from heartbeat data
    pub fn from_heartbeat(hive_id: String, workers: Vec<WorkerInfo>, timestamp_ms: i64) -> Self {
        let worker_count = workers.len();
        let (vram_used_gb, ram_used_gb) = calculate_resources(&workers);

        Self {
            hive_id,
            workers,
            last_heartbeat_ms: timestamp_ms,
            vram_used_gb,
            ram_used_gb,
            worker_count,
        }
    }

    /// Get resource information
    pub fn resource_info(&self) -> ResourceInfo {
        ResourceInfo {
            vram_used_gb: self.vram_used_gb,
            ram_used_gb: self.ram_used_gb,
            worker_count: self.worker_count,
        }
    }

    /// Check if heartbeat is recent
    pub fn is_recent(&self, max_age_ms: i64) -> bool {
        let now = chrono::Utc::now().timestamp_millis();
        now - self.last_heartbeat_ms < max_age_ms
    }
}

/// Calculate resource usage from worker list
///
/// For now, uses placeholder values.
/// Future: Calculate from actual worker metadata.
fn calculate_resources(workers: &[WorkerInfo]) -> (f32, f32) {
    // Placeholder calculation
    // Assume each worker uses 4GB VRAM and 2GB RAM
    let vram_used = workers.len() as f32 * 4.0;
    let ram_used = workers.len() as f32 * 2.0;
    (vram_used, ram_used)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hive_runtime_state_from_heartbeat() {
        let workers = vec![
            WorkerInfo {
                worker_id: "worker-1".to_string(),
                state: "Idle".to_string(),
                last_heartbeat: "2025-10-21T10:00:00Z".to_string(),
                health_status: "healthy".to_string(),
                url: "http://localhost:9300".to_string(),
                model_id: Some("llama-3-8b".to_string()),
                backend: Some("cuda".to_string()),
                device_id: Some(0),
                vram_bytes: Some(8_000_000_000),
                ram_bytes: Some(2_000_000_000),
                cpu_percent: Some(15.0),
                gpu_percent: Some(25.0),
            },
            WorkerInfo {
                worker_id: "worker-2".to_string(),
                state: "Busy".to_string(),
                last_heartbeat: "2025-10-21T10:00:00Z".to_string(),
                health_status: "healthy".to_string(),
                url: "http://localhost:9301".to_string(),
                model_id: Some("llama-3-8b".to_string()),
                backend: Some("cuda".to_string()),
                device_id: Some(1),
                vram_bytes: Some(8_000_000_000),
                ram_bytes: Some(2_000_000_000),
                cpu_percent: Some(20.0),
                gpu_percent: Some(75.0),
            },
        ];

        let state = HiveRuntimeState::from_heartbeat(
            "localhost".to_string(),
            workers.clone(),
            1729504800000,
        );

        assert_eq!(state.hive_id, "localhost");
        assert_eq!(state.worker_count, 2);
        assert_eq!(state.workers.len(), 2);
        assert_eq!(state.vram_used_gb, 8.0); // 2 workers * 4GB
        assert_eq!(state.ram_used_gb, 4.0); // 2 workers * 2GB
    }

    #[test]
    fn test_resource_info() {
        let state = HiveRuntimeState {
            hive_id: "test".to_string(),
            workers: vec![],
            last_heartbeat_ms: 1729504800000,
            vram_used_gb: 16.0,
            ram_used_gb: 8.0,
            worker_count: 4,
        };

        let info = state.resource_info();
        assert_eq!(info.vram_used_gb, 16.0);
        assert_eq!(info.ram_used_gb, 8.0);
        assert_eq!(info.worker_count, 4);
    }

    #[test]
    fn test_is_recent_true() {
        let now = chrono::Utc::now().timestamp_millis();
        let state = HiveRuntimeState {
            hive_id: "test".to_string(),
            workers: vec![],
            last_heartbeat_ms: now - 10_000, // 10 seconds ago
            vram_used_gb: 0.0,
            ram_used_gb: 0.0,
            worker_count: 0,
        };

        assert!(state.is_recent(30_000)); // Within 30 seconds
    }

    #[test]
    fn test_is_recent_false() {
        let now = chrono::Utc::now().timestamp_millis();
        let state = HiveRuntimeState {
            hive_id: "test".to_string(),
            workers: vec![],
            last_heartbeat_ms: now - 60_000, // 60 seconds ago
            vram_used_gb: 0.0,
            ram_used_gb: 0.0,
            worker_count: 0,
        };

        assert!(!state.is_recent(30_000)); // Not within 30 seconds
    }

    #[test]
    fn test_calculate_resources_empty() {
        let (vram, ram) = calculate_resources(&[]);
        assert_eq!(vram, 0.0);
        assert_eq!(ram, 0.0);
    }

    #[test]
    fn test_calculate_resources_multiple_workers() {
        let workers = vec![
            WorkerInfo {
                worker_id: "w1".to_string(),
                state: "Idle".to_string(),
                last_heartbeat: "2025-10-21T10:00:00Z".to_string(),
                health_status: "healthy".to_string(),
                url: "http://localhost:9300".to_string(),
                model_id: None,
                backend: None,
                device_id: None,
                vram_bytes: None,
                ram_bytes: None,
                cpu_percent: None,
                gpu_percent: None,
            },
            WorkerInfo {
                worker_id: "w2".to_string(),
                state: "Busy".to_string(),
                last_heartbeat: "2025-10-21T10:00:00Z".to_string(),
                health_status: "healthy".to_string(),
                url: "http://localhost:9301".to_string(),
                model_id: None,
                backend: None,
                device_id: None,
                vram_bytes: None,
                ram_bytes: None,
                cpu_percent: None,
                gpu_percent: None,
            },
            WorkerInfo {
                worker_id: "w3".to_string(),
                state: "Loading".to_string(),
                last_heartbeat: "2025-10-21T10:00:00Z".to_string(),
                health_status: "degraded".to_string(),
                url: "http://localhost:9302".to_string(),
                model_id: None,
                backend: None,
                device_id: None,
                vram_bytes: None,
                ram_bytes: None,
                cpu_percent: None,
                gpu_percent: None,
            },
        ];

        let (vram, ram) = calculate_resources(&workers);
        assert_eq!(vram, 12.0); // 3 workers * 4GB
        assert_eq!(ram, 6.0); // 3 workers * 2GB
    }
}
