//! Typed snapshot exported from Registry for orchestrator-core and placement.
//!
//! Status: IMPLEMENTED (used by registry.rs snapshots() method).
//! TODO: Populate vram_total_bytes, vram_free_bytes, compute_capability from device discovery.
//! TODO: Populate perf_hints from steady-state benchmarks or EWMA of recent decode times.
//! Integration: Consumed by orchestratord placement logic (not yet implemented).

use crate::health::HealthStatus;

#[derive(Debug, Clone, PartialEq)]
pub struct PoolSnapshot {
    pub pool_id: String,
    pub health: HealthStatus,
    pub engine_version: Option<String>,
    pub device_mask: Option<String>,
    pub slots_total: Option<i32>,
    pub slots_free: Option<i32>,
    pub vram_total_bytes: Option<u64>,
    pub vram_free_bytes: Option<u64>,
    pub compute_capability: Option<String>,
    pub perf_hints: Option<serde_json::Value>,
    pub draining: bool,
}
