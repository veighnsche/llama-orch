//! Internal pool entry structure for Registry.
//!
//! Status: IMPLEMENTED (used by registry.rs).
//! TODO: Add VRAM fields (vram_total_bytes, vram_free_bytes) per CHECKLIST.md "Registry & Contracts".
//! TODO: Add compute_capability field for placement heuristics.
//! TODO: Consider adding perf_tokens_per_s and first_token_ms for steady-state perf hints.

use crate::health::HealthStatus;

#[derive(Debug, Clone)]
pub struct PoolEntry {
    pub health: HealthStatus,
    pub last_heartbeat_ms: Option<i64>,
    pub version: Option<String>,
    pub last_error: Option<String>,
    pub active_leases: i32,
    pub engine_version: Option<String>,
    pub engine_digest: Option<String>,
    pub engine_catalog_id: Option<String>,
    pub device_mask: Option<String>,
    pub slots_total: Option<i32>,
    pub slots_free: Option<i32>,
    pub perf_hints: Option<serde_json::Value>,
    pub draining: bool,
}
