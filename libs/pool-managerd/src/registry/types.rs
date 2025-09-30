//! Types shared across registry submodules.
//!
//! Status: IMPLEMENTED (used by registry.rs update() method).
//! TODO: Add VRAM fields (vram_total_bytes, vram_free_bytes) if needed for partial updates.
//! TODO: Add compute_capability if needed for partial updates.

#[derive(Debug, Clone)]
pub struct UpdateFields {
    pub engine_version: Option<String>,
    pub device_mask: Option<String>,
    pub slots_total: Option<i32>,
    pub slots_free: Option<i32>,
    pub perf_hints: Option<serde_json::Value>,
}
