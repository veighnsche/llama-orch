//! Health and readiness (planning-only).
//!
//! Status: IMPLEMENTED (used by registry and BDD tests).
//! Spec: ORCH-3002 (live=true when manager up; ready=true only after preload+health check).
//! Usage: Registry stores HealthStatus per pool; orchestratord queries via control API.
//! Integration: Used by registry.rs, orchestratord/api/control.rs, test-harness/bdd.
//! Tests: Unit tests exist (OC-POOL-3001, registry.rs:464-480), BDD tests use this (pool_manager.rs:3).
//! TODO: Consider adding optional fields (last_check_at, consecutive_failures) for richer health tracking.
//! TODO: If no expansion needed, this file is complete and can stay as-is.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthStatus {
    pub live: bool,
    pub ready: bool,
}
