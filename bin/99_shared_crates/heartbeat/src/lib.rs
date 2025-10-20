// TEAM-135: Migrated from llm-worker-rbee by TEAM-135
// Original implementation: TEAM-115
// Purpose: Generic heartbeat protocol for health monitoring
// NOTE: Moved to shared-crates because BOTH workers AND hives send heartbeats
// TEAM-151: Extended to support hive aggregation and queen receiving
// TEAM-151: Refactored into modular structure

#![warn(missing_docs)]
#![warn(clippy::all)]

//! Heartbeat mechanism for health monitoring across the rbee system
//!
//! This crate provides heartbeat logic for all three binaries:
//! - **Worker:** Sends heartbeats to hive (Worker → Hive)
//! - **Hive:** Collects worker heartbeats + sends aggregated heartbeats to queen (Hive → Queen)
//! - **Queen:** Receives aggregated heartbeats from hives
//!
//! # Architecture
//!
//! ```text
//! Worker → Hive: POST /v1/heartbeat (30s interval)
//!   Payload: { worker_id, timestamp, health_status }
//!
//! Hive → Queen: POST /v1/heartbeat (15s interval)
//!   Payload: { hive_id, timestamp, workers: [...] }
//!   (aggregates ALL worker states from registry)
//! ```
//!
//! # Module Structure
//!
//! - `types` - Payload types and enums
//! - `worker` - Worker → Hive heartbeat logic
//! - `hive` - Hive → Queen heartbeat logic
//!
//! # Example Usage
//!
//! **Worker:**
//! ```no_run
//! use rbee_heartbeat::worker::{WorkerHeartbeatConfig, start_worker_heartbeat_task};
//!
//! let config = WorkerHeartbeatConfig::new(
//!     "worker-123".to_string(),
//!     "http://localhost:8600".to_string(),
//! );
//! let handle = start_worker_heartbeat_task(config);
//! ```
//!
//! **Hive:**
//! ```no_run
//! use rbee_heartbeat::hive::{HiveHeartbeatConfig, start_hive_heartbeat_task};
//! // (also implement WorkerStateProvider trait)
//! ```
//!
//! Created by: TEAM-115
//! Extended by: TEAM-151 (hive aggregation)
//! Refactored by: TEAM-151 (modular structure)

// Modules
pub mod types;
pub mod worker;
pub mod hive;

// ============================================================================
// Re-exports for Convenience
// ============================================================================

// Re-export commonly used types
pub use types::{
    HealthStatus, HiveHeartbeatPayload, WorkerHeartbeatPayload, WorkerState,
};

// Re-export worker heartbeat functionality
pub use worker::{start_worker_heartbeat_task, WorkerHeartbeatConfig};

// Re-export hive heartbeat functionality
pub use hive::{start_hive_heartbeat_task, HiveHeartbeatConfig, WorkerStateProvider};
