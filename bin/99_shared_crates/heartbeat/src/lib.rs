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
//! **Category:** Protocol
//! **Pattern:** Command Pattern with Traits
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! This crate provides heartbeat logic for workers:
//! - **Worker:** Sends heartbeats to queen (Worker → Queen)
//!
//! # Architecture (TEAM-261)
//!
//! ```text
//! Worker → Queen: POST /v1/worker-heartbeat (30s interval)
//!   Payload: { worker_id, timestamp, health_status }
//! ```
//!
//! # Module Structure
//!
//! - `types` - Payload types and enums
//! - `worker` - Worker → Queen heartbeat logic
//!
//! # Example Usage
//!
//! **Worker:**
//! ```no_run
//! use rbee_heartbeat::worker::{WorkerHeartbeatConfig, start_worker_heartbeat_task};
//!
//! let config = WorkerHeartbeatConfig::new(
//!     "worker-123".to_string(),
//!     "http://localhost:8500".to_string(),  // Queen endpoint
//! );
//! let handle = start_worker_heartbeat_task(config);
//! ```
//!
//! Created by: TEAM-115
//! Extended by: TEAM-151 (hive aggregation)
//! Refactored by: TEAM-151 (modular structure)
//! Simplified by: TEAM-261 (removed hive aggregation)
//! Cleaned by: TEAM-262 (removed obsolete hive logic)

// Modules
// TEAM-262: Removed hive, hive_receiver, queen_receiver modules (obsolete after TEAM-261)
pub mod queen;
pub mod traits; // TEAM-159: Trait abstractions for receivers
pub mod types;
pub mod worker;

// ============================================================================
// Re-exports for Convenience
// ============================================================================
// TEAM-262: Simplified after TEAM-261 removed hive heartbeat aggregation

// Re-export commonly used types
pub use types::{HealthStatus, WorkerHeartbeatPayload};

// Re-export worker heartbeat functionality
pub use worker::{start_worker_heartbeat_task, WorkerHeartbeatConfig};

// TEAM-158: Re-export queen heartbeat functionality
// TEAM-262: Removed HeartbeatHandler - no longer needed
pub use queen::HeartbeatAcknowledgement;

// TEAM-159: Re-export traits
pub use traits::{
    CatalogError, CpuDevice, DeviceBackend, DeviceCapabilities, DeviceDetector, DeviceResponse,
    GpuDevice, HiveCatalog, HiveRecord, HiveStatus, WorkerRegistry,
};

// ============================================================================
// HTTP ENDPOINT WRAPPER (optional feature)
// ============================================================================
// TEAM-262: Removed after TEAM-261 - queen handles worker heartbeats directly
