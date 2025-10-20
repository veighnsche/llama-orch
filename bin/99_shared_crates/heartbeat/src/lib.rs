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
//! This crate provides heartbeat logic for all three binaries:
//! - **Worker:** Sends heartbeats to hive (Worker → Hive)
//! - **Hive:** Collects worker heartbeats + sends aggregated heartbeats to queen (Hive → Queen)
//! - **Queen:** Receives aggregated heartbeats from hives
//!
//! # Interface
//!
//! ## Entrypoints
//! ```rust
//! // Queen receives hive heartbeats
//! pub async fn handle_hive_heartbeat<C, D>(
//!     catalog: Arc<C>,
//!     payload: HiveHeartbeatPayload,
//!     device_detector: Arc<D>,
//! ) -> Result<HeartbeatAcknowledgement, HeartbeatError>
//!
//! // Hive receives worker heartbeats
//! pub async fn handle_worker_heartbeat<R>(
//!     registry: Arc<R>,
//!     payload: WorkerHeartbeatPayload,
//! ) -> Result<HeartbeatResponse, HeartbeatError>
//! ```
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
pub mod hive;
pub mod hive_receiver; // TEAM-159: Hive worker heartbeat receiver
pub mod queen;
pub mod queen_receiver; // TEAM-159: Queen hive heartbeat receiver
pub mod traits; // TEAM-159: Trait abstractions for receivers
pub mod types;
pub mod worker;

// ============================================================================
// Re-exports for Convenience
// ============================================================================

// Re-export commonly used types
pub use types::{HealthStatus, HiveHeartbeatPayload, WorkerHeartbeatPayload, WorkerState};

// Re-export worker heartbeat functionality
pub use worker::{start_worker_heartbeat_task, WorkerHeartbeatConfig};

// Re-export hive heartbeat functionality
pub use hive::{start_hive_heartbeat_task, HiveHeartbeatConfig, WorkerStateProvider};

// TEAM-158: Re-export queen heartbeat functionality
pub use queen::{HeartbeatAcknowledgement, HeartbeatHandler};

// TEAM-159: Re-export receiver functionality
pub use hive_receiver::{handle_worker_heartbeat, HeartbeatResponse};
pub use queen_receiver::handle_hive_heartbeat;

// TEAM-159: Re-export traits
pub use traits::{
    CatalogError, CpuDevice, DeviceBackend, DeviceCapabilities, DeviceDetector, DeviceResponse,
    GpuDevice, HiveCatalog, HiveRecord, HiveStatus, WorkerRegistry,
};

// ============================================================================
// HTTP ENDPOINT WRAPPER (optional feature)
// ============================================================================

#[cfg(feature = "http")]
use axum::{extract::State, http::StatusCode, Json};

#[cfg(feature = "http")]
#[derive(Clone)]
pub struct HeartbeatState<D: DeviceDetector> {
    pub hive_catalog: std::sync::Arc<dyn HiveCatalog>,
    pub device_detector: std::sync::Arc<D>,
}

/// POST /heartbeat - HTTP endpoint handler
///
/// TEAM-164: Thin HTTP wrapper around handle_hive_heartbeat()
#[cfg(feature = "http")]
pub async fn http_handle_heartbeat<D: DeviceDetector + Send + Sync + 'static>(
    State(state): State<HeartbeatState<D>>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<queen_receiver::HeartbeatAcknowledgement>, (StatusCode, String)> {
    handle_hive_heartbeat(state.hive_catalog, payload, state.device_detector)
        .await
        .map(Json)
        .map_err(|e| match e {
            queen_receiver::HeartbeatError::HiveNotFound(id) => {
                (StatusCode::NOT_FOUND, format!("Hive {} not found", id))
            }
            queen_receiver::HeartbeatError::DeviceDetection(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Device detection failed: {}", msg))
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        })
}
