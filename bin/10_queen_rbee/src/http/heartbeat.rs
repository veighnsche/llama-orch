//! Heartbeat HTTP endpoints
//!
//! TEAM-186: Heartbeat handling for queen-rbee
//! TEAM-186: Changed from catalog to registry (heartbeat is runtime data)
//! TEAM-186: Added new hive discovery workflow
//! TEAM-217: Investigated Oct 22, 2025 - Behavior inventory complete
//! TEAM-261: Added worker heartbeat endpoint (workers send directly to queen)
//!
//! **Heartbeat flows:**
//! 1. Hive heartbeat (DEPRECATED - will be removed)
//! 2. Worker heartbeat (NEW - workers send directly to queen)

use axum::{extract::State, http::StatusCode, Json};
use queen_rbee_worker_registry::WorkerRegistry; // TEAM-262: Renamed
use rbee_heartbeat::WorkerHeartbeatPayload; // TEAM-262: Removed HiveHeartbeatPayload
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// State for the heartbeat endpoint
///
/// TEAM-186: Changed from HiveCatalog to HiveRegistry
/// TEAM-262: Renamed to WorkerRegistry
#[derive(Clone)]
pub struct HeartbeatState {
    /// Registry for runtime worker state (RAM)
    /// TEAM-186: Heartbeat data goes to registry, not catalog
    /// TEAM-262: Renamed to WorkerRegistry (field name kept for compatibility)
    pub hive_registry: Arc<WorkerRegistry>,
}

/// HTTP response for heartbeat acknowledgement
#[derive(Serialize)]
pub struct HttpHeartbeatAcknowledgement {
    /// Status of the heartbeat processing (e.g., "ok", "error")
    pub status: String,
    /// Human-readable message about the heartbeat result
    pub message: String,
}

// ============================================================================
// WORKER HEARTBEAT ENDPOINT (TEAM-261)
// ============================================================================
//
// Workers send heartbeats DIRECTLY to queen (not through hive)
// This simplifies architecture and makes queen the single source of truth
//

/// POST /v1/worker-heartbeat - Handle worker heartbeat
///
/// TEAM-261: Workers send heartbeats directly to queen
///
/// **Flow:**
/// 1. Receive heartbeat from worker
/// 2. Update worker registry with status
/// 3. Track last seen timestamp
/// 4. Return acknowledgement
///
/// **Note:** This bypasses hive - workers talk directly to queen
pub async fn handle_worker_heartbeat(
    State(_state): State<HeartbeatState>,
    Json(payload): Json<WorkerHeartbeatPayload>,
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    // TEAM-261: Update worker state in registry
    // TODO: Implement worker registry update
    // For now, just log and acknowledge

    eprintln!(
        "💓 Worker heartbeat: worker_id={}, timestamp={}, health_status={:?}",
        payload.worker_id, payload.timestamp, payload.health_status
    );

    // TODO: Add to worker registry:
    // state.worker_registry.update_worker_state(&payload.worker_id, payload.clone());

    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Heartbeat received from worker {}", payload.worker_id),
    }))
}
