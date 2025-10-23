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

/// POST /v1/heartbeat - Handle hive heartbeat
///
/// TEAM-186: Changed to use registry instead of catalog
/// TEAM-186: Added new hive discovery check
///
/// **Flow:**
/// 1. Check if hive is new (not in registry)
/// 2. If new → trigger new hive discovery workflow
/// 3. Update registry (RAM) with full hive state
/// 4. Return acknowledgement
///
/// **Note**: No catalog updates! Catalog is for configuration only.
/// Heartbeat data (workers, status, timestamp) goes to registry (RAM).
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    // TEAM-186: Check if this is a new hive (not in registry)
    let is_new_hive = state.hive_registry.get_hive_state(&payload.hive_id).is_none();

    if is_new_hive {
        // TEAM-186: New hive discovered! Trigger discovery workflow
        handle_new_hive_discovery(&state, &payload).await?;
    }

    // TEAM-186: Update registry with full hive state (workers, timestamp, everything!)
    state.hive_registry.update_hive_state(&payload.hive_id, payload.clone());

    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Heartbeat received from {}", payload.hive_id),
    }))
}

/// Handle new hive discovery workflow
///
/// TEAM-186: Added new hive discovery workflow
///
/// **Triggered when**: A hive sends heartbeat but isn't in registry
async fn handle_new_hive_discovery(
    _state: &HeartbeatState,
    payload: &HiveHeartbeatPayload,
) -> Result<(), (StatusCode, String)> {
    // TEAM-186: New hive discovery workflow
    // /**
    //  * first check if the hive already exist in the catalog
    //  * if it does, do nothing the registry happens in the update_hive_state
    //  * if it doesn't, then say that the hive is not recognized by the queen and alert
    //  */
    // TEAM-186: Log discovery
    eprintln!("🆕 New hive discovered: {}", payload.hive_id);

    Ok(())
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
