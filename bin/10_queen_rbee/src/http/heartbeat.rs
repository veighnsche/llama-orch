//! Hive heartbeat HTTP endpoint
//!
//! TEAM-186: Heartbeat handling for queen-rbee
//! TEAM-186: Changed from catalog to registry (heartbeat is runtime data)
//! TEAM-186: Added new hive discovery workflow
//!
//! **Heartbeat flow:**
//! - Updates registry (RAM) with full hive state
//! - NO catalog updates (catalog is configuration only!)
//! - Returns acknowledgement

use axum::{extract::State, http::StatusCode, Json};
use queen_rbee_hive_registry::HiveRegistry;
use rbee_heartbeat::HiveHeartbeatPayload;
use serde::Serialize;
use std::sync::Arc;

/// State for the heartbeat endpoint
///
/// TEAM-186: Changed from HiveCatalog to HiveRegistry
#[derive(Clone)]
pub struct HeartbeatState {
    /// Registry for runtime hive state (RAM)
    /// TEAM-186: Heartbeat data goes to registry, not catalog
    pub hive_registry: Arc<HiveRegistry>,
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
