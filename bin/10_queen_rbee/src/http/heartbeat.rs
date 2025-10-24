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
use hive_contract::HiveHeartbeat; // TEAM-284: Hive heartbeat types
use queen_rbee_hive_registry::HiveRegistry; // TEAM-284: Hive registry
use queen_rbee_worker_registry::WorkerRegistry; // TEAM-262: Renamed
use serde::Serialize; // TEAM-284: Removed unused Deserialize
use std::sync::Arc;
use worker_contract::WorkerHeartbeat; // TEAM-284: Worker heartbeat types

/// State for the heartbeat endpoint
///
/// TEAM-186: Changed from HiveCatalog to HiveRegistry
/// TEAM-262: Renamed to WorkerRegistry
/// TEAM-284: Added HiveRegistry
#[derive(Clone)]
pub struct HeartbeatState {
    /// Registry for runtime worker state (RAM)
    /// TEAM-262: Renamed to WorkerRegistry
    pub worker_registry: Arc<WorkerRegistry>,
    
    /// Registry for runtime hive state (RAM)
    /// TEAM-284: Added hive registry
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
    State(state): State<HeartbeatState>,
    Json(heartbeat): Json<WorkerHeartbeat>,  // TEAM-284: Changed to WorkerHeartbeat
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    // TEAM-261: Update worker state in registry
    // TEAM-284: Now properly implemented with worker-contract types

    eprintln!(
        "💓 Worker heartbeat: worker_id={}, status={:?}",
        heartbeat.worker.id, heartbeat.worker.status
    );

    // TEAM-284: Update worker registry
    state.worker_registry.update_worker(heartbeat.clone());

    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Heartbeat received from worker {}", heartbeat.worker.id),
    }))
}

// ============================================================================
// HIVE HEARTBEAT ENDPOINT (TEAM-284)
// ============================================================================
//
// Hives send heartbeats DIRECTLY to queen (mirrors worker pattern)
// This allows queen to track hive health and availability
//

/// POST /v1/hive-heartbeat - Handle hive heartbeat
///
/// TEAM-284: Hives send heartbeats directly to queen (mirrors worker pattern)
///
/// **Flow:**
/// 1. Receive heartbeat from hive
/// 2. Update hive registry with status
/// 3. Track last seen timestamp
/// 4. Return acknowledgement
///
/// **Note:** This mirrors the worker heartbeat pattern
pub async fn handle_hive_heartbeat(
    State(state): State<HeartbeatState>,
    Json(heartbeat): Json<HiveHeartbeat>,  // TEAM-284: Changed to HiveHeartbeat
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    // TEAM-284: Update hive state in registry (now properly implemented)

    eprintln!(
        "🐝 Hive heartbeat: hive_id={}, status={:?}",
        heartbeat.hive.id, heartbeat.hive.operational_status
    );

    // TEAM-284: Update hive registry
    state.hive_registry.update_hive(heartbeat.clone());

    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Heartbeat received from hive {}", heartbeat.hive.id),
    }))
}
