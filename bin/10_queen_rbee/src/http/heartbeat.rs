//! Hive heartbeat HTTP endpoint
//!
//! TEAM-186: Heartbeat handling for queen-rbee
//!
//! **Simplified heartbeat:**
//! - Verifies hive exists in catalog
//! - Updates heartbeat timestamp
//! - Returns acknowledgement

use axum::{extract::State, http::StatusCode, Json};
use queen_rbee_hive_catalog::HiveCatalog;
use rbee_heartbeat::HiveHeartbeatPayload;
use serde::Serialize;
use std::sync::Arc;

/// State for the heartbeat endpoint
#[derive(Clone)]
pub struct HeartbeatState {
    /// Catalog of registered hives
    pub hive_catalog: Arc<HiveCatalog>,
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
/// **Flow:**
/// 1. Verify hive exists in catalog
/// 2. Update heartbeat timestamp
/// 3. Return acknowledgement
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    // Verify hive exists
    state
        .hive_catalog
        .get_hive(&payload.hive_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("Hive {} not found", payload.hive_id),
            )
        })?;

    // Update heartbeat timestamp
    let timestamp_ms = chrono::DateTime::parse_from_rfc3339(&payload.timestamp)
        .map(|dt| dt.timestamp_millis())
        .unwrap_or_else(|_| chrono::Utc::now().timestamp_millis());

    state
        .hive_catalog
        .update_heartbeat(&payload.hive_id, timestamp_ms)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Heartbeat received from {}", payload.hive_id),
    }))
}
