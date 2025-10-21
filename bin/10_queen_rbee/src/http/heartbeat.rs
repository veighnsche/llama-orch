//! Hive heartbeat HTTP endpoint
//!
//! TEAM-186: HTTP wrapper for heartbeat processing

use axum::{extract::State, http::StatusCode, Json};
use queen_rbee_hive_catalog::HiveCatalog;
use rbee_heartbeat::HiveHeartbeatPayload;
use serde::Serialize;
use std::sync::Arc;

use super::device_detector::HttpDeviceDetector;

#[derive(Clone)]
pub struct HeartbeatState {
    pub hive_catalog: Arc<HiveCatalog>,
    pub device_detector: Arc<HttpDeviceDetector>,
}

#[derive(Serialize)]
pub struct HttpHeartbeatAcknowledgement {
    pub status: String,
    pub message: String,
}

/// POST /v1/heartbeat - Handle hive heartbeat
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    // Call binary-specific heartbeat logic
    let response =
        crate::heartbeat::handle_hive_heartbeat(state.hive_catalog, payload, state.device_detector)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(HttpHeartbeatAcknowledgement {
        status: response.status,
        message: response.message,
    }))
}
