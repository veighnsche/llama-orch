//! Heartbeat endpoint for hive health monitoring
//!
//! Created by: TEAM-158
//! Modified by: TEAM-159 (consolidated logic into shared crate)
//!
//! Receives heartbeats from rbee-hive instances and triggers device detection
//! on first heartbeat.
//!
//! Uses the shared rbee-heartbeat crate for types and logic.

use axum::{extract::State, http::StatusCode, Json};
use queen_rbee_hive_catalog::HiveCatalog;
use rbee_heartbeat::{HeartbeatAcknowledgement, HiveHeartbeatPayload};
use std::sync::Arc;

use crate::http::device_detector::HttpDeviceDetector;

/// Shared state for heartbeat endpoint
#[derive(Clone)]
pub struct HeartbeatState {
    pub hive_catalog: Arc<HiveCatalog>,
    // TEAM-159: Device detector for first heartbeat flow
    pub device_detector: Arc<HttpDeviceDetector>,
}

/// Handle POST /heartbeat
///
/// TEAM-158: Receives heartbeat from hive and triggers device detection on first heartbeat
/// TEAM-159: Now uses shared heartbeat logic from rbee-heartbeat crate
///
/// Uses HiveHeartbeatPayload from rbee-heartbeat crate
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<HeartbeatAcknowledgement>, (StatusCode, String)> {
    // TEAM-159: Use shared heartbeat handler
    rbee_heartbeat::handle_hive_heartbeat(
        state.hive_catalog,
        payload,
        state.device_detector,
    )
    .await
    .map(Json)
    .map_err(|e| match e {
        rbee_heartbeat::queen_receiver::HeartbeatError::HiveNotFound(id) => {
            (StatusCode::NOT_FOUND, format!("Hive {} not found", id))
        }
        rbee_heartbeat::queen_receiver::HeartbeatError::DeviceDetection(msg) => {
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Device detection failed: {}", msg))
        }
        _ => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use queen_rbee_hive_catalog::{HiveRecord, HiveStatus};
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_heartbeat_updates_timestamp() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let catalog = Arc::new(HiveCatalog::new(&db_path).await.unwrap());

        // Add a hive
        let now_ms = chrono::Utc::now().timestamp_millis();
        let hive = HiveRecord {
            id: "test-hive".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8600,
            ssh_host: None,
            ssh_port: None,
            ssh_user: None,
            status: HiveStatus::Online,
            last_heartbeat_ms: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };
        catalog.add_hive(hive).await.unwrap();

        let state = HeartbeatState {
            hive_catalog: catalog.clone(),
            device_detector: Arc::new(HttpDeviceDetector::new()),
        };

        // Send heartbeat
        let now = chrono::Utc::now();
        let payload = HiveHeartbeatPayload {
            hive_id: "test-hive".to_string(),
            timestamp: now.to_rfc3339(),
            workers: vec![],
        };

        let result = handle_heartbeat(State(state), Json(payload)).await;
        assert!(result.is_ok());
        let ack = result.unwrap().0;
        assert!(ack.acknowledged);

        // Verify heartbeat was updated
        let updated_hive = catalog.get_hive("test-hive").await.unwrap().unwrap();
        assert!(updated_hive.last_heartbeat_ms.is_some());
    }

    #[tokio::test]
    async fn test_heartbeat_unknown_hive() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.db");

        let catalog = Arc::new(HiveCatalog::new(&db_path).await.unwrap());

        let state = HeartbeatState {
            hive_catalog: catalog,
            device_detector: Arc::new(HttpDeviceDetector::new()),
        };

        // Send heartbeat for non-existent hive
        let payload = HiveHeartbeatPayload {
            hive_id: "unknown-hive".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            workers: vec![],
        };

        let result = handle_heartbeat(State(state), Json(payload)).await;
        assert!(result.is_err());
        let (status, _msg) = result.unwrap_err();
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
