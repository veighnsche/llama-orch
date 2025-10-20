//! Heartbeat endpoint for hive health monitoring
//!
//! Created by: TEAM-158
//!
//! Receives heartbeats from rbee-hive instances and triggers device detection
//! on first heartbeat.
//!
//! Uses the shared rbee-heartbeat crate for types and patterns.

use axum::{extract::State, http::StatusCode, Json};
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::{HiveCatalog, HiveStatus};
use rbee_heartbeat::{HeartbeatAcknowledgement, HiveHeartbeatPayload};
use serde::Deserialize;
use std::sync::Arc;

const ACTOR_QUEEN_HEARTBEAT: &str = "👑 queen-heartbeat";
const ACTION_HEARTBEAT: &str = "heartbeat";
const ACTION_DEVICE_DETECTION: &str = "device_detection";
const ACTION_ERROR: &str = "error";

/// Device detection response from hive
#[derive(Debug, Deserialize)]
pub struct DeviceResponse {
    pub cpu: CpuInfo,
    pub gpus: Vec<GpuInfo>,
    pub models: usize,
    pub workers: usize,
}

#[derive(Debug, Deserialize)]
pub struct CpuInfo {
    pub cores: u32,
    pub ram_gb: u32,
}

#[derive(Debug, Deserialize)]
pub struct GpuInfo {
    pub id: String,
    pub name: String,
    pub vram_gb: u32,
}

/// Shared state for heartbeat endpoint
#[derive(Clone)]
pub struct HeartbeatState {
    pub hive_catalog: Arc<HiveCatalog>,
}

/// Handle POST /heartbeat
///
/// TEAM-158: Receives heartbeat from hive and triggers device detection on first heartbeat
/// Uses HiveHeartbeatPayload from rbee-heartbeat crate
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<HeartbeatAcknowledgement>, (StatusCode, String)> {
    // TEAM-158: Parse timestamp to milliseconds for catalog
    let timestamp_ms = chrono::DateTime::parse_from_rfc3339(&payload.timestamp)
        .map(|dt| dt.timestamp_millis())
        .unwrap_or_else(|_| chrono::Utc::now().timestamp_millis());

    // TEAM-158: Update heartbeat in catalog
    state
        .hive_catalog
        .update_heartbeat(&payload.hive_id, timestamp_ms)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // TEAM-158: Check if this is first heartbeat (ONLY narrate on first heartbeat)
    let hive = state
        .hive_catalog
        .get_hive(&payload.hive_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or_else(|| (StatusCode::NOT_FOUND, "Hive not found".to_string()))?;

    // TEAM-158: If first heartbeat (status is Unknown), trigger device detection
    if matches!(hive.status, HiveStatus::Unknown) {
        Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_HEARTBEAT, &payload.hive_id)
            .human(format!("First heartbeat from {}. Checking capabilities...", payload.hive_id))
            .emit();

        // TEAM-158: Request device detection from hive
        let hive_url = format!("http://{}:{}/v1/devices", hive.host, hive.port);

        Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_DEVICE_DETECTION, &payload.hive_id)
            .human(format!(
                "Unknown capabilities of beehive {}. Asking the beehive to detect devices",
                payload.hive_id
            ))
            .emit();

        let client = reqwest::Client::new();
        let response = client.get(&hive_url).send().await.map_err(|e| {
            Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_ERROR, &payload.hive_id)
                .human(format!("Failed to request device detection: {}", e))
                .error_kind("device_detection_failed")
                .emit();
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to request device detection: {}", e),
            )
        })?;

        let devices: DeviceResponse = response.json().await.map_err(|e| {
            Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_ERROR, &payload.hive_id)
                .human(format!("Failed to parse device response: {}", e))
                .error_kind("device_parse_failed")
                .emit();
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to parse device response: {}", e))
        })?;

        // TEAM-158: Build device summary for narration
        let gpu_summary = if devices.gpus.is_empty() {
            "no GPUs".to_string()
        } else {
            devices
                .gpus
                .iter()
                .map(|gpu| format!("{} {} ({}GB)", gpu.id, gpu.name, gpu.vram_gb))
                .collect::<Vec<_>>()
                .join(", ")
        };

        Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_DEVICE_DETECTION, &payload.hive_id)
            .human(format!(
                "The beehive {} has cpu ({} cores, {}GB RAM), {}, model catalog has {} models, {} workers available",
                payload.hive_id,
                devices.cpu.cores,
                devices.cpu.ram_gb,
                gpu_summary,
                devices.models,
                devices.workers
            ))
            .emit();

        // TEAM-158: Update hive status to Online
        state.hive_catalog.update_hive_status(&payload.hive_id, HiveStatus::Online).await.map_err(
            |e| {
                Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_ERROR, &payload.hive_id)
                    .human(format!("Failed to update hive status: {}", e))
                    .error_kind("status_update_failed")
                    .emit();
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to update hive status: {}", e))
            },
        )?;

        Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_DEVICE_DETECTION, &payload.hive_id)
            .human(format!("Hive {} is now online", payload.hive_id))
            .emit();
    }

    Ok(Json(HeartbeatAcknowledgement::success()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use queen_rbee_hive_catalog::HiveRecord;
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

        let state = HeartbeatState { hive_catalog: catalog.clone() };

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

        let state = HeartbeatState { hive_catalog: catalog };

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
