//! Heartbeat HTTP endpoints
//!
//! TEAM-362: Hive telemetry with worker details
//! TEAM-363: Cleaned up deprecated code per RULE ZERO

use axum::{extract::State, http::StatusCode, Json};
use hive_contract::HiveHeartbeat; // TEAM-284: Hive heartbeat types
use queen_rbee_hive_registry::HiveRegistry; // TEAM-284: Hive registry
use queen_rbee_worker_registry::WorkerRegistry; // TEAM-262: Renamed
use serde::{Deserialize, Serialize}; // TEAM-288: Added Deserialize for HeartbeatEvent
use std::sync::Arc;
use tokio::sync::broadcast; // TEAM-288: Broadcast channel for real-time events
use worker_contract::WorkerHeartbeat; // TEAM-284: Worker heartbeat types

// TEAM-362: Worker telemetry
use rbee_hive_monitor::ProcessStats;

// TEAM-363: Clean HeartbeatEvent enum (RULE ZERO compliant)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HeartbeatEvent {
    /// TEAM-362: Hive telemetry with worker details
    HiveTelemetry {
        hive_id: String,
        timestamp: String,
        workers: Vec<ProcessStats>,
    },
    
    /// Queen's own heartbeat (sent every 2.5 seconds)
    Queen {
        workers_online: usize,
        workers_available: usize,
        hives_online: usize,
        hives_available: usize,
        worker_ids: Vec<String>,
        hive_ids: Vec<String>,
        timestamp: String,
    },
}

/// State for the heartbeat endpoint
///
/// TEAM-362: WorkerRegistry populated from Hive telemetry
#[derive(Clone)]
pub struct HeartbeatState {
    /// Worker registry (populated from Hive telemetry)
    pub worker_registry: Arc<WorkerRegistry>,

    /// Hive registry (stores hive info + workers)
    pub hive_registry: Arc<HiveRegistry>,

    /// Broadcast channel for real-time events
    pub event_tx: broadcast::Sender<HeartbeatEvent>,
}

/// HTTP response for heartbeat acknowledgement
#[derive(Serialize)]
pub struct HttpHeartbeatAcknowledgement {
    /// Status of the heartbeat processing (e.g., "ok", "error")
    pub status: String,
    /// Human-readable message about the heartbeat result
    pub message: String,
}

/// POST /v1/hive-heartbeat - Handle hive telemetry
///
/// TEAM-362: Receives hive telemetry with worker details every 1s
pub async fn handle_hive_heartbeat(
    State(state): State<HeartbeatState>,
    Json(heartbeat): Json<HiveHeartbeat>,
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    eprintln!(
        "🐝 Hive telemetry: hive_id={}, workers={}",
        heartbeat.hive.id, heartbeat.workers.len()
    );

    // TEAM-362: Store hive info
    state.hive_registry.update_hive(heartbeat.clone());
    
    // TEAM-362: Store worker telemetry for scheduling
    state.hive_registry.update_workers(&heartbeat.hive.id, heartbeat.workers.clone());

    // TEAM-362: Broadcast telemetry event to SSE stream
    let event = HeartbeatEvent::HiveTelemetry {
        hive_id: heartbeat.hive.id.clone(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        workers: heartbeat.workers,
    };
    let _ = state.event_tx.send(event);

    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Telemetry received from hive {}", heartbeat.hive.id),
    }))
}
