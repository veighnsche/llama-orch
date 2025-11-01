//! Heartbeat HTTP endpoints
//!
//! TEAM-362: Hive telemetry with worker details
//! TEAM-363: Cleaned up deprecated code per RULE ZERO

use axum::{extract::State, http::StatusCode, Json};
 // TEAM-284: Hive heartbeat types
use observability_narration_core::n; // TEAM-373: Narration for hive ready callback
use queen_rbee_telemetry_registry::TelemetryRegistry; // TEAM-374: Telemetry registry
use serde::{Deserialize, Serialize}; // TEAM-288: Added Deserialize for HeartbeatEvent
use std::sync::Arc;
use tokio::sync::broadcast; // TEAM-288: Broadcast channel for real-time events
 // TEAM-284: Worker heartbeat types

// TEAM-362: Worker telemetry
use rbee_hive_monitor::ProcessStats;

// TEAM-363: Clean HeartbeatEvent enum (RULE ZERO compliant)
/// Heartbeat events broadcast to SSE clients
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HeartbeatEvent {
    /// TEAM-362: Hive telemetry with worker details
    HiveTelemetry {
        /// Hive identifier
        hive_id: String,
        /// Timestamp of telemetry
        timestamp: String,
        /// Worker process stats
        workers: Vec<ProcessStats>,
    },
    
    /// Queen's own heartbeat (sent every 2.5 seconds)
    Queen {
        /// Number of workers online
        workers_online: usize,
        /// Number of workers available for work
        workers_available: usize,
        /// Number of hives online
        hives_online: usize,
        /// Number of hives available
        hives_available: usize,
        /// List of worker IDs
        worker_ids: Vec<String>,
        /// List of hive IDs
        hive_ids: Vec<String>,
        /// Timestamp of heartbeat
        timestamp: String,
    },
}

/// State for the heartbeat endpoint
///
/// TEAM-362: WorkerRegistry populated from Hive telemetry
/// TEAM-374: Updated to use TelemetryRegistry
#[derive(Clone)]
pub struct HeartbeatState {
    /// Worker registry (populated from Hive telemetry)
    /// TEAM-374: Now uses TelemetryRegistry
    pub worker_registry: Arc<TelemetryRegistry>,

    /// Hive registry (stores hive info + workers)
    /// TEAM-374: Now uses TelemetryRegistry
    pub hive_registry: Arc<TelemetryRegistry>,

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

// TEAM-374: DELETED handle_hive_heartbeat() - replaced by SSE subscription
// Old POST-based continuous telemetry receiver is deprecated.
// Queen now subscribes to hive SSE streams (hive_subscriber.rs)

/// Hive ready callback payload
/// TEAM-373: Discovery callback from hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveReadyCallback {
    /// Hive identifier
    pub hive_id: String,
    /// Hive URL (e.g., "http://192.168.1.100:7835")
    pub hive_url: String,
}

/// POST /v1/hive/ready - Discovery callback from hive
///
/// TEAM-373: Changed from continuous telemetry to one-time callback.
/// When hive sends this, Queen subscribes to its SSE stream.
///
/// Discovery flow:
/// 1. Hive detects Queen (via GET /capabilities or startup)
/// 2. Hive sends POST /v1/hive/ready (one-time callback)
/// 3. Queen subscribes to GET /v1/heartbeats/stream on hive
/// 4. Continuous telemetry flows via SSE
pub async fn handle_hive_ready(
    State(state): State<HeartbeatState>,
    Json(callback): Json<HiveReadyCallback>,
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    eprintln!(
        "🐝 Hive ready callback: hive_id={}, url={}",
        callback.hive_id, callback.hive_url
    );
    
    // TEAM-373: Store initial hive info (register URL)
    // Note: HiveRegistry doesn't have register_hive_url yet, will be added
    // For now, the SSE subscription will populate the registry
    
    // TEAM-373: Start SSE subscription to this hive
    let _subscription_handle = crate::hive_subscriber::start_hive_subscription(
        callback.hive_url.clone(),
        callback.hive_id.clone(),
        state.hive_registry.clone(),
        state.event_tx.clone(),
    );
    
    n!("hive_ready", "✅ Hive {} ready, subscription started", callback.hive_id);
    
    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Subscribed to hive {} SSE stream", callback.hive_id),
    }))
}
