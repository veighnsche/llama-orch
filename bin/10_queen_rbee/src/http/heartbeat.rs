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
use serde::{Deserialize, Serialize}; // TEAM-288: Added Deserialize for HeartbeatEvent
use std::sync::Arc;
use tokio::sync::broadcast; // TEAM-288: Broadcast channel for real-time events
use worker_contract::WorkerHeartbeat; // TEAM-284: Worker heartbeat types

/// TEAM-288: Heartbeat event types for real-time broadcasting
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HeartbeatEvent {
    /// Worker heartbeat received
    Worker {
        worker_id: String,
        status: String,
        timestamp: String,
    },
    /// Hive heartbeat received
    Hive {
        hive_id: String,
        status: String,
        timestamp: String,
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
/// TEAM-186: Changed from HiveCatalog to HiveRegistry
/// TEAM-262: Renamed to WorkerRegistry
/// TEAM-284: Added HiveRegistry
/// TEAM-288: Added broadcast channel for real-time event streaming
#[derive(Clone)]
pub struct HeartbeatState {
    /// Registry for runtime worker state (RAM)
    /// TEAM-262: Renamed to WorkerRegistry
    pub worker_registry: Arc<WorkerRegistry>,
    
    /// Registry for runtime hive state (RAM)
    /// TEAM-284: Added hive registry
    pub hive_registry: Arc<HiveRegistry>,
    
    /// TEAM-288: Broadcast channel for real-time heartbeat events
    /// All heartbeats (worker, hive, queen) are broadcast here
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

    // TEAM-288: Broadcast worker heartbeat event for real-time streaming
    let event = HeartbeatEvent::Worker {
        worker_id: heartbeat.worker.id.clone(),
        status: format!("{:?}", heartbeat.worker.status),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    let _ = state.event_tx.send(event); // Ignore error if no subscribers

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

    // TEAM-288: Broadcast hive heartbeat event for real-time streaming
    let event = HeartbeatEvent::Hive {
        hive_id: heartbeat.hive.id.clone(),
        status: format!("{:?}", heartbeat.hive.operational_status),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    let _ = state.event_tx.send(event); // Ignore error if no subscribers

    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Heartbeat received from hive {}", heartbeat.hive.id),
    }))
}
