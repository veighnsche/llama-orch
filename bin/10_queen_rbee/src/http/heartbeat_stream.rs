//! SSE endpoint for live heartbeat updates
//!
//! TEAM-285: Real-time heartbeat streaming for web UI
//!
//! **Purpose:**
//! Stream live updates of all heartbeats (workers + hives) to connected clients.
//! This allows web UIs to show real-time status without polling.
//!
//! **Flow:**
//! 1. Client connects to GET /v1/heartbeats/stream
//! 2. Server sends initial snapshot of all heartbeats
//! 3. Server sends updates as new heartbeats arrive
//! 4. Client receives SSE events with JSON payloads

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    response::IntoResponse,
};
use futures::stream::{self, Stream};
use serde::Serialize;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use super::heartbeat::HeartbeatState;

/// Combined heartbeat status for SSE streaming
#[derive(Serialize, Clone)]
pub struct HeartbeatSnapshot {
    /// Timestamp of this snapshot
    pub timestamp: String,
    /// Number of online workers
    pub workers_online: usize,
    /// Number of available workers
    pub workers_available: usize,
    /// Number of online hives
    pub hives_online: usize,
    /// Number of available hives
    pub hives_available: usize,
    /// List of all worker IDs (online)
    pub worker_ids: Vec<String>,
    /// List of all hive IDs (online)
    pub hive_ids: Vec<String>,
}

/// GET /v1/heartbeats/stream - SSE endpoint for live heartbeat updates
///
/// TEAM-285: Real-time heartbeat streaming
///
/// **Flow:**
/// 1. Send initial snapshot
/// 2. Send updates every 5 seconds
/// 3. Keep connection alive with periodic pings
///
/// **SSE Format:**
/// ```text
/// event: heartbeat
/// data: {"timestamp":"2025-10-24T19:00:00Z","workers_online":3,"hives_online":1,...}
/// ```
pub async fn handle_heartbeat_stream(
    State(state): State<HeartbeatState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = stream::unfold(state, |state| async move {
        // Create snapshot
        let snapshot = create_snapshot(&state);
        
        // Serialize to JSON
        let json = serde_json::to_string(&snapshot).unwrap_or_else(|_| "{}".to_string());
        
        // Create SSE event
        let event = Event::default()
            .event("heartbeat")
            .data(json);
        
        // Wait 5 seconds before next update
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        Some((Ok(event), state))
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Create heartbeat snapshot from current registry state
fn create_snapshot(state: &HeartbeatState) -> HeartbeatSnapshot {
    let workers_online = state.worker_registry.count_online();
    let workers_available = state.worker_registry.count_available();
    let hives_online = state.hive_registry.count_online();
    let hives_available = state.hive_registry.count_available();
    
    let worker_ids: Vec<String> = state
        .worker_registry
        .list_online_workers()
        .into_iter()
        .map(|w| w.id)
        .collect();
    
    let hive_ids: Vec<String> = state
        .hive_registry
        .list_online_hives()
        .into_iter()
        .map(|h| h.id)
        .collect();
    
    HeartbeatSnapshot {
        timestamp: chrono::Utc::now().to_rfc3339(),
        workers_online,
        workers_available,
        hives_online,
        hives_available,
        worker_ids,
        hive_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hive_contract::{HiveHeartbeat, HiveInfo};
    use queen_rbee_hive_registry::HiveRegistry;
    use queen_rbee_worker_registry::WorkerRegistry;
    use shared_contract::{HealthStatus, OperationalStatus};
    use worker_contract::{WorkerHeartbeat, WorkerInfo, WorkerStatus};

    #[test]
    fn test_create_snapshot_empty() {
        let state = HeartbeatState {
            worker_registry: Arc::new(WorkerRegistry::new()),
            hive_registry: Arc::new(HiveRegistry::new()),
        };

        let snapshot = create_snapshot(&state);
        
        assert_eq!(snapshot.workers_online, 0);
        assert_eq!(snapshot.hives_online, 0);
        assert!(snapshot.worker_ids.is_empty());
        assert!(snapshot.hive_ids.is_empty());
    }

    #[test]
    fn test_create_snapshot_with_data() {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let hive_registry = Arc::new(HiveRegistry::new());

        // Add worker
        let worker = WorkerInfo {
            id: "worker-1".to_string(),
            model_id: "test-model".to_string(),
            device: "cpu:0".to_string(),
            port: 9301,
            status: WorkerStatus::Ready,
            implementation: "test".to_string(),
            version: "0.1.0".to_string(),
        };
        worker_registry.update_worker(WorkerHeartbeat::new(worker));

        // Add hive
        let hive = HiveInfo {
            id: "hive-1".to_string(),
            hostname: "localhost".to_string(),
            port: 9200,
            operational_status: OperationalStatus::Ready,
            health_status: HealthStatus::Healthy,
            version: "0.1.0".to_string(),
        };
        hive_registry.update_hive(HiveHeartbeat::new(hive));

        let state = HeartbeatState {
            worker_registry,
            hive_registry,
        };

        let snapshot = create_snapshot(&state);
        
        assert_eq!(snapshot.workers_online, 1);
        assert_eq!(snapshot.hives_online, 1);
        assert_eq!(snapshot.worker_ids, vec!["worker-1"]);
        assert_eq!(snapshot.hive_ids, vec!["hive-1"]);
    }
}
