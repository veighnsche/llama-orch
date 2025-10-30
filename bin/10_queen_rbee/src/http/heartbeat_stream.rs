//! SSE endpoint for live heartbeat updates
//!
//! TEAM-362: Streams hive telemetry with worker details
//! TEAM-363: Cleaned up per RULE ZERO

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
};
use futures::stream::Stream;
use std::convert::Infallible;
use std::time::Duration;
use tokio::time::interval;

use super::heartbeat::{HeartbeatEvent, HeartbeatState};

/// GET /v1/heartbeats/stream - SSE endpoint for live heartbeat updates
///
/// TEAM-362: Streams Queen heartbeats (2.5s) and Hive telemetry (1s)
pub async fn handle_heartbeat_stream(
    State(state): State<HeartbeatState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // TEAM-288: Subscribe to broadcast channel for real-time events
    let mut event_rx = state.event_tx.subscribe();

    // TEAM-288: Create interval for queen's own heartbeat (every 2.5 seconds)
    let mut queen_interval = interval(Duration::from_millis(2500));
    queen_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // TEAM-362: Stream Queen heartbeats and Hive telemetry events
    let stream = async_stream::stream! {
        loop {
            tokio::select! {
                // Queen's own heartbeat every 2.5 seconds
                _ = queen_interval.tick() => {
                    let event = create_queen_heartbeat(&state);
                    let json = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
                    yield Ok(Event::default().event("heartbeat").data(json));
                }

                // Forward all events (HiveTelemetry from hives)
                Ok(event) = event_rx.recv() => {
                    let json = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
                    yield Ok(Event::default().event("heartbeat").data(json));
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Create queen's own heartbeat with current system status
fn create_queen_heartbeat(state: &HeartbeatState) -> HeartbeatEvent {
    let workers_online = state.worker_registry.count_online();
    let workers_available = state.worker_registry.count_available();
    let hives_online = state.hive_registry.count_online();
    let hives_available = state.hive_registry.count_available();

    let worker_ids: Vec<String> =
        state.worker_registry.list_online_workers().into_iter().map(|w| w.id).collect();

    let hive_ids: Vec<String> =
        state.hive_registry.list_online_hives().into_iter().map(|h| h.id).collect();

    HeartbeatEvent::Queen {
        workers_online,
        workers_available,
        hives_online,
        hives_available,
        worker_ids,
        hive_ids,
        timestamp: chrono::Utc::now().to_rfc3339(),
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

        let state = HeartbeatState { worker_registry, hive_registry };

        let snapshot = create_snapshot(&state);

        assert_eq!(snapshot.workers_online, 1);
        assert_eq!(snapshot.hives_online, 1);
        assert_eq!(snapshot.worker_ids, vec!["worker-1"]);
        assert_eq!(snapshot.hive_ids, vec!["hive-1"]);
    }
}
