//! Heartbeat endpoint handler
//!
//! Receives heartbeats from workers to track their health status.
//!
//! # Architecture
//!
//! Worker → Hive: POST /v1/heartbeat (this endpoint)
//!   - Updates worker registry with last_heartbeat timestamp
//!
//! Hive → Queen: Periodic task (see shared heartbeat crate)
//!   - Aggregates ALL worker states from registry
//!   - Sends to queen every 15s
//!
//! Created by: TEAM-115
//! Modified by: TEAM-151 (removed immediate relay, using periodic task instead)
//! Modified by: TEAM-159 (consolidated logic into shared crate)

use axum::{extract::State, http::StatusCode, Json};
use tracing::debug;

use crate::http::routes::AppState;

// Re-export types from shared heartbeat crate for consistency
pub use rbee_heartbeat::{HealthStatus, WorkerHeartbeatPayload as HeartbeatRequest};

// TEAM-159: Re-export response type from shared crate
pub use rbee_heartbeat::HeartbeatResponse;

/// Handle POST /v1/heartbeat
///
/// Receives heartbeat from worker and updates registry.
/// 
/// # Arguments
/// * `state` - Application state (registry, etc.)
/// * `payload` - Heartbeat payload from worker
///
/// # Returns
/// * `200 OK` - Heartbeat received
/// * `404 Not Found` - Worker not found in registry
///
/// # Architecture Reference
/// Per Phase 10 (a_Claude_Sonnet_4_5_refined_this.md lines 300-313):
/// - Worker → Hive: POST /v1/heartbeat (this endpoint)
/// - Hive updates registry
/// - Hive → Queen: Periodic task aggregates ALL workers (see shared heartbeat crate)
///
/// TEAM-159: Now uses shared heartbeat logic from rbee-heartbeat crate
pub async fn handle_heartbeat(
    State(state): State<AppState>,
    Json(payload): Json<HeartbeatRequest>,
) -> Result<(StatusCode, Json<HeartbeatResponse>), (StatusCode, String)> {
    debug!(
        worker_id = %payload.worker_id,
        timestamp = %payload.timestamp,
        "Received worker heartbeat"
    );

    // TEAM-159: Use shared heartbeat handler
    rbee_heartbeat::handle_worker_heartbeat(state.registry, payload)
        .await
        .map(|resp| (StatusCode::OK, Json(resp)))
        .map_err(|e| match e {
            rbee_heartbeat::hive_receiver::HeartbeatError::WorkerNotFound(id) => {
                (StatusCode::NOT_FOUND, format!("Worker {} not found in registry", id))
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provisioner::ModelProvisioner;
    use crate::registry::{WorkerInfo, WorkerRegistry, WorkerState};
    use model_catalog::ModelCatalog;
    use rbee_hive::download_tracker::DownloadTracker;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::SystemTime;

    #[tokio::test]
    async fn test_heartbeat_success() {
        let registry = Arc::new(WorkerRegistry::new());
        let catalog = Arc::new(ModelCatalog::new(":memory:".to_string()));
        let provisioner = Arc::new(ModelProvisioner::new(PathBuf::from("/tmp")));
        let download_tracker = Arc::new(DownloadTracker::new());
        let addr: std::net::SocketAddr = "127.0.0.1:9200".parse().unwrap();

        let state = AppState {
            registry: registry.clone(),
            model_catalog: catalog,
            provisioner,
            download_tracker,
            server_addr: addr,
            expected_token: "test-token".to_string(),
            audit_logger: None,
        };

        // Register a worker
        let worker = WorkerInfo {
            id: "worker-123".to_string(),
            url: "http://localhost:8081".to_string(),
            model_ref: "test-model".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            state: WorkerState::Idle,
            last_activity: SystemTime::now(),
            slots_total: 1,
            slots_available: 1,
            failed_health_checks: 0,
            pid: Some(12345),
            restart_count: 0,
            last_restart: None,
        };
        registry.register(worker).await;

        // Send heartbeat
        let payload = HeartbeatRequest {
            worker_id: "worker-123".to_string(),
            timestamp: "2025-10-19T00:00:00Z".to_string(),
            health_status: HealthStatus::Healthy,
        };

        let result = handle_heartbeat(State(state), Json(payload)).await;
        assert!(result.is_ok());
        let (status, _response) = result.unwrap();
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_heartbeat_unknown_worker() {
        let registry = Arc::new(WorkerRegistry::new());
        let catalog = Arc::new(ModelCatalog::new(":memory:".to_string()));
        let provisioner = Arc::new(ModelProvisioner::new(PathBuf::from("/tmp")));
        let download_tracker = Arc::new(DownloadTracker::new());
        let addr: std::net::SocketAddr = "127.0.0.1:9200".parse().unwrap();

        let state = AppState {
            registry,
            model_catalog: catalog,
            provisioner,
            download_tracker,
            server_addr: addr,
            expected_token: "test-token".to_string(),
            audit_logger: None,
        };

        // Send heartbeat for non-existent worker
        let payload = HeartbeatRequest {
            worker_id: "unknown-worker".to_string(),
            timestamp: "2025-10-19T00:00:00Z".to_string(),
            health_status: HealthStatus::Healthy,
        };

        let result = handle_heartbeat(State(state), Json(payload)).await;
        assert!(result.is_err());
        let (status, _msg) = result.unwrap_err();
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
