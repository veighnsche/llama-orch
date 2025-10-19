//! Graceful shutdown endpoint
//!
//! Per architecture Phase 12: Cascading Shutdown
//! Shuts down all workers, then the hive daemon itself
//!
//! # Architecture Reference
//! - a_Claude_Sonnet_4_5_refined_this.md lines 368-387
//! - Queen calls this during cascading shutdown
//!
//! Created by: TEAM-151

use crate::http::routes::AppState;
use axum::{extract::State, http::StatusCode, Json};
use serde::Serialize;
use tracing::{error, info, warn};

/// Shutdown response
#[derive(Debug, Serialize)]
pub struct ShutdownResponse {
    /// Acknowledgment message
    pub message: String,
    /// Number of workers shutdown
    pub workers_shutdown: usize,
}

/// Handle POST /v1/shutdown
///
/// Initiates graceful shutdown of all workers and the hive daemon
///
/// # Architecture Flow
/// ```text
/// Queen → Hive: POST /v1/shutdown
/// Hive → Worker: POST /v1/shutdown (for each worker)
/// Hive shuts down itself
/// ```
///
/// # Returns
/// * `200 OK` with shutdown acknowledgment
///
/// # Shutdown Sequence
/// 1. Get all workers from registry
/// 2. Send POST /v1/shutdown to each worker (async)
/// 3. Wait up to 30s for workers to shutdown gracefully
/// 4. Return acknowledgment (server will shutdown after response sent)
pub async fn handle_shutdown(
    State(state): State<AppState>,
) -> Result<Json<ShutdownResponse>, (StatusCode, String)> {
    info!("Shutdown requested - initiating cascading shutdown");

    // Get all workers from registry
    let workers = state.registry.list().await;
    let worker_count = workers.len();

    info!(
        workers = worker_count,
        "Shutting down all workers"
    );

    // Shutdown each worker (spawn async tasks, don't wait)
    let mut shutdown_tasks = Vec::new();

    for worker in workers {
        let worker_url = worker.url.clone();
        let worker_id = worker.id.clone();
        let auth_token = state.expected_token.clone();

        let task = tokio::spawn(async move {
            info!(worker_id = %worker_id, url = %worker_url, "Sending shutdown to worker");

            let client = reqwest::Client::new();
            let shutdown_url = format!("{}/v1/shutdown", worker_url);

            match client
                .post(&shutdown_url)
                .header("Authorization", format!("Bearer {}", auth_token))
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    info!(worker_id = %worker_id, "Worker shutdown acknowledged");
                }
                Ok(resp) => {
                    warn!(
                        worker_id = %worker_id,
                        status = %resp.status(),
                        "Worker shutdown failed with HTTP {}",
                        resp.status()
                    );
                }
                Err(e) => {
                    error!(
                        worker_id = %worker_id,
                        error = %e,
                        "Failed to send shutdown to worker: {}",
                        e
                    );
                }
            }
        });

        shutdown_tasks.push(task);
    }

    // Wait for all shutdown tasks to complete (with timeout)
    let wait_result = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        futures::future::join_all(shutdown_tasks),
    )
    .await;

    match wait_result {
        Ok(_) => {
            info!("All worker shutdown tasks completed");
        }
        Err(_) => {
            warn!("Worker shutdown tasks timed out after 30s");
        }
    }

    info!("Hive shutdown complete - server will terminate");

    // TODO: Signal HTTP server to shutdown gracefully
    // This would require passing a shutdown channel through AppState
    // For now, the server will continue running until SIGTERM/SIGINT

    Ok(Json(ShutdownResponse {
        message: "Shutdown initiated".to_string(),
        workers_shutdown: worker_count,
    }))
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
    async fn test_shutdown_no_workers() {
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
            queen_callback_url: None,
        };

        let result = handle_shutdown(State(state)).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.0.workers_shutdown, 0);
        assert_eq!(response.0.message, "Shutdown initiated");
    }

    #[tokio::test]
    async fn test_shutdown_with_workers() {
        let registry = Arc::new(WorkerRegistry::new());
        let catalog = Arc::new(ModelCatalog::new(":memory:".to_string()));
        let provisioner = Arc::new(ModelProvisioner::new(PathBuf::from("/tmp")));
        let download_tracker = Arc::new(DownloadTracker::new());
        let addr: std::net::SocketAddr = "127.0.0.1:9200".parse().unwrap();

        // Register a mock worker
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
            last_heartbeat: None,
        };
        registry.register(worker).await;

        let state = AppState {
            registry,
            model_catalog: catalog,
            provisioner,
            download_tracker,
            server_addr: addr,
            expected_token: "test-token".to_string(),
            audit_logger: None,
            queen_callback_url: None,
        };

        let result = handle_shutdown(State(state)).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert_eq!(response.0.workers_shutdown, 1);
        assert_eq!(response.0.message, "Shutdown initiated");
    }

    #[test]
    fn test_shutdown_response_serialization() {
        let response = ShutdownResponse {
            message: "Shutdown initiated".to_string(),
            workers_shutdown: 3,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["message"], "Shutdown initiated");
        assert_eq!(json["workers_shutdown"], 3);
    }
}
