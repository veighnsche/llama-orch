//! Worker management endpoints
//!
//! Per test-001-mvp.md Phase 5: Worker Startup
//! - POST /v1/workers/spawn - Spawn a new worker
//! - POST /v1/workers/ready - Worker ready callback
//! - GET /v1/workers/list - List all workers
//!
//! Created by: TEAM-026
//! Modified by: TEAM-027

use crate::registry::{WorkerInfo, WorkerRegistry, WorkerState};
use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tracing::{error, info};
use uuid::Uuid;

/// Spawn worker request
#[derive(Debug, Deserialize)]
pub struct SpawnWorkerRequest {
    /// Model reference (e.g., "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    pub model_ref: String,
    /// Backend (e.g., "metal", "cuda", "cpu")
    pub backend: String,
    /// Device ID
    pub device: u32,
    /// Model file path (from catalog)
    pub model_path: String,
}

/// Spawn worker response
#[derive(Debug, Serialize)]
pub struct SpawnWorkerResponse {
    /// Worker ID (UUID)
    pub worker_id: String,
    /// Worker URL
    pub url: String,
    /// Current state
    pub state: String,
}

/// Worker ready callback request
///
/// Per test-001-mvp.md lines 148-157
#[derive(Debug, Deserialize)]
pub struct WorkerReadyRequest {
    /// Worker ID
    pub worker_id: String,
    /// Worker URL
    pub url: String,
    /// Model reference
    pub model_ref: String,
    /// Backend
    pub backend: String,
    /// Device ID
    pub device: u32,
}

/// Worker ready callback response
#[derive(Debug, Serialize)]
pub struct WorkerReadyResponse {
    /// Acknowledgment message
    pub message: String,
}

/// List workers response
#[derive(Debug, Serialize)]
pub struct ListWorkersResponse {
    /// List of workers
    pub workers: Vec<WorkerInfo>,
}

/// Handle POST /v1/workers/spawn
///
/// Spawns a new worker process
pub async fn handle_spawn_worker(
    State(registry): State<Arc<WorkerRegistry>>,
    Json(request): Json<SpawnWorkerRequest>,
) -> Result<Json<SpawnWorkerResponse>, (StatusCode, String)> {
    info!(
        model_ref = %request.model_ref,
        backend = %request.backend,
        device = request.device,
        "Spawning worker"
    );

    // TEAM-027: Generate worker ID
    let worker_id = format!("worker-{}", Uuid::new_v4());

    // TEAM-027: Determine port (simple allocation: start at 8081)
    // For MVP, use simple sequential allocation
    // Production should use proper port allocation or OS-assigned ports
    let workers = registry.list().await;
    let port = 8081 + workers.len() as u16;
    
    // TEAM-027: Get hostname for URL
    let hostname = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "localhost".to_string());
    let url = format!("http://{}:{}", hostname, port);

    // TEAM-027: Get worker binary path (same directory as rbee-hive)
    let worker_binary = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("llm-worker-rbee")))
        .unwrap_or_else(|| std::path::PathBuf::from("llm-worker-rbee"));

    // TEAM-027: Generate API key
    let api_key = format!("key-{}", Uuid::new_v4());

    // TEAM-027: Callback URL (this server's address)
    // For MVP, assume we're listening on the same hostname
    let callback_url = format!("http://{}:8080/v1/workers/ready", hostname);

    // Spawn worker process
    // Per test-001-mvp.md lines 136-143
    let spawn_result = tokio::process::Command::new(worker_binary)
        .arg("--worker-id")
        .arg(&worker_id)
        .arg("--model")
        .arg(&request.model_path)
        .arg("--backend")
        .arg(&request.backend)
        .arg("--device")
        .arg(request.device.to_string())
        .arg("--port")
        .arg(port.to_string())
        .arg("--api-key")
        .arg(&api_key)
        .arg("--callback-url")
        .arg(&callback_url)
        .spawn();

    match spawn_result {
        Ok(_child) => {
            // Register worker in loading state
            let worker = WorkerInfo {
                id: worker_id.clone(),
                url: url.clone(),
                model_ref: request.model_ref,
                backend: request.backend,
                device: request.device,
                state: WorkerState::Loading,
                last_activity: SystemTime::now(),
                slots_total: 1,
                slots_available: 0,
            };

            registry.register(worker).await;

            info!(
                worker_id = %worker_id,
                url = %url,
                "Worker spawned successfully"
            );

            Ok(Json(SpawnWorkerResponse {
                worker_id,
                url,
                state: "loading".to_string(),
            }))
        }
        Err(e) => {
            error!(error = %e, "Failed to spawn worker");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to spawn worker: {}", e),
            ))
        }
    }
}

/// Handle POST /v1/workers/ready
///
/// Worker ready callback - worker reports it's ready to accept requests
pub async fn handle_worker_ready(
    State(registry): State<Arc<WorkerRegistry>>,
    Json(request): Json<WorkerReadyRequest>,
) -> Json<WorkerReadyResponse> {
    info!(
        worker_id = %request.worker_id,
        url = %request.url,
        "Worker ready callback received"
    );

    // Update worker state to idle
    registry
        .update_state(&request.worker_id, WorkerState::Idle)
        .await;

    Json(WorkerReadyResponse {
        message: "Worker registered as ready".to_string(),
    })
}

/// Handle GET /v1/workers/list
///
/// List all workers
pub async fn handle_list_workers(
    State(registry): State<Arc<WorkerRegistry>>,
) -> Json<ListWorkersResponse> {
    let workers = registry.list().await;

    Json(ListWorkersResponse { workers })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_worker_request_deserialization() {
        let json = r#"{
            "model_ref": "hf:test/model",
            "backend": "cpu",
            "device": 0,
            "model_path": "/models/test.gguf"
        }"#;

        let request: SpawnWorkerRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model_ref, "hf:test/model");
        assert_eq!(request.backend, "cpu");
        assert_eq!(request.device, 0);
    }

    #[test]
    fn test_worker_ready_request_deserialization() {
        let json = r#"{
            "worker_id": "worker-123",
            "url": "http://localhost:8081",
            "model_ref": "hf:test/model",
            "backend": "cpu",
            "device": 0
        }"#;

        let request: WorkerReadyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.worker_id, "worker-123");
        assert_eq!(request.url, "http://localhost:8081");
    }
}
