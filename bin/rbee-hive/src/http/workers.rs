//! Worker management endpoints
//!
//! Per test-001-mvp.md Phase 5: Worker Startup
//! - POST /v1/workers/spawn - Spawn a new worker
//! - POST /v1/workers/ready - Worker ready callback
//! - GET /v1/workers/list - List all workers
//!
//! TEAM-030: Worker registry is ephemeral, model catalog is persistent (SQLite)
//!
//! Created by: TEAM-026
//! Modified by: TEAM-027, TEAM-029, TEAM-030

use crate::http::routes::AppState;
use crate::registry::{WorkerInfo, WorkerState};
use axum::{extract::State, http::StatusCode, Json};
use model_catalog::ModelInfo;
use serde::{Deserialize, Serialize};
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
/// TEAM-029: Added model provisioning with catalog
pub async fn handle_spawn_worker(
    State(state): State<AppState>,
    Json(request): Json<SpawnWorkerRequest>,
) -> Result<Json<SpawnWorkerResponse>, (StatusCode, String)> {
    info!(
        model_ref = %request.model_ref,
        backend = %request.backend,
        device = request.device,
        "Spawning worker"
    );

    // TEAM-029: Phase 3 - Check model catalog and provision if needed
    let (provider, reference) = parse_model_ref(&request.model_ref)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid model_ref: {}", e)))?;

    info!("Checking model catalog for {}/{}", provider, reference);
    
    let model_path = match state
        .model_catalog
        .find_model(&reference, &provider)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Catalog error: {}", e)))?
    {
        Some(model_info) => {
            info!("Model found in catalog: {}", model_info.local_path);
            model_info.local_path
        }
        None => {
            info!("Model not found in catalog, provisioning...");
            
            // Download model
            let downloaded_path = state
                .provisioner
                .download_model(&reference, &provider)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Download failed: {}", e)))?;

            let path_str = downloaded_path.to_string_lossy().to_string();
            
            // Register in catalog
            let model_info = ModelInfo {
                reference: reference.clone(),
                provider: provider.clone(),
                local_path: path_str.clone(),
                size_bytes: state
                    .provisioner
                    .get_model_size(&downloaded_path)
                    .unwrap_or(0),
                downloaded_at: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64,
            };

            state
                .model_catalog
                .register_model(&model_info)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Catalog registration failed: {}", e)))?;

            info!("Model provisioned and registered: {}", path_str);
            path_str
        }
    };

    // TEAM-027: Generate worker ID
    let worker_id = format!("worker-{}", Uuid::new_v4());

    // TEAM-027: Determine port (simple allocation: start at 8081)
    // For MVP, use simple sequential allocation
    // Production should use proper port allocation or OS-assigned ports
    let workers = state.registry.list().await;
    let port = 8081 + workers.len() as u16;
    
    // TEAM-027: Get hostname for URL
    // TEAM-035: For localhost testing, use 127.0.0.1 to avoid hostname resolution issues
    let hostname = std::env::var("LLORCH_WORKER_HOST")
        .unwrap_or_else(|_| {
            hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok())
                .unwrap_or_else(|| "127.0.0.1".to_string())
        });
    let url = format!("http://{}:{}", hostname, port);

    // TEAM-027: Get worker binary path (same directory as rbee-hive)
    let worker_binary = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("llm-worker-rbee")))
        .unwrap_or_else(|| std::path::PathBuf::from("llm-worker-rbee"));

    // TEAM-027: Generate API key
    let api_key = format!("key-{}", Uuid::new_v4());

    // TEAM-027: Callback URL (this server's address)
    // TEAM-035: For localhost testing, use 127.0.0.1
    let callback_url = format!("http://{}:8080/v1/workers/ready", hostname);

    // Spawn worker process
    // Per test-001-mvp.md lines 136-143
    // TEAM-029: Use model_path from catalog/provisioner instead of request.model_path
    // TEAM-035: Worker only accepts: --worker-id, --model, --port, --callback-url
    let spawn_result = tokio::process::Command::new(worker_binary)
        .arg("--worker-id")
        .arg(&worker_id)
        .arg("--model")
        .arg(&model_path)  // TEAM-029: Use provisioned model path
        .arg("--port")
        .arg(port.to_string())
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

            state.registry.register(worker).await;

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

/// Parse model reference into provider and reference
/// TEAM-029: Helper function
fn parse_model_ref(model_ref: &str) -> Result<(String, String), String> {
    if let Some((provider, reference)) = model_ref.split_once(':') {
        Ok((provider.to_string(), reference.to_string()))
    } else {
        Err(format!("Invalid model_ref format: expected 'provider:reference', got '{}'", model_ref))
    }
}

/// Handle POST /v1/workers/ready
///
/// Worker ready callback - worker reports it's ready to accept requests
pub async fn handle_worker_ready(
    State(state): State<AppState>,
    Json(request): Json<WorkerReadyRequest>,
) -> Json<WorkerReadyResponse> {
    info!(
        worker_id = %request.worker_id,
        url = %request.url,
        "Worker ready callback received"
    );

    // Update worker state to idle
    state.registry
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
    State(state): State<AppState>,
) -> Json<ListWorkersResponse> {
    let workers = state.registry.list().await;

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

    // TEAM-031: Additional comprehensive tests
    #[test]
    fn test_parse_model_ref_valid() {
        let result = parse_model_ref("hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
        assert!(result.is_ok());
        let (provider, reference) = result.unwrap();
        assert_eq!(provider, "hf");
        assert_eq!(reference, "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
    }

    #[test]
    fn test_parse_model_ref_invalid() {
        let result = parse_model_ref("invalid-format");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid model_ref format"));
    }

    #[test]
    fn test_parse_model_ref_multiple_colons() {
        let result = parse_model_ref("hf:org:model:version");
        assert!(result.is_ok());
        let (provider, reference) = result.unwrap();
        assert_eq!(provider, "hf");
        assert_eq!(reference, "org:model:version");
    }

    #[test]
    fn test_spawn_worker_response_serialization() {
        let response = SpawnWorkerResponse {
            worker_id: "worker-123".to_string(),
            url: "http://localhost:8081".to_string(),
            state: "loading".to_string(),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["worker_id"], "worker-123");
        assert_eq!(json["url"], "http://localhost:8081");
        assert_eq!(json["state"], "loading");
    }

    #[test]
    fn test_worker_ready_response_serialization() {
        let response = WorkerReadyResponse {
            message: "Worker registered as ready".to_string(),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["message"], "Worker registered as ready");
    }

    #[test]
    fn test_list_workers_response_serialization() {
        use crate::registry::{WorkerInfo, WorkerState};
        use std::time::SystemTime;

        let worker = WorkerInfo {
            id: "worker-1".to_string(),
            url: "http://localhost:8081".to_string(),
            model_ref: "hf:test/model".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            state: WorkerState::Idle,
            last_activity: SystemTime::now(),
            slots_total: 1,
            slots_available: 1,
        };

        let response = ListWorkersResponse {
            workers: vec![worker],
        };

        let json = serde_json::to_value(&response).unwrap();
        assert!(json["workers"].is_array());
        assert_eq!(json["workers"].as_array().unwrap().len(), 1);
    }
}
