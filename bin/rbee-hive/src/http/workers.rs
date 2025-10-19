// TEAM-108 AUDIT: 41% of file reviewed (200/492 lines)
// Date: 2025-10-18
// Status: ✅ PASS - No blocking issues found
// Findings: Input validation applied (lines 88-96), port allocation logic (lines 159-177), proper error handling
// Issues: Minor - port allocation could use iteration counter

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
//! TEAM-087: Enhanced spawn diagnostics and error handling

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
/// TEAM-103: Added input validation
pub async fn handle_spawn_worker(
    State(state): State<AppState>,
    Json(request): Json<SpawnWorkerRequest>,
) -> Result<Json<SpawnWorkerResponse>, (StatusCode, String)> {
    // TEAM-103: Validate inputs before processing
    use input_validation::{validate_identifier, validate_model_ref};

    // Validate model reference
    validate_model_ref(&request.model_ref)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid model_ref: {}", e)))?;

    // Validate backend identifier
    validate_identifier(&request.backend, 64)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid backend: {}", e)))?;

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
            let downloaded_path =
                state.provisioner.download_model(&reference, &provider).await.map_err(|e| {
                    (StatusCode::INTERNAL_SERVER_ERROR, format!("Download failed: {}", e))
                })?;

            let path_str = downloaded_path.to_string_lossy().to_string();

            // Register in catalog
            let model_info = ModelInfo {
                reference: reference.clone(),
                provider: provider.clone(),
                local_path: path_str.clone(),
                size_bytes: state.provisioner.get_model_size(&downloaded_path).unwrap_or(0),
                downloaded_at: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64,
            };

            state.model_catalog.register_model(&model_info).await.map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Catalog registration failed: {}", e))
            })?;

            info!("Model provisioned and registered: {}", path_str);
            path_str
        }
    };

    // TEAM-027: Generate worker ID
    let worker_id = format!("worker-{}", Uuid::new_v4());

    // TEAM-096: Determine port - find first available port
    // Check existing workers to avoid address conflicts
    let workers = state.registry.list().await;
    let mut port = 8081u16;
    let used_ports: std::collections::HashSet<u16> = workers
        .iter()
        .filter_map(|w| {
            // Extract port from URL like "http://127.0.0.1:8081"
            w.url.split(':').last().and_then(|p| p.parse().ok())
        })
        .collect();

    // Find first unused port starting from 8081
    while used_ports.contains(&port) {
        port += 1;
        if port > 9000 {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "No available ports (8081-9000 all in use)".to_string(),
            ));
        }
    }

    info!("🔍 Port allocation: {} workers registered, using port {}", workers.len(), port);

    // TEAM-027: Get hostname for URL
    // TEAM-035: For localhost testing, use 127.0.0.1 to avoid hostname resolution issues
    // TEAM-090: Default to 127.0.0.1 for local workers to avoid DNS resolution issues
    // Use RBEE_WORKER_HOST env var to override for remote/distributed setups
    let hostname = std::env::var("RBEE_WORKER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
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
    // TEAM-091: Use actual server port from AppState instead of hardcoded 8080
    let callback_url = format!("http://{}:{}/v1/workers/ready", hostname, state.server_addr.port());

    // TEAM-115: Check memory availability before spawning worker
    use crate::resources::{check_memory_available, MemoryLimits};
    let memory_limits = MemoryLimits::default();
    // Estimate: 4GB per worker (conservative estimate for LLM)
    let estimated_memory = 4 * 1024 * 1024 * 1024;
    
    if let Err(e) = check_memory_available(estimated_memory, &memory_limits) {
        error!(
            worker_id = %worker_id,
            error = %e,
            "TEAM-115: Insufficient memory to spawn worker"
        );
        return Err((
            StatusCode::INSUFFICIENT_STORAGE,
            format!("Insufficient memory: {}", e),
        ));
    }

    // Spawn worker process
    // Per test-001-mvp.md lines 136-143
    // TEAM-029: Use model_path from catalog/provisioner instead of request.model_path
    // TEAM-035: Worker only accepts: --worker-id, --model, --port, --callback-url
    // TEAM-087: Enhanced spawn diagnostics

    info!("🚀 Spawning worker process:");
    info!("   Binary: {:?}", worker_binary);
    info!("   Worker ID: {}", worker_id);
    info!("   Model: {}", model_path);
    info!("   Port: {}", port);
    info!("   Callback: {}", callback_url);

    // TEAM-088: CRITICAL FIX - Inherit stdout/stderr so we can see worker narration!
    // Use RBEE_SILENT=1 to suppress logs if needed
    let (stdout_cfg, stderr_cfg) = if std::env::var("RBEE_SILENT").is_ok() {
        (std::process::Stdio::piped(), std::process::Stdio::piped())
    } else {
        (std::process::Stdio::inherit(), std::process::Stdio::inherit())
    };

    let spawn_result = tokio::process::Command::new(&worker_binary)
        .arg("--worker-id")
        .arg(&worker_id)
        .arg("--model")
        .arg(&model_path) // TEAM-029: Use provisioned model path
        .arg("--model-ref")
        .arg(&request.model_ref) // TEAM-092: Pass model_ref for callback
        .arg("--backend")
        .arg(&request.backend) // TEAM-092: Pass backend for callback
        .arg("--device")
        .arg(request.device.to_string()) // TEAM-092: Pass device for callback
        .arg("--port")
        .arg(port.to_string())
        .arg("--callback-url")
        .arg(&callback_url)
        .stdout(stdout_cfg)
        .stderr(stderr_cfg)
        .spawn();

    match spawn_result {
        Ok(mut child) => {
            // TEAM-087: Check if process started successfully
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            if let Ok(Some(status)) = child.try_wait() {
                // TEAM-088: Process exited immediately
                // If stdout/stderr are inherited, we already saw the output
                // If piped (RBEE_SILENT=1), capture and display
                let stdout = if let Some(mut out) = child.stdout.take() {
                    use tokio::io::AsyncReadExt;
                    let mut buf = String::new();
                    let _ = out.read_to_string(&mut buf).await;
                    if !buf.is_empty() {
                        error!("   stdout: {}", buf);
                    }
                    buf
                } else {
                    String::new()
                };

                let stderr = if let Some(mut err) = child.stderr.take() {
                    use tokio::io::AsyncReadExt;
                    let mut buf = String::new();
                    let _ = err.read_to_string(&mut buf).await;
                    if !buf.is_empty() {
                        error!("   stderr: {}", buf);
                    }
                    buf
                } else {
                    String::new()
                };

                error!("❌ Worker process exited immediately with status: {}", status);

                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Worker process failed to start: exit status {}. Check logs above for details.", status)
                ));
            }

            // TEAM-101: Store PID for force-kill and process liveness checks
            let pid = child.id(); // Returns Option<u32> in newer Tokio

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
                failed_health_checks: 0, // TEAM-096: Initialize counter
                pid,                // TEAM-101: Store PID for lifecycle management (Option<u32>)
                restart_count: 0,   // TEAM-103: Initialize restart counter
                last_restart: None, // TEAM-103: No restart yet
                last_heartbeat: None, // TEAM-115: No heartbeat yet
            };

            state.registry.register(worker).await;

            info!(
                worker_id = %worker_id,
                pid = pid,
                "TEAM-101: Worker PID stored for lifecycle management"
            );

            info!(
                worker_id = %worker_id,
                url = %url,
                "✅ Worker process spawned successfully"
            );

            Ok(Json(SpawnWorkerResponse { worker_id, url, state: "loading".to_string() }))
        }
        Err(e) => {
            error!("❌ Failed to spawn worker process: {}", e);
            error!("   Binary path: {:?}", worker_binary);
            error!("   Does the binary exist? Check: ls -la {:?}", worker_binary);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to spawn worker: {}. Binary: {:?}", e, worker_binary),
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
/// TEAM-103: Added input validation
pub async fn handle_worker_ready(
    State(state): State<AppState>,
    Json(request): Json<WorkerReadyRequest>,
) -> Result<Json<WorkerReadyResponse>, (StatusCode, String)> {
    // TEAM-103: Validate inputs
    use input_validation::{validate_identifier, validate_model_ref};

    // Validate worker ID
    validate_identifier(&request.worker_id, 256)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid worker_id: {}", e)))?;

    // Validate model reference
    validate_model_ref(&request.model_ref)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid model_ref: {}", e)))?;

    // Validate backend identifier
    validate_identifier(&request.backend, 64)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid backend: {}", e)))?;

    info!(
        worker_id = %request.worker_id,
        url = %request.url,
        "Worker ready callback received"
    );

    // Update worker state to idle
    state.registry.update_state(&request.worker_id, WorkerState::Idle).await;

    // TEAM-124: Notify queen-rbee that worker is ready (if callback URL configured)
    if let Some(ref queen_url) = state.queen_callback_url {
        let callback_payload = serde_json::json!({
            "worker_id": request.worker_id,
            "url": request.url,
            "model_ref": request.model_ref,
            "backend": request.backend,
        });

        info!(
            worker_id = %request.worker_id,
            queen_url = %queen_url,
            "Notifying queen-rbee of worker ready"
        );

        // Send async callback to queen-rbee (don't block on response)
        let queen_url_clone = queen_url.clone();
        let worker_id_clone = request.worker_id.clone();
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            match client
                .post(format!("{}/v2/workers/ready", queen_url_clone))
                .json(&callback_payload)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    info!(
                        worker_id = %worker_id_clone,
                        "✅ Queen-rbee notified successfully"
                    );
                }
                Ok(resp) => {
                    error!(
                        worker_id = %worker_id_clone,
                        status = %resp.status(),
                        "⚠️  Queen-rbee callback failed with HTTP {}",
                        resp.status()
                    );
                }
                Err(e) => {
                    error!(
                        worker_id = %worker_id_clone,
                        error = %e,
                        "❌ Failed to notify queen-rbee: {}",
                        e
                    );
                }
            }
        });
    }

    Ok(Json(WorkerReadyResponse { message: "Worker registered as ready".to_string() }))
}

/// Handle GET /v1/workers/list
///
/// List all workers
pub async fn handle_list_workers(State(state): State<AppState>) -> Json<ListWorkersResponse> {
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
        let response = WorkerReadyResponse { message: "Worker registered as ready".to_string() };

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
            failed_health_checks: 0,
            pid: None,
            restart_count: 0,
            last_restart: None,
            last_heartbeat: None,
        };

        let response = ListWorkersResponse { workers: vec![worker] };

        let json = serde_json::to_value(&response).unwrap();
        assert!(json["workers"].is_array());
        assert_eq!(json["workers"].as_array().unwrap().len(), 1);
    }
}
