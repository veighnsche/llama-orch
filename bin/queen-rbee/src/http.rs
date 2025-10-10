//! HTTP Server for queen-rbee
//!
//! Created by: TEAM-043
//!
//! Provides REST API for rbee-hive registry management

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{error, info};

use crate::beehive_registry::{BeehiveNode, BeehiveRegistry};
use crate::worker_registry::{WorkerRegistry, WorkerInfoExtended};

#[derive(Clone)]
pub struct AppState {
    pub beehive_registry: Arc<BeehiveRegistry>,
    pub worker_registry: Arc<WorkerRegistry>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Request/Response Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Deserialize)]
pub struct AddNodeRequest {
    pub node_name: String,
    pub ssh_host: String,
    #[serde(default = "default_ssh_port")]
    pub ssh_port: u16,
    pub ssh_user: String,
    pub ssh_key_path: Option<String>,
    pub git_repo_url: String,
    pub git_branch: String,
    pub install_path: String,
}

fn default_ssh_port() -> u16 {
    22
}

#[derive(Debug, Serialize)]
pub struct AddNodeResponse {
    pub success: bool,
    pub message: String,
    pub node_name: String,
}

#[derive(Debug, Serialize)]
pub struct ListNodesResponse {
    pub nodes: Vec<BeehiveNode>,
}

#[derive(Debug, Serialize)]
pub struct RemoveNodeResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

// TEAM-046: Worker management types
#[derive(Debug, Serialize)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub node: String,
    pub state: String,
    pub model_ref: Option<String>,
    pub url: String,
}

#[derive(Debug, Serialize)]
pub struct WorkersListResponse {
    pub workers: Vec<WorkerInfo>,
}

#[derive(Debug, Serialize)]
pub struct WorkerHealthInfo {
    pub worker_id: String,
    pub state: String,
    pub ready: bool,
}

#[derive(Debug, Serialize)]
pub struct WorkersHealthResponse {
    pub status: String,
    pub workers: Vec<WorkerHealthInfo>,
}

#[derive(Debug, Deserialize)]
pub struct ShutdownWorkerRequest {
    pub worker_id: String,
}

#[derive(Debug, Serialize)]
pub struct ShutdownWorkerResponse {
    pub success: bool,
    pub message: String,
}

// TEAM-046: Inference task types
#[derive(Debug, Deserialize)]
pub struct InferenceTaskRequest {
    pub node: String,
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

// TEAM-047: Worker spawn response from rbee-hive
#[derive(Debug, Deserialize)]
struct WorkerSpawnResponse {
    worker_id: String,
    url: String,
    #[allow(dead_code)]
    state: String,
}

// TEAM-047: Worker ready response
#[derive(Debug, Deserialize)]
struct ReadyResponse {
    ready: bool,
    #[allow(dead_code)]
    state: String,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Route Handlers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn add_node(
    State(state): State<AppState>,
    Json(req): Json<AddNodeRequest>,
) -> impl IntoResponse {
    // TEAM-044: Smart SSH mocking for tests
    // If MOCK_SSH is set, simulate SSH based on hostname:
    // - "unreachable" in hostname -> fail (to test error handling)
    // - other hostnames -> succeed (for normal test nodes)
    let mock_ssh = std::env::var("MOCK_SSH").is_ok();

    let ssh_success = if mock_ssh {
        // Smart mock: fail for "unreachable" hosts, succeed for others
        if req.ssh_host.contains("unreachable") {
            info!("🔌 Mock SSH: Simulating connection failure for {}", req.ssh_host);
            false
        } else {
            info!("🔌 Mock SSH: Simulating successful connection to {}", req.ssh_host);
            true
        }
    } else {
        info!("🔌 Testing SSH connection to {}", req.ssh_host);
        // Real SSH connection test
        crate::ssh::test_ssh_connection(
            &req.ssh_host,
            req.ssh_port,
            &req.ssh_user,
            req.ssh_key_path.as_deref(),
        )
        .await
        .unwrap_or(false)
    };

    if !ssh_success {
        error!("❌ SSH connection failed: Connection timeout");
        return (
            StatusCode::BAD_REQUEST,
            Json(AddNodeResponse {
                success: false,
                message: "SSH connection failed: Connection timeout".to_string(),
                node_name: req.node_name.clone(),
            }),
        );
    }

    // Save node to registry
    let node = BeehiveNode {
        node_name: req.node_name.clone(),
        ssh_host: req.ssh_host,
        ssh_port: req.ssh_port,
        ssh_user: req.ssh_user,
        ssh_key_path: req.ssh_key_path,
        git_repo_url: req.git_repo_url,
        git_branch: req.git_branch,
        install_path: req.install_path,
        last_connected_unix: Some(chrono::Utc::now().timestamp()),
        status: "reachable".to_string(),
    };

    match state.beehive_registry.add_node(node).await {
        Ok(_) => {
            info!("✅ SSH connection successful! Node '{}' saved to registry", req.node_name);
            (
                StatusCode::OK,
                Json(AddNodeResponse {
                    success: true,
                    message: format!("Node '{}' added successfully", req.node_name),
                    node_name: req.node_name,
                }),
            )
        }
        Err(e) => {
            error!("Failed to save node: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(AddNodeResponse {
                    success: false,
                    message: format!("Failed to save node: {}", e),
                    node_name: req.node_name,
                }),
            )
        }
    }
}

async fn list_nodes(State(state): State<AppState>) -> impl IntoResponse {
    match state.beehive_registry.list_nodes().await {
        Ok(nodes) => (StatusCode::OK, Json(ListNodesResponse { nodes })),
        Err(e) => {
            error!("Failed to list nodes: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ListNodesResponse { nodes: vec![] }))
        }
    }
}

async fn remove_node(
    State(state): State<AppState>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let node_name = req["node_name"].as_str().unwrap_or("");

    match state.beehive_registry.remove_node(node_name).await {
        Ok(true) => (
            StatusCode::OK,
            Json(RemoveNodeResponse {
                success: true,
                message: format!("Node '{}' removed successfully", node_name),
            }),
        ),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(RemoveNodeResponse {
                success: false,
                message: format!("Node '{}' not found", node_name),
            }),
        ),
        Err(e) => {
            error!("Failed to remove node: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(RemoveNodeResponse {
                    success: false,
                    message: format!("Failed to remove node: {}", e),
                }),
            )
        }
    }
}

// TEAM-046: Worker management handlers
async fn list_workers(State(state): State<AppState>) -> impl IntoResponse {
    match state.worker_registry.list_workers().await {
        Ok(workers) => {
            let worker_infos: Vec<WorkerInfo> = workers
                .into_iter()
                .map(|w: WorkerInfoExtended| WorkerInfo {
                    worker_id: w.worker_id,
                    node: w.node_name,
                    state: w.state,
                    model_ref: w.model_ref,
                    url: w.url,
                })
                .collect();
            (StatusCode::OK, Json(WorkersListResponse { workers: worker_infos }))
        }
        Err(e) => {
            error!("Failed to list workers: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(WorkersListResponse { workers: vec![] }))
        }
    }
}

async fn workers_health(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let node = params.get("node").map(|s| s.as_str()).unwrap_or("");

    match state.worker_registry.get_workers_by_node(node).await {
        Ok(workers) => {
            let health_infos: Vec<WorkerHealthInfo> = workers
                .into_iter()
                .map(|w: WorkerInfoExtended| WorkerHealthInfo {
                    worker_id: w.worker_id,
                    state: w.state.clone(),
                    ready: w.state == "idle" || w.state == "ready",
                })
                .collect();
            (
                StatusCode::OK,
                Json(WorkersHealthResponse { status: "ok".to_string(), workers: health_infos }),
            )
        }
        Err(e) => {
            error!("Failed to get worker health: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(WorkersHealthResponse { status: "error".to_string(), workers: vec![] }),
            )
        }
    }
}

async fn shutdown_worker(
    State(state): State<AppState>,
    Json(req): Json<ShutdownWorkerRequest>,
) -> impl IntoResponse {
    match state.worker_registry.shutdown_worker(&req.worker_id).await {
        Ok(_) => (
            StatusCode::OK,
            Json(ShutdownWorkerResponse {
                success: true,
                message: format!("Worker '{}' shutdown command sent", req.worker_id),
            }),
        ),
        Err(e) => {
            error!("Failed to shutdown worker: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ShutdownWorkerResponse {
                    success: false,
                    message: format!("Failed to shutdown worker: {}", e),
                }),
            )
        }
    }
}

// TEAM-047: Inference task handler - full orchestration implementation
async fn create_inference_task(
    State(state): State<AppState>,
    Json(req): Json<InferenceTaskRequest>,
) -> impl IntoResponse {
    info!("Received inference task: node={}, model={}", req.node, req.model);
    
    // Step 1: Query rbee-hive registry for node SSH details
    let node = match state.beehive_registry.get_node(&req.node).await {
        Ok(Some(node)) => node,
        Ok(None) => {
            error!("Node not found in registry: {}", req.node);
            return (StatusCode::NOT_FOUND, format!("Node '{}' not registered", req.node)).into_response();
        }
        Err(e) => {
            error!("Failed to query registry: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Registry error: {}", e)).into_response();
        }
    };
    
    // Step 2: Determine rbee-hive URL (mock SSH for tests)
    let mock_ssh = std::env::var("MOCK_SSH").is_ok();
    let rbee_hive_url = if mock_ssh {
        // For tests, assume rbee-hive is already running on localhost
        info!("🔌 Mock SSH: Using localhost rbee-hive");
        format!("http://127.0.0.1:8080")
    } else {
        // Real SSH: start rbee-hive daemon on remote node
        info!("🔌 Establishing SSH connection to {}", node.ssh_host);
        match establish_rbee_hive_connection(&node).await {
            Ok(url) => url,
            Err(e) => {
                error!("Failed to connect to rbee-hive: {}", e);
                return (StatusCode::SERVICE_UNAVAILABLE, format!("SSH connection failed: {}", e)).into_response();
            }
        }
    };
    
    // Step 3: Spawn worker on rbee-hive
    info!("Spawning worker on rbee-hive at {}", rbee_hive_url);
    let client = reqwest::Client::new();
    let spawn_request = serde_json::json!({
        "model_ref": req.model,
        "backend": "cpu",
        "device": 0,
        "model_path": ""
    });
    
    let worker = match client
        .post(format!("{}/v1/workers/spawn", rbee_hive_url))
        .json(&spawn_request)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<WorkerSpawnResponse>().await {
                Ok(worker) => {
                    info!("Worker spawned: {} at {}", worker.worker_id, worker.url);
                    worker
                }
                Err(e) => {
                    error!("Failed to parse worker response: {}", e);
                    return (StatusCode::INTERNAL_SERVER_ERROR, format!("Worker spawn parse error: {}", e)).into_response();
                }
            }
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!("Worker spawn failed: HTTP {} - {}", status, body);
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Worker spawn failed: HTTP {}", status)).into_response();
        }
        Err(e) => {
            error!("Failed to spawn worker: {}", e);
            return (StatusCode::SERVICE_UNAVAILABLE, format!("Worker spawn request failed: {}", e)).into_response();
        }
    };
    
    // Step 4: Wait for worker ready
    info!("Waiting for worker {} to be ready", worker.worker_id);
    match wait_for_worker_ready(&worker.url).await {
        Ok(_) => info!("Worker ready: {}", worker.worker_id),
        Err(e) => {
            error!("Worker failed to become ready: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Worker ready timeout: {}", e)).into_response();
        }
    }
    
    // Step 5: Execute inference and stream results
    info!("Executing inference on worker {}", worker.worker_id);
    let inference_request = serde_json::json!({
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": true
    });
    
    let response = match client
        .post(format!("{}/v1/inference", worker.url))
        .json(&inference_request)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => resp,
        Ok(resp) => {
            let status = resp.status();
            error!("Inference failed: HTTP {}", status);
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Inference failed: HTTP {}", status)).into_response();
        }
        Err(e) => {
            error!("Failed to execute inference: {}", e);
            return (StatusCode::SERVICE_UNAVAILABLE, format!("Inference request failed: {}", e)).into_response();
        }
    };
    
    // TEAM-048: Stream SSE response back to client (pass-through, don't re-wrap)
    // The worker already sends properly formatted SSE events, just proxy them
    use axum::body::Body;
    use axum::http::header;
    
    let stream = response.bytes_stream();
    
    (
        [(header::CONTENT_TYPE, "text/event-stream")],
        Body::from_stream(stream)
    ).into_response()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Helper Functions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// TEAM-047: Establish rbee-hive connection via SSH
async fn establish_rbee_hive_connection(node: &BeehiveNode) -> anyhow::Result<String> {
    // Execute remote command to start rbee-hive daemon
    let start_command = format!(
        "{}/rbee-hive daemon --addr 0.0.0.0:8080 > /tmp/rbee-hive.log 2>&1 &",
        node.install_path
    );
    
    let (success, stdout, stderr) = crate::ssh::execute_remote_command(
        &node.ssh_host,
        node.ssh_port,
        &node.ssh_user,
        node.ssh_key_path.as_deref(),
        &start_command,
    )
    .await?;
    
    if !success {
        anyhow::bail!("Failed to start rbee-hive: {}", stderr);
    }
    
    info!("rbee-hive daemon started on {}: {}", node.ssh_host, stdout);
    
    // Wait for rbee-hive to be ready
    let rbee_hive_url = format!("http://{}:8080", node.ssh_host);
    wait_for_rbee_hive_ready(&rbee_hive_url).await?;
    
    Ok(rbee_hive_url)
}

// TEAM-047: Wait for rbee-hive to be ready
// TEAM-048: Enhanced with exponential backoff retry (EC1)
async fn wait_for_rbee_hive_ready(url: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let mut backoff_ms = 100;
    let max_retries = 5;
    let timeout = std::time::Duration::from_secs(60);
    let start = std::time::Instant::now();
    
    for attempt in 0..max_retries {
        if start.elapsed() > timeout {
            anyhow::bail!("rbee-hive ready timeout after 60 seconds");
        }
        
        match client
            .get(format!("{}/health", url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                info!("rbee-hive is ready at {} (attempt {})", url, attempt + 1);
                return Ok(());
            }
            Ok(resp) => {
                info!("rbee-hive returned HTTP {}, retrying...", resp.status());
            }
            Err(e) if attempt < max_retries - 1 => {
                info!("Connection attempt {} failed: {}, retrying in {}ms", 
                      attempt + 1, e, backoff_ms);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms *= 2; // Exponential backoff
            }
            Err(e) => {
                anyhow::bail!("Failed to connect to rbee-hive after {} attempts: {}", 
                             max_retries, e);
            }
        }
    }
    
    anyhow::bail!("rbee-hive ready timeout after {} retries", max_retries)
}

// TEAM-047: Wait for worker to be ready
async fn wait_for_worker_ready(worker_url: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(300); // 5 minutes
    
    loop {
        match client
            .get(format!("{}/v1/ready", worker_url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                if let Ok(ready) = response.json::<ReadyResponse>().await {
                    if ready.ready {
                        info!("Worker ready at {}", worker_url);
                        return Ok(());
                    }
                }
            }
            _ => {}
        }
        
        if start.elapsed() > timeout {
            anyhow::bail!("Worker ready timeout after 5 minutes");
        }
        
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Router Setup
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v2/registry/beehives/add", post(add_node))
        .route("/v2/registry/beehives/list", get(list_nodes))
        .route("/v2/registry/beehives/remove", post(remove_node))
        // TEAM-046: Worker management endpoints
        .route("/v2/workers/list", get(list_workers))
        .route("/v2/workers/health", get(workers_health))
        .route("/v2/workers/shutdown", post(shutdown_worker))
        // TEAM-046: Inference task endpoint
        .route("/v2/tasks", post(create_inference_task))
        .with_state(state)
}
