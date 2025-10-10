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

// TEAM-046: Inference task handler (stub for now)
async fn create_inference_task(
    State(_state): State<AppState>,
    Json(req): Json<InferenceTaskRequest>,
) -> impl IntoResponse {
    info!("Received inference task: node={}, model={}", req.node, req.model);
    
    // TODO: Implement full orchestration flow
    // 1. Query rbee-hive registry for node SSH details
    // 2. Establish SSH connection
    // 3. Start rbee-hive on remote node
    // 4. Request worker from rbee-hive
    // 5. Stream inference results
    
    (
        StatusCode::NOT_IMPLEMENTED,
        "Inference orchestration not yet implemented",
    )
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
