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
use tracing::{info, error};

use crate::beehive_registry::{BeehiveNode, BeehiveRegistry};
use crate::worker_registry::WorkerRegistry;

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
    info!("🔌 Testing SSH connection to {}", req.ssh_host);

    // Test SSH connection
    let ssh_success = crate::ssh::test_ssh_connection(
        &req.ssh_host,
        req.ssh_port,
        &req.ssh_user,
        req.ssh_key_path.as_deref(),
    )
    .await
    .unwrap_or(false);

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
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ListNodesResponse { nodes: vec![] }),
            )
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Router Setup
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v2/registry/beehives/add", post(add_node))
        .route("/v2/registry/beehives/list", get(list_nodes))
        .route("/v2/registry/beehives/remove", post(remove_node))
        .with_state(state)
}
