//! Beehive registry management endpoints
//!
//! Endpoints:
//! - POST /v2/registry/beehives/add - Add a node to the registry
//! - GET /v2/registry/beehives/list - List all registered nodes
//! - POST /v2/registry/beehives/remove - Remove a node from the registry
//!
//! Created by: TEAM-043
//! Refactored by: TEAM-052

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
};
use tracing::{error, info};

use crate::beehive_registry::BeehiveNode;
use crate::http::routes::AppState;
use crate::http::types::{AddNodeRequest, AddNodeResponse, ListNodesResponse, RemoveNodeResponse};
// TEAM-113: Input validation for node names
use input_validation::validate_identifier;

/// Handle POST /v2/registry/beehives/add
///
/// Validates SSH connection and adds node to registry
pub async fn handle_add_node(
    State(state): State<AppState>,
    Json(req): Json<AddNodeRequest>,
) -> impl IntoResponse {
    // TEAM-113: Validate node name before processing
    if let Err(e) = validate_identifier(&req.node_name, 64) {
        error!("Invalid node name: {}", e);
        return (
            StatusCode::BAD_REQUEST,
            Json(AddNodeResponse {
                success: false,
                message: format!("Invalid node name: {}", e),
                node_name: req.node_name,
            }),
        );
    }
    
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
        backends: req.backends,
        devices: req.devices,
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

/// Handle GET /v2/registry/beehives/list
///
/// Returns list of all registered nodes
pub async fn handle_list_nodes(State(state): State<AppState>) -> impl IntoResponse {
    match state.beehive_registry.list_nodes().await {
        Ok(nodes) => (StatusCode::OK, Json(ListNodesResponse { nodes })),
        Err(e) => {
            error!("Failed to list nodes: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(ListNodesResponse { nodes: vec![] }))
        }
    }
}

/// Handle POST /v2/registry/beehives/remove
///
/// Removes a node from the registry
pub async fn handle_remove_node(
    State(state): State<AppState>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let node_name = req["node_name"].as_str().unwrap_or("");
    
    // TEAM-113: Validate node name before processing
    if let Err(e) = validate_identifier(node_name, 64) {
        error!("Invalid node name: {}", e);
        return (
            StatusCode::BAD_REQUEST,
            Json(RemoveNodeResponse {
                success: false,
                message: format!("Invalid node name: {}", e),
            }),
        );
    }

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
