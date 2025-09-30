//! Node management API endpoints for CLOUD_PROFILE
//!
//! Implements:
//! - POST /v2/nodes/register - GPU nodes register on startup
//! - POST /v2/nodes/{id}/heartbeat - Periodic health reporting
//! - DELETE /v2/nodes/{id} - Graceful deregistration
//! - GET /v2/nodes - List all registered nodes

use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::{StatusCode, HeaderMap},
    response::{IntoResponse, Json},
};
use pool_registry_types::NodeInfo;
use service_registry::{RegisterRequest, RegisterResponse, HeartbeatRequest, HeartbeatResponse};
use tracing::{info, warn};

/// Validate Bearer token from Authorization header
fn validate_token(headers: &HeaderMap, state: &AppState) -> bool {
    // If no token configured, allow all requests (backward compat)
    let expected_token = match std::env::var("LLORCH_API_TOKEN") {
        Ok(token) if !token.is_empty() => token,
        _ => return true, // No token required
    };

    // Extract Authorization header
    let auth_header = match headers.get("authorization") {
        Some(h) => h.to_str().unwrap_or(""),
        None => return false,
    };

    // Check Bearer token
    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        token == expected_token
    } else {
        false
    }
}

/// POST /v2/nodes/register
///
/// GPU nodes call this on startup to register with the control plane.
///
/// Request:
/// ```json
/// {
///   "node_id": "gpu-node-1",
///   "machine_id": "machine-alpha",
///   "address": "http://192.168.1.100:9200",
///   "pools": ["pool-0", "pool-1"],
///   "capabilities": { "gpus": [...], "cpu_cores": 16, ... },
///   "version": "0.1.0"
/// }
/// ```
///
/// Response: 200 OK with confirmation
pub async fn register_node(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    // Validate Bearer token
    if !validate_token(&headers, &state) {
        warn!("Unauthorized node registration attempt");
        return (
            StatusCode::UNAUTHORIZED,
            Json(RegisterResponse {
                success: false,
                message: "Unauthorized: Invalid or missing API token".to_string(),
                node_id: req.node_id.clone(),
            }),
        );
    }

    info!(
        node_id = %req.node_id,
        machine_id = %req.machine_id,
        address = %req.address,
        pools = ?req.pools,
        "Node registration request"
    );

    // Check if cloud profile is enabled
    if !state.cloud_profile_enabled() {
        warn!("Node registration attempted but cloud profile is disabled");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(RegisterResponse {
                success: false,
                message: "Cloud profile not enabled. Set ORCHESTRATORD_CLOUD_PROFILE=true".to_string(),
                node_id: req.node_id.clone(),
            }),
        );
    }

    // Create NodeInfo
    let node = NodeInfo::new(
        req.node_id.clone(),
        req.machine_id,
        req.address,
        req.pools,
        req.capabilities,
    );

    // Register with service registry
    match state.service_registry().register(node) {
        Ok(_) => {
            info!(node_id = %req.node_id, "Node registered successfully");
            (
                StatusCode::OK,
                Json(RegisterResponse {
                    success: true,
                    message: "Registered successfully".to_string(),
                    node_id: req.node_id,
                }),
            )
        }
        Err(e) => {
            warn!(node_id = %req.node_id, error = %e, "Node registration failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(RegisterResponse {
                    success: false,
                    message: format!("Registration failed: {}", e),
                    node_id: req.node_id,
                }),
            )
        }
    }
}

/// POST /v2/nodes/{id}/heartbeat
///
/// GPU nodes send periodic heartbeat with pool status.
///
/// Request:
/// ```json
/// {
///   "timestamp": "2025-09-30T22:15:00Z",
///   "pools": [
///     {
///       "pool_id": "pool-0",
///       "ready": true,
///       "draining": false,
///       "slots_free": 3,
///       "slots_total": 4,
///       "vram_free_bytes": 18000000000,
///       "engine": "llamacpp"
///     }
///   ]
/// }
/// ```
///
/// Response: 200 OK with next heartbeat interval
pub async fn heartbeat_node(
    State(state): State<AppState>,
    Path(node_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<HeartbeatRequest>,
) -> impl IntoResponse {
    // Validate Bearer token
    if !validate_token(&headers, &state) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(HeartbeatResponse {
                success: false,
                next_heartbeat_ms: 10_000,
            }),
        );
    }

    tracing::debug!(node_id = %node_id, pools_count = req.pools.len(), "Heartbeat received");

    if !state.cloud_profile_enabled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HeartbeatResponse {
                success: false,
                next_heartbeat_ms: 10_000,
            }),
        );
    }

    match state.service_registry().heartbeat(&node_id) {
        Ok(_) => {
            // Convert HeartbeatPoolStatus to PoolSnapshot and store
            let pool_snapshots: Vec<pool_registry_types::PoolSnapshot> = req.pools.iter().map(|p| {
                pool_registry_types::PoolSnapshot {
                    pool_id: p.pool_id.clone(),
                    node_id: Some(node_id.clone()),
                    ready: p.ready,
                    draining: p.draining,
                    slots_free: p.slots_free,
                    slots_total: p.slots_total,
                    vram_free_bytes: p.vram_free_bytes,
                    engine: p.engine.clone(),
                    models_available: vec![],
                }
            }).collect();
            
            state.service_registry().update_pool_status(&node_id, pool_snapshots);
            
            (
                StatusCode::OK,
                Json(HeartbeatResponse {
                    success: true,
                    next_heartbeat_ms: 10_000,
                }),
            )
        }
        Err(e) => {
            warn!(node_id = %node_id, error = %e, "Heartbeat failed");
            (
                StatusCode::NOT_FOUND,
                Json(HeartbeatResponse {
                    success: false,
                    next_heartbeat_ms: 10_000,
                }),
            )
        }
    }
}

/// DELETE /v2/nodes/{id}
///
/// GPU nodes call this on graceful shutdown to deregister.
pub async fn deregister_node(
    State(state): State<AppState>,
    Path(node_id): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // Validate Bearer token
    if !validate_token(&headers, &state) {
        return StatusCode::UNAUTHORIZED;
    }

    info!(node_id = %node_id, "Node deregistration request");

    if !state.cloud_profile_enabled() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }

    match state.service_registry().deregister(&node_id) {
        Ok(_) => {
            info!(node_id = %node_id, "Node deregistered successfully");
            StatusCode::NO_CONTENT
        }
        Err(e) => {
            warn!(node_id = %node_id, error = %e, "Deregistration failed");
            StatusCode::NOT_FOUND
        }
    }
}

/// GET /v2/nodes
///
/// List all registered nodes (for debugging/monitoring)
pub async fn list_nodes(State(state): State<AppState>) -> impl IntoResponse {
    if !state.cloud_profile_enabled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Cloud profile not enabled"
            })),
        );
    }

    let nodes = state.service_registry().get_online_nodes();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "nodes": nodes,
            "count": nodes.len()
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;
    use crate::app::router::build_router;

    #[tokio::test]
    async fn test_register_node_disabled_cloud_profile() {
        // Cloud profile disabled by default
        let state = AppState::new();
        let app = build_router(state);

        let req_body = serde_json::json!({
            "node_id": "test-node",
            "machine_id": "test-machine",
            "address": "http://localhost:9200",
            "pools": ["pool-0"],
            "capabilities": {
                "gpus": [],
                "cpu_cores": 8,
                "ram_total_bytes": 32000000000u64
            }
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v2/nodes/register")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_register_node_cloud_profile_enabled() {
        // Enable cloud profile
        std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
        
        let state = AppState::new();
        let app = build_router(state);

        let req_body = serde_json::json!({
            "node_id": "test-node",
            "machine_id": "test-machine",
            "address": "http://localhost:9200",
            "pools": ["pool-0"],
            "capabilities": {
                "gpus": [],
                "cpu_cores": 8,
                "ram_total_bytes": 32000000000u64
            },
            "version": "0.1.0"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v2/nodes/register")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        
        std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
    }

    #[tokio::test]
    async fn test_heartbeat_node_not_registered() {
        std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
        
        let state = AppState::new();
        let app = build_router(state);

        let req_body = serde_json::json!({
            "timestamp": "2025-09-30T22:00:00Z",
            "pools": []
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v2/nodes/unknown-node/heartbeat")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        
        std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
    }

    #[tokio::test]
    async fn test_list_nodes_empty() {
        std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
        
        let state = AppState::new();
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v2/nodes")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        
        std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
    }
}
