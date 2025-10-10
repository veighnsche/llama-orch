//! Worker management endpoints
//!
//! Endpoints:
//! - GET /v2/workers/list - List all workers
//! - GET /v2/workers/health - Get worker health status
//! - POST /v2/workers/shutdown - Shutdown a worker
//!
//! Created by: TEAM-046
//! Refactored by: TEAM-052

use axum::{
    extract::{Json, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use std::collections::HashMap;
use tracing::error;

use crate::http::routes::AppState;
use crate::http::types::{
    ShutdownWorkerRequest, ShutdownWorkerResponse, WorkerHealthInfo, WorkerInfo,
    WorkersHealthResponse, WorkersListResponse,
};
use crate::worker_registry::WorkerInfoExtended;

/// Handle GET /v2/workers/list
///
/// Returns list of all registered workers
pub async fn handle_list_workers(State(state): State<AppState>) -> impl IntoResponse {
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

/// Handle GET /v2/workers/health
///
/// Returns health status of workers on a specific node
pub async fn handle_workers_health(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
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

/// Handle POST /v2/workers/shutdown
///
/// Sends shutdown command to a worker
pub async fn handle_shutdown_worker(
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
