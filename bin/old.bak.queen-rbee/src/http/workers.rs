//! Worker management endpoints
//!
//! Endpoints:
//! - GET /v2/workers/list - List all workers
//! - GET /v2/workers/health - Get worker health status
//! - POST /v2/workers/shutdown - Shutdown a worker
//! - POST /v2/workers/ready - Worker ready notification (TEAM-124)
//!
//! Created by: TEAM-046
//! Refactored by: TEAM-052

use axum::{
    extract::{Json, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use std::collections::HashMap;
use tracing::{error, info};

use crate::http::routes::AppState;
use crate::http::types::{
    RegisterWorkerRequest, RegisterWorkerResponse, ShutdownWorkerRequest, ShutdownWorkerResponse,
    WorkerHealthInfo, WorkerInfo, WorkerReadyNotification, WorkerReadyResponse,
    WorkersHealthResponse, WorkersListResponse,
};
use crate::worker_registry::{WorkerInfoExtended, WorkerState};

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

/// Handle POST /v2/workers/register
///
/// Register a new worker from rbee-hive
/// Created by: TEAM-084
pub async fn handle_register_worker(
    State(state): State<AppState>,
    Json(req): Json<RegisterWorkerRequest>,
) -> impl IntoResponse {
    use crate::worker_registry::WorkerInfo as RegistryWorkerInfo;

    // Convert request to registry format
    let worker = RegistryWorkerInfo {
        id: req.worker_id.clone(),
        url: req.url.clone(),
        model_ref: req.model_ref.clone(),
        backend: req.backend.clone(),
        device: req.device,
        state: WorkerState::Idle, // Default to Idle on registration
        slots_total: req.slots_total.unwrap_or(1),
        slots_available: req.slots_total.unwrap_or(1),
        vram_bytes: req.vram_bytes,
        node_name: req.node_name.clone(),
    };

    // Register the worker
    state.worker_registry.register(worker).await;

    tracing::info!("✅ Worker registered: {} from node {}", req.worker_id, req.node_name);

    (
        StatusCode::OK,
        Json(RegisterWorkerResponse {
            success: true,
            message: format!("Worker '{}' registered successfully", req.worker_id),
            worker_id: req.worker_id,
        }),
    )
}

/// Handle POST /v2/workers/ready
///
/// Receive worker ready notification from rbee-hive
/// Created by: TEAM-124
pub async fn handle_worker_ready(
    State(state): State<AppState>,
    Json(notification): Json<WorkerReadyNotification>,
) -> impl IntoResponse {
    info!(
        worker_id = %notification.worker_id,
        url = %notification.url,
        model_ref = %notification.model_ref,
        "✅ Worker ready notification received from rbee-hive"
    );

    // Update worker state to idle (ready to accept requests)
    match state.worker_registry.update_worker_state(&notification.worker_id, WorkerState::Idle).await {
        Ok(_) => {
            info!(
                worker_id = %notification.worker_id,
                "Worker state updated to Idle"
            );
            (
                StatusCode::OK,
                Json(WorkerReadyResponse {
                    success: true,
                    message: format!("Worker '{}' marked as ready", notification.worker_id),
                }),
            )
        }
        Err(e) => {
            error!(
                worker_id = %notification.worker_id,
                error = %e,
                "Failed to update worker state"
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(WorkerReadyResponse {
                    success: false,
                    message: format!("Failed to update worker state: {}", e),
                }),
            )
        }
    }
}
