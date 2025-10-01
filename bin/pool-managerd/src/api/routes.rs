//! HTTP route handlers for pool-managerd API.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    middleware,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use super::auth;
use crate::core::registry::Registry;
use crate::lifecycle::preload;
// TODO: Remove PreparedEngine - migrating to worker-orcd
// use provisioners_engine_provisioner::PreparedEngine;

/// Shared state for API handlers
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<Mutex<Registry>>,
}

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

// TODO: Remove PreparedEngine - migrating to worker-orcd
/*
/// Preload request body
#[derive(Deserialize)]
pub struct PreloadRequest {
    pub prepared: PreparedEngine,
}
*/

/// Preload response
#[derive(Serialize)]
pub struct PreloadResponse {
    pub pool_id: String,
    pub pid: u32,
    pub handoff_path: String,
}

/// Pool status response
#[derive(Serialize)]
pub struct PoolStatusResponse {
    pub pool_id: String,
    pub live: bool,
    pub ready: bool,
    pub active_leases: i32,
    pub engine_version: Option<String>,
}

/// Error response
#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

/// Create router with all routes
///
/// Authentication middleware is applied to all routes except /health.
/// Uses auth-min library for timing-safe Bearer token validation.
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        // TODO: Remove preload route - migrating to worker-orcd
        // .route("/pools/:id/preload", post(preload_pool))
        .route("/pools/:id/status", get(pool_status))
        .layer(middleware::from_fn(auth::auth_middleware))
        .with_state(state)
}

/// GET /health
async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

// TODO: Remove this function - migrating to worker-orcd
/*
/// POST /pools/{id}/preload
async fn preload_pool(
    State(state): State<AppState>,
    Path(_pool_id): Path<String>,
    Json(req): Json<PreloadRequest>,
) -> Result<Json<PreloadResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut registry = state.registry.lock().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: format!("registry lock failed: {}", e) }),
        )
    })?;

    match preload::execute(req.prepared, &mut registry) {
        Ok(outcome) => Ok(Json(PreloadResponse {
            pool_id: outcome.pool_id,
            pid: outcome.pid,
            handoff_path: outcome.handoff_path.to_string_lossy().to_string(),
        })),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: format!("preload failed: {}", e) }),
        )),
    }
}
*/

/// GET /pools/{id}/status
async fn pool_status(
    State(state): State<AppState>,
    Path(pool_id): Path<String>,
) -> Result<Json<PoolStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let registry = state.registry.lock().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: format!("registry lock failed: {}", e) }),
        )
    })?;

    let health = registry.get_health(&pool_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: format!("pool {} not found", pool_id) }),
        )
    })?;

    let active_leases = registry.get_active_leases(&pool_id);
    let engine_version = registry.get_engine_version(&pool_id);

    Ok(Json(PoolStatusResponse {
        pool_id,
        live: health.live,
        ready: health.ready,
        active_leases,
        engine_version,
    }))
}
