//! HTTP route configuration
//!
//! Mirrors the pattern from llm-worker-rbee/src/http/routes.rs
//!
//! # Endpoints
//! - `GET /v1/health` - Health check endpoint
//! - `POST /v1/workers/spawn` - Spawn a new worker
//! - `POST /v1/workers/ready` - Worker ready callback
//! - `GET /v1/workers/list` - List all workers
//! - `POST /v1/models/download` - Download a model
//! - `GET /v1/models/download/progress` - SSE progress stream
//!
//! Created by: TEAM-026

use crate::http::{health, models, workers};
use crate::registry::WorkerRegistry;
use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;

/// Create HTTP router with all endpoints
///
/// # Arguments
/// * `registry` - Worker registry (shared state)
///
/// # Returns
/// Router with all endpoints configured
pub fn create_router(registry: Arc<WorkerRegistry>) -> Router {
    Router::new()
        // Health endpoint
        .route("/v1/health", get(health::handle_health))
        // Worker management
        .route("/v1/workers/spawn", post(workers::handle_spawn_worker))
        .route("/v1/workers/ready", post(workers::handle_worker_ready))
        .route("/v1/workers/list", get(workers::handle_list_workers))
        // Model management
        .route("/v1/models/download", post(models::handle_download_model))
        .route(
            "/v1/models/download/progress",
            get(models::handle_download_progress),
        )
        .with_state(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let registry = Arc::new(WorkerRegistry::new());
        let _router = create_router(registry);
        // Router creation should not panic
    }
}
