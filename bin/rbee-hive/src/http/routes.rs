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
//! Modified by: TEAM-029, TEAM-030, TEAM-034

use crate::http::{health, metrics, middleware::auth_middleware, models, workers}; // TEAM-104: Added metrics
use crate::provisioner::ModelProvisioner;
use crate::registry::WorkerRegistry;
use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use model_catalog::ModelCatalog;
use rbee_hive::download_tracker::DownloadTracker;
use std::sync::Arc;

/// Shared application state
/// TEAM-030: Worker registry is ephemeral, model catalog is persistent (SQLite)
/// TEAM-034: Added download tracker for SSE streaming
/// TEAM-091: Added server_addr for correct callback URL construction
/// TEAM-102: Added expected_token for authentication
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<WorkerRegistry>,
    pub model_catalog: Arc<ModelCatalog>,
    pub provisioner: Arc<ModelProvisioner>,
    pub download_tracker: Arc<DownloadTracker>,
    pub server_addr: std::net::SocketAddr,
    // TEAM-102: API token for authentication (loaded from file via secrets-management)
    pub expected_token: String,
}

/// Create HTTP router with all endpoints
///
/// # Arguments
/// * `registry` - Worker registry (shared state)
/// * `model_catalog` - Model catalog (shared state)
/// * `provisioner` - Model provisioner (shared state)
/// * `download_tracker` - Download tracker (shared state)
/// * `server_addr` - Server bind address for callback URL construction
/// * `expected_token` - API token for authentication (TEAM-102)
///
/// # Returns
/// Router with all endpoints configured
pub fn create_router(
    registry: Arc<WorkerRegistry>,
    model_catalog: Arc<ModelCatalog>,
    provisioner: Arc<ModelProvisioner>,
    download_tracker: Arc<DownloadTracker>,
    server_addr: std::net::SocketAddr,
    expected_token: String, // TEAM-102: API token for authentication
) -> Router {
    let state = AppState { 
        registry, 
        model_catalog, 
        provisioner, 
        download_tracker, 
        server_addr,
        expected_token, // TEAM-102
    };

    // TEAM-102: Split routes into public and protected
    // TEAM-104: Added /metrics and Kubernetes health endpoints (public)
    let public_routes = Router::new()
        // Health endpoints (public - no auth required)
        .route("/v1/health", get(health::handle_health))
        .route("/health/live", get(health::handle_liveness)) // TEAM-104: Kubernetes liveness
        .route("/health/ready", get(health::handle_readiness)) // TEAM-104: Kubernetes readiness
        // TEAM-104: Metrics endpoint (public - Prometheus scraping)
        .route("/metrics", get(metrics::handle_metrics));

    let protected_routes = Router::new()
        // Worker management (protected)
        .route("/v1/workers/spawn", post(workers::handle_spawn_worker))
        .route("/v1/workers/ready", post(workers::handle_worker_ready))
        .route("/v1/workers/list", get(workers::handle_list_workers))
        // Model management (protected)
        .route("/v1/models/download", post(models::handle_download_model))
        .route("/v1/models/download/progress", get(models::handle_download_progress))
        // TEAM-102: Apply authentication middleware to all protected routes
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // TEAM-102: Merge public and protected routes
    public_routes
        .merge(protected_routes)
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        use crate::provisioner::ModelProvisioner;
        use model_catalog::ModelCatalog;
        use rbee_hive::download_tracker::DownloadTracker;
        use std::path::PathBuf;

        let registry = Arc::new(WorkerRegistry::new());
        let catalog = Arc::new(ModelCatalog::new(":memory:".to_string()));
        let provisioner = Arc::new(ModelProvisioner::new(PathBuf::from("/tmp")));
        let download_tracker = Arc::new(DownloadTracker::new());
        let addr: std::net::SocketAddr = "127.0.0.1:9200".parse().unwrap();
        let expected_token = "test-token-12345".to_string(); // TEAM-102
        let _router = create_router(registry, catalog, provisioner, download_tracker, addr, expected_token);
        // Router creation should not panic
    }
}
