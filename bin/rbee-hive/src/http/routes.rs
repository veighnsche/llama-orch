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

use crate::http::{health, models, workers};
use crate::provisioner::ModelProvisioner;
use crate::registry::WorkerRegistry;
use axum::{
    routing::{get, post},
    Router,
};
use model_catalog::ModelCatalog;
use rbee_hive::download_tracker::DownloadTracker;
use std::sync::Arc;

/// Shared application state
/// TEAM-030: Worker registry is ephemeral, model catalog is persistent (SQLite)
/// TEAM-034: Added download tracker for SSE streaming
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<WorkerRegistry>,
    pub model_catalog: Arc<ModelCatalog>,
    pub provisioner: Arc<ModelProvisioner>,
    pub download_tracker: Arc<DownloadTracker>,
}

/// Create HTTP router with all endpoints
///
/// # Arguments
/// * `registry` - Worker registry (shared state)
/// * `model_catalog` - Model catalog (shared state)
/// * `provisioner` - Model provisioner (shared state)
/// * `download_tracker` - Download tracker (shared state)
///
/// # Returns
/// Router with all endpoints configured
pub fn create_router(
    registry: Arc<WorkerRegistry>,
    model_catalog: Arc<ModelCatalog>,
    provisioner: Arc<ModelProvisioner>,
    download_tracker: Arc<DownloadTracker>,
) -> Router {
    let state = AppState { registry, model_catalog, provisioner, download_tracker };

    Router::new()
        // Health endpoint
        .route("/v1/health", get(health::handle_health))
        // Worker management
        .route("/v1/workers/spawn", post(workers::handle_spawn_worker))
        .route("/v1/workers/ready", post(workers::handle_worker_ready))
        .route("/v1/workers/list", get(workers::handle_list_workers))
        // Model management
        .route("/v1/models/download", post(models::handle_download_model))
        .route("/v1/models/download/progress", get(models::handle_download_progress))
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
        let _router = create_router(registry, catalog, provisioner, download_tracker);
        // Router creation should not panic
    }
}
