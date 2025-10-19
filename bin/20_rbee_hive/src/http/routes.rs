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

use crate::http::{capacity, devices, health, heartbeat, metrics, middleware::auth_middleware, models, shutdown, workers}; // TEAM-104: Added metrics, TEAM-115: Added heartbeat, TEAM-151: Added devices, capacity, shutdown
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
/// TEAM-114: Added audit_logger for security audit trail
/// TEAM-124: Added queen_callback_url for worker ready notifications
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<WorkerRegistry>,
    pub model_catalog: Arc<ModelCatalog>,
    pub provisioner: Arc<ModelProvisioner>,
    pub download_tracker: Arc<DownloadTracker>,
    pub server_addr: std::net::SocketAddr,
    // TEAM-102: API token for authentication (loaded from file via secrets-management)
    pub expected_token: String,
    // TEAM-114: Audit logger for security events (disabled by default for home lab mode)
    pub audit_logger: Option<Arc<audit_logging::AuditLogger>>,
    // TEAM-124: Queen-rbee callback URL for worker ready notifications (optional for standalone mode)
    pub queen_callback_url: Option<String>,
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
/// * `audit_logger` - Audit logger for security events (TEAM-114)
/// * `queen_callback_url` - Queen-rbee callback URL for worker ready notifications (TEAM-124)
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
    audit_logger: Option<Arc<audit_logging::AuditLogger>>, // TEAM-114: Audit logging
    queen_callback_url: Option<String>, // TEAM-124: Queen-rbee callback URL
) -> Router {
    let state = AppState {
        registry,
        model_catalog,
        provisioner,
        download_tracker,
        server_addr,
        expected_token, // TEAM-102
        audit_logger,   // TEAM-114
        queen_callback_url, // TEAM-124
    };

    // TEAM-102: Split routes into public and protected
    // TEAM-104: Added /metrics and Kubernetes health endpoints (public)
    // TEAM-113: Removed Kubernetes health endpoints (drift prevention)
    let public_routes = Router::new()
        // Health endpoints (public - no auth required)
        .route("/v1/health", get(health::handle_health))
        // TEAM-104: Kubernetes probes - REMOVED BY TEAM-113 (see health.rs for explanation)
        // .route("/health/live", get(health::handle_liveness)) // ⚠️ REMOVED - Kubernetes drift
        // .route("/health/ready", get(health::handle_readiness)) // ⚠️ REMOVED - Kubernetes drift
        // TEAM-104: Metrics endpoint (public - Prometheus scraping)
        .route("/metrics", get(metrics::handle_metrics));

    let protected_routes = Router::new()
        // Worker management (protected)
        .route("/v1/workers/spawn", post(workers::handle_spawn_worker))
        .route("/v1/workers/ready", post(workers::handle_worker_ready))
        .route("/v1/workers/list", get(workers::handle_list_workers))
        // TEAM-115: Heartbeat endpoint (protected)
        .route("/v1/heartbeat", post(heartbeat::handle_heartbeat))
        // Model management (protected)
        .route("/v1/models/download", post(models::handle_download_model))
        .route("/v1/models/download/progress", get(models::handle_download_progress))
        // TEAM-151: Device detection (protected - queen-only)
        .route("/v1/devices", get(devices::handle_devices))
        // TEAM-151: VRAM capacity check (protected - queen-only)
        .route("/v1/capacity", get(capacity::handle_capacity_check))
        // TEAM-151: Graceful shutdown (protected - queen-only)
        .route("/v1/shutdown", post(shutdown::handle_shutdown))
        // TEAM-102: Apply authentication middleware to all protected routes
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // TEAM-102: Merge public and protected routes
    public_routes.merge(protected_routes).with_state(state)
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
        let audit_logger = None; // TEAM-114: Disabled for tests
        let queen_callback_url = None; // TEAM-124: No callback in tests
        let _router = create_router(
            registry,
            catalog,
            provisioner,
            download_tracker,
            addr,
            expected_token,
            audit_logger,
            queen_callback_url,
        );
        // Router creation should not panic
    }
}
