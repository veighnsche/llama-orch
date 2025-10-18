//! HTTP route configuration
//!
//! Mirrors the pattern from rbee-hive/src/http/routes.rs
//!
//! # Endpoints
//! - `GET /health` - Health check endpoint
//! - `POST /v2/registry/beehives/add` - Add node to registry
//! - `GET /v2/registry/beehives/list` - List all nodes
//! - `POST /v2/registry/beehives/remove` - Remove node from registry
//! - `GET /v2/workers/list` - List all workers
//! - `GET /v2/workers/health` - Get worker health status
//! - `POST /v2/workers/shutdown` - Shutdown a worker
//! - `POST /v2/tasks` - Create inference task
//!
//! Created by: TEAM-043
//! Refactored by: TEAM-052

use crate::beehive_registry::BeehiveRegistry;
use crate::http::{beehives, health, inference, middleware::auth_middleware, workers};
use crate::worker_registry::WorkerRegistry;
use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use std::sync::Arc;

/// Shared application state
/// TEAM-114: Added audit_logger for security audit trail
#[derive(Clone)]
pub struct AppState {
    pub beehive_registry: Arc<BeehiveRegistry>,
    pub worker_registry: Arc<WorkerRegistry>,
    // TEAM-102: API token for authentication (loaded from file via secrets-management)
    pub expected_token: String,
    // TEAM-114: Audit logger for security events (disabled by default for home lab mode)
    pub audit_logger: Option<Arc<audit_logging::AuditLogger>>,
}

/// Create HTTP router with all endpoints
///
/// # Arguments
/// * `beehive_registry` - Beehive registry (shared state)
/// * `worker_registry` - Worker registry (shared state)
/// * `expected_token` - API token for authentication (TEAM-102)
/// * `audit_logger` - Audit logger for security events (TEAM-114)
///
/// # Returns
/// Router with all endpoints configured
pub fn create_router(
    beehive_registry: Arc<BeehiveRegistry>,
    worker_registry: Arc<WorkerRegistry>,
    expected_token: String, // TEAM-102: API token for authentication
    audit_logger: Option<Arc<audit_logging::AuditLogger>>, // TEAM-114: Audit logging
) -> Router {
    let state = AppState {
        beehive_registry,
        worker_registry,
        expected_token, // TEAM-102
        audit_logger,   // TEAM-114
    };

    // TEAM-102: Split routes into public and protected
    let public_routes = Router::new()
        // Health endpoint (public - no auth required)
        .route("/health", get(health::handle_health));

    let protected_routes = Router::new()
        // Beehive registry endpoints (protected)
        .route("/v2/registry/beehives/add", post(beehives::handle_add_node))
        .route("/v2/registry/beehives/list", get(beehives::handle_list_nodes))
        .route("/v2/registry/beehives/remove", post(beehives::handle_remove_node))
        // Worker management endpoints (protected)
        .route("/v2/workers/list", get(workers::handle_list_workers))
        .route("/v2/workers/health", get(workers::handle_workers_health))
        .route("/v2/workers/shutdown", post(workers::handle_shutdown_worker))
        .route("/v2/workers/register", post(workers::handle_register_worker)) // TEAM-084: Worker registration
        // Inference task endpoints (protected)
        .route("/v2/tasks", post(inference::handle_create_inference_task))
        .route("/v1/inference", post(inference::handle_inference_request)) // TEAM-084: Direct inference endpoint
        // TEAM-102: Apply authentication middleware to all protected routes
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // TEAM-102: Merge public and protected routes
    public_routes.merge(protected_routes).with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_router_creation() {
        let beehive_registry = Arc::new(BeehiveRegistry::new(None).await.unwrap());
        let worker_registry = Arc::new(WorkerRegistry::new());
        let expected_token = "test-token-12345".to_string(); // TEAM-102
        let audit_logger = None; // TEAM-114: Disabled for tests
        let _router =
            create_router(beehive_registry, worker_registry, expected_token, audit_logger);
        // Router creation should not panic
    }
}
