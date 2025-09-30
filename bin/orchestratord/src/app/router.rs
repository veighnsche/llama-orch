use super::auth_min::bearer_auth_middleware;
use super::middleware::correlation_id_layer;
use crate::{api, state::AppState};
use axum::{
    middleware,
    routing::{delete, get, post},
    Router,
};

pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Capabilities (v2)
        .route("/v2/meta/capabilities", get(api::control::get_capabilities))
        // Node management (CLOUD_PROFILE)
        .route("/v2/nodes/register", post(api::nodes::register_node))
        .route("/v2/nodes/:id/heartbeat", post(api::nodes::heartbeat_node))
        .route("/v2/nodes/:id", delete(api::nodes::deregister_node))
        .route("/v2/nodes", get(api::nodes::list_nodes))
        // Control
        .route("/v2/pools/:id/health", get(api::control::get_pool_health))
        .route("/v2/pools/:id/drain", post(api::control::drain_pool))
        .route("/v2/pools/:id/reload", post(api::control::reload_pool))
        // Purge (v2 only; stub handler returns 202 Accepted)
        .route("/v2/pools/:id/purge", post(api::control::purge_pool_v2))
        // Worker registration
        .route("/v2/workers/register", post(api::control::register_worker))
        // Catalog
        .route("/v2/catalog/models", post(api::catalog::create_model))
        .route("/v2/catalog/models/:id", get(api::catalog::get_model))
        .route("/v2/catalog/models/:id", delete(api::catalog::delete_model))
        .route("/v2/catalog/models/:id/verify", post(api::catalog::verify_model))
        .route("/v2/catalog/models/:id/state", post(api::catalog::set_model_state))
        .route("/v2/catalog/availability", get(api::catalog_availability::get_catalog_availability))
        // Data
        .route("/v2/tasks", post(api::data::create_task))
        .route("/v2/tasks/:id/events", get(api::data::stream_task))
        .route("/v2/tasks/:id/cancel", post(api::data::cancel_task))
        // Sessions
        .route("/v2/sessions/:id", get(api::data::get_session))
        .route("/v2/sessions/:id", delete(api::data::delete_session))
        // Artifacts
        .route("/v2/artifacts", post(api::artifacts::create_artifact))
        .route("/v2/artifacts/:id", get(api::artifacts::get_artifact))
        // Observability
        .route("/metrics", get(api::observability::metrics_endpoint))
        // Middleware layers (order: correlation id, then Bearer auth)
        // All endpoints except /metrics require Bearer token authentication
        .layer(middleware::from_fn(correlation_id_layer))
        .layer(middleware::from_fn(bearer_auth_middleware))
        .with_state(state)
}
