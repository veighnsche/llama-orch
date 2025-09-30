use super::auth_min::bearer_identity_layer;
use super::middleware::{api_key_layer, correlation_id_layer};
use crate::{api, state::AppState};
use axum::{
    middleware,
    routing::{delete, get, post},
    Router,
};

pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Capabilities
        .route("/v1/capabilities", get(api::control::get_capabilities))
        .route("/v2/capabilities", get(api::control::get_capabilities))
        // Control
        .route("/v1/pools/:id/health", get(api::control::get_pool_health))
        .route("/v2/pools/:id/health", get(api::control::get_pool_health))
        .route("/v1/pools/:id/drain", post(api::control::drain_pool))
        .route("/v2/pools/:id/drain", post(api::control::drain_pool))
        .route("/v1/pools/:id/reload", post(api::control::reload_pool))
        .route("/v2/pools/:id/reload", post(api::control::reload_pool))
        // Purge (v2 only; stub handler returns 202 Accepted)
        .route("/v2/pools/:id/purge", post(api::control::purge_pool_v2))
        // Worker registration
        .route("/v1/workers/register", post(api::control::register_worker))
        .route("/v2/workers/register", post(api::control::register_worker))
        // Catalog
        .route("/v1/catalog/models", post(api::catalog::create_model))
        .route("/v1/catalog/models/:id", get(api::catalog::get_model))
        .route("/v1/catalog/models/:id", delete(api::catalog::delete_model))
        .route("/v1/catalog/models/:id/verify", post(api::catalog::verify_model))
        .route("/v1/catalog/models/:id/state", post(api::catalog::set_model_state))
        // Data
        .route("/v1/tasks", post(api::data::create_task))
        .route("/v2/tasks", post(api::data::create_task))
        .route("/v1/tasks/:id/stream", get(api::data::stream_task))
        .route("/v2/tasks/:id/events", get(api::data::stream_task))
        .route("/v1/tasks/:id/cancel", post(api::data::cancel_task))
        .route("/v2/tasks/:id/cancel", post(api::data::cancel_task))
        // Sessions
        .route("/v1/sessions/:id", get(api::data::get_session))
        .route("/v1/sessions/:id", delete(api::data::delete_session))
        .route("/v2/sessions/:id", get(api::data::get_session))
        .route("/v2/sessions/:id", delete(api::data::delete_session))
        // Artifacts
        .route("/v1/artifacts", post(api::artifacts::create_artifact))
        .route("/v1/artifacts/:id", get(api::artifacts::get_artifact))
        // Observability
        .route("/metrics", get(api::observability::metrics_endpoint))
        // Layers (order: correlation id, then auth)
        .layer(middleware::from_fn(correlation_id_layer))
        .layer(middleware::from_fn(bearer_identity_layer))
        .layer(middleware::from_fn(api_key_layer))
        .with_state(state)
}
