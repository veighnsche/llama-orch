pub mod admission;
pub mod metrics;

pub mod backpressure;
pub mod domain;
pub mod errors;
pub mod http;
pub mod placement;
pub mod services;
pub mod sse;
pub mod state;

use axum::{
    routing::{get, post},
    Router,
};

/// Build the Axum router for the orchestrator.
/// Handlers are stubs during the pre-code phase; routes are wired for provider verification and DX.
pub fn build_app() -> Router<state::AppState> {
    let state = state::default_state();
    Router::new()
        // Data plane
        .route("/v1/tasks", post(http::data::create_task))
        .route("/v1/tasks/:id/stream", get(http::data::stream_task))
        .route("/v1/tasks/:id/cancel", post(http::data::cancel_task))
        .route(
            "/v1/sessions/:id",
            get(http::data::get_session).delete(http::data::delete_session),
        )
        // Control plane
        .route("/v1/pools/:id/drain", post(http::control::drain_pool))
        .route("/v1/pools/:id/reload", post(http::control::reload_pool))
        .route("/v1/pools/:id/health", get(http::control::get_pool_health))
        .route("/v1/replicasets", get(http::control::list_replicasets))
        .route("/v1/capabilities", get(http::control::get_capabilities))
        // Catalog (planning)
        .route(
            "/v1/catalog/models",
            post(http::catalog::create_catalog_model),
        )
        .route(
            "/v1/catalog/models/:id",
            get(http::catalog::get_catalog_model),
        )
        .route(
            "/v1/catalog/models/:id/verify",
            post(http::catalog::verify_catalog_model),
        )
        // Lifecycle control (planning) — align to OpenAPI
        .route(
            "/v1/catalog/models/:id/state",
            post(http::control::set_model_state),
        )
        // Observability
        .route("/metrics", get(http::observability::metrics_endpoint))
        .with_state(state)
}
