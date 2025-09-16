pub mod admission;
pub mod metrics;

pub mod domain;
pub mod services;
pub mod http;
pub mod sse;
pub mod backpressure;
pub mod placement;
pub mod state;
pub mod errors;

use axum::{routing::{get, post}, Router};

/// Build the Axum router for the orchestrator.
/// Handlers are stubs during the pre-code phase; routes are wired for provider verification and DX.
pub fn build_app() -> Router<state::AppState> {
    let state = state::default_state();
    Router::new()
        // Data plane
        .route("/v1/tasks", post(http::handlers::create_task))
        .route("/v1/tasks/:id/stream", get(http::handlers::stream_task))
        .route("/v1/tasks/:id/cancel", post(http::handlers::cancel_task))
        .route("/v1/sessions/:id", get(http::handlers::get_session).delete(http::handlers::delete_session))
        // Control plane
        .route("/v1/pools/:id/drain", post(http::handlers::drain_pool))
        .route("/v1/pools/:id/reload", post(http::handlers::reload_pool))
        .route("/v1/pools/:id/health", get(http::handlers::get_pool_health))
        .route("/v1/replicasets", get(http::handlers::list_replicasets))
        // Observability
        .route("/metrics", get(http::handlers::metrics_endpoint))
        .with_state(state)
}
