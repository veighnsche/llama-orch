use axum::{routing::{get, post, delete}, Router};
use crate::{api, state::AppState};

pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Capabilities
        .route("/v1/capabilities", get(api::control::get_capabilities))
        // Control
        .route("/v1/pools/:id/health", get(api::control::get_pool_health))
        .route("/v1/pools/:id/drain", post(api::control::drain_pool))
        .route("/v1/pools/:id/reload", post(api::control::reload_pool))
        // Data
        .route("/v1/tasks", post(api::data::create_task))
        .route("/v1/tasks/:id/stream", get(api::data::stream_task))
        .route("/v1/tasks/:id/cancel", post(api::data::cancel_task))
        // Sessions
        .route("/v1/sessions/:id", get(api::data::get_session))
        .route("/v1/sessions/:id", delete(api::data::delete_session))
        // Artifacts
        .route("/v1/artifacts", post(api::artifacts::create_artifact))
        .route("/v1/artifacts/:id", get(api::artifacts::get_artifact))
        // Observability
        .route("/metrics", get(api::observability::metrics_endpoint))
        .with_state(state)
}
