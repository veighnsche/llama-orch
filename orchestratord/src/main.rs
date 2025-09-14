use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router, response::Response,
};
use contracts_api_types as api;

#[derive(Clone, Default)]
struct AppState;

// Data plane — OrchQueue v1
async fn create_task(_state: State<AppState>, _body: Json<api::TaskRequest>) -> Response {
    // AC: Handlers must remain unimplemented during pre-code phase.
    unimplemented!()
}

async fn stream_task(_state: State<AppState>, _path: Path<String>) -> Response {
    unimplemented!()
}

async fn cancel_task(_state: State<AppState>, _path: Path<String>) -> Response {
    unimplemented!()
}

async fn get_session(_state: State<AppState>, _path: Path<String>) -> Response {
    unimplemented!()
}

async fn delete_session(_state: State<AppState>, _path: Path<String>) -> Response {
    unimplemented!()
}

// Control plane
async fn drain_pool(
    _state: State<AppState>,
    _path: Path<String>,
    _body: Json<api::control::DrainRequest>,
) -> Response {
    unimplemented!()
}

async fn reload_pool(
    _state: State<AppState>,
    _path: Path<String>,
    _body: Json<api::control::ReloadRequest>,
) -> Response {
    unimplemented!()
}

async fn get_pool_health(_state: State<AppState>, _path: Path<String>) -> Response {
    unimplemented!()
}

async fn list_replicasets(_state: State<AppState>) -> Response {
    unimplemented!()
}

fn main() {
    // Build the router with all routes wired. Do not start the server in pre-code phase.
    let state = AppState;
    let app: Router<AppState> = Router::new()
        // Data plane
        .route("/v1/tasks", post(create_task))
        .route("/v1/tasks/:id/stream", get(stream_task))
        .route("/v1/tasks/:id/cancel", post(cancel_task))
        .route("/v1/sessions/:id", get(get_session).delete(delete_session))
        // Control plane
        .route("/v1/pools/:id/drain", post(drain_pool))
        .route("/v1/pools/:id/reload", post(reload_pool))
        .route("/v1/pools/:id/health", get(get_pool_health))
        .route("/v1/replicasets", get(list_replicasets))
        .with_state(state);

    // Avoid unused warning in pre-code phase
    let _ = app;
    println!("orchestratord routes wired (stubs)");
}
