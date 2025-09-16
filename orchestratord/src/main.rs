mod metrics;

use axum::{
    extract::{Path, State},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use contracts_api_types as api;
use http::{header::CONTENT_TYPE, HeaderMap};

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
        // Observability
        .route("/metrics", get(metrics_endpoint))
        .with_state(state);

    // Avoid unused warning in pre-code phase
    let _ = app;
    println!("orchestratord routes wired (stubs)");
}

// Observability: Prometheus metrics endpoint
async fn metrics_endpoint() -> Response {
    let body = crate::metrics::gather_metrics_text();
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "text/plain; version=0.0.4".parse().unwrap());
    (headers, body).into_response()
}

#[cfg(test)]
mod tests {
    use std::fs;

    #[test]
    fn metrics_text_includes_required_names() {
        // Read linter config to get required names
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(1)
            .unwrap()
            .to_path_buf();
        let lint_path = root.join("ci/metrics.lint.json");
        let lint_text = fs::read_to_string(&lint_path).expect("read metrics.lint.json");
        let lint_json: serde_json::Value =
            serde_json::from_str(&lint_text).expect("parse metrics.lint.json");

        let text = crate::metrics::gather_metrics_text();
        let names = lint_json["required_metrics"].as_array().unwrap();
        for m in names {
            let name = m["name"].as_str().unwrap();
            // Prometheus text format includes a TYPE line for each registered metric
            let needle = format!("# TYPE {} ", name);
            assert!(
                text.contains(&needle) || text.contains(name),
                "missing metric {} in /metrics text",
                name
            );
        }
    }
}
