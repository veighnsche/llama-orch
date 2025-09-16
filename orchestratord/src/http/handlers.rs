use axum::{
    extract::{Path, State},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json,
};
use contracts_api_types as api;
use http::{header::CONTENT_TYPE, HeaderMap};

use crate::{metrics, state::AppState};

// Data plane — OrchQueue v1
pub async fn create_task(_state: State<AppState>, _body: Json<api::TaskRequest>) -> Response {
    // AC: Handlers must remain unimplemented during pre-code phase.
    unimplemented!()
}

pub async fn stream_task(_state: State<AppState>, _path: Path<String>) -> Response {
    unimplemented!()
}

pub async fn cancel_task(_state: State<AppState>, _path: Path<String>) -> Response {
    unimplemented!()
}

pub async fn get_session(_state: State<AppState>, _path: Path<String>) -> Response {
    unimplemented!()
}

pub async fn delete_session(_state: State<AppState>, _path: Path<String>) -> Response {
    unimplemented!()
}

// Control plane
pub async fn drain_pool(
    _state: State<AppState>,
    _path: Path<String>,
    _body: Json<api::control::DrainRequest>,
) -> Response {
    unimplemented!()
}

pub async fn reload_pool(
    _state: State<AppState>,
    _path: Path<String>,
    _body: Json<api::control::ReloadRequest>,
) -> Response {
    unimplemented!()
}

pub async fn get_pool_health(_state: State<AppState>, _path: Path<String>) -> Response {
    unimplemented!()
}

pub async fn list_replicasets(_state: State<AppState>) -> Response {
    unimplemented!()
}

// Observability: Prometheus metrics endpoint
pub async fn metrics_endpoint() -> Response {
    let body = metrics::gather_metrics_text();
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "text/plain; version=0.0.4".parse().unwrap());
    (headers, body).into_response()
}
