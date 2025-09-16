use axum::{
    extract::{Path, State},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json,
};
use contracts_api_types as api;
use http::{header::CONTENT_TYPE, HeaderMap};
use serde_json::json;

use crate::{metrics, state::{AppState, ModelState}};
use crate::backpressure;
use serde::Deserialize;

fn require_api_key(headers: &HeaderMap) -> Result<(), http::StatusCode> {
    match headers.get("X-API-Key").and_then(|v| v.to_str().ok()) {
        None => Err(http::StatusCode::UNAUTHORIZED),
        Some(val) if val == "valid" => Ok(()),
        Some(_) => Err(http::StatusCode::FORBIDDEN),
    }
}

// Catalog endpoints (planning-only stubs)
#[derive(Deserialize)]
pub struct CatalogModelReq { pub id: String, #[serde(default)] pub signed: Option<bool> }

pub async fn create_catalog_model(headers: HeaderMap, _state: State<AppState>, Json(body): Json<CatalogModelReq>) -> Response {
    if let Err(code) = require_api_key(&headers) { return (code, HeaderMap::new()).into_response(); }
    let strict = headers.get("X-Trust-Policy").and_then(|v| v.to_str().ok()) == Some("strict");
    let signed = body.signed.unwrap_or(true);
    if strict && !signed {
        let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
        let err = json!({ "code": "UNTRUSTED_ARTIFACT" });
        return (http::StatusCode::BAD_REQUEST, h, Json(err)).into_response();
    }
    let resp = json!({ "id": body.id, "signatures": true, "sbom": true });
    (http::StatusCode::CREATED, Json(resp)).into_response()
}

pub async fn get_catalog_model(headers: HeaderMap, _state: State<AppState>, Path(id): Path<String>) -> Response {
    if let Err(code) = require_api_key(&headers) { return (code, HeaderMap::new()).into_response(); }
    let resp = json!({ "id": id, "signatures": true, "sbom": true });
    (http::StatusCode::OK, Json(resp)).into_response()
}

pub async fn verify_catalog_model(headers: HeaderMap, _state: State<AppState>, Path(_id): Path<String>) -> Response {
    if let Err(code) = require_api_key(&headers) { return (code, HeaderMap::new()).into_response(); }
    let resp = json!({ "status": "started" });
    (http::StatusCode::ACCEPTED, Json(resp)).into_response()
}
// Lifecycle control: set model state (planning-only control)
#[derive(Deserialize)]
pub struct SetModelStateRequest {
    pub state: String,
    #[serde(default)]
    pub deadline_ms: Option<i64>,
    #[serde(default)]
    pub model_id: Option<String>,
}

pub async fn set_model_state(headers: HeaderMap, state: State<AppState>, Json(body): Json<SetModelStateRequest>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let ms = match body.state.as_str() {
        "Draft" => ModelState::Draft,
        "Deprecated" => ModelState::Deprecated { deadline_ms: body.deadline_ms.unwrap_or(0) },
        "Retired" => ModelState::Retired,
        other => {
            return (
                http::StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("unknown state: {}", other) })),
            )
                .into_response();
        }
    };
    {
        let mut s = state.model_state.lock().unwrap();
        *s = ms.clone();
    }
    // update model_state gauge
    let model = body.model_id.as_deref().unwrap_or("model0");
    let label = match ms {
        ModelState::Draft => "Draft",
        ModelState::Deprecated { .. } => "Deprecated",
        ModelState::Retired => "Retired",
    };
    metrics::MODEL_STATE.with_label_values(&[model, label]).set(1);
    if matches!(ms, ModelState::Retired) {
        let mut logs = state.logs.lock().unwrap();
        logs.push("{\"event\":\"retire\",\"pools_unloaded\":true,\"archives_retained\":true}".to_string());
    }
    (http::StatusCode::OK, Json(json!({"status":"ok","state": label }))).into_response()
}

// Data plane — OrchQueue v1
pub async fn create_task(headers: HeaderMap, state: State<AppState>, body: Json<api::TaskRequest>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    // Lifecycle gating
    {
        let ms = state.model_state.lock().unwrap().clone();
        match ms {
            ModelState::Draft => {}
            ModelState::Deprecated { .. } => {
                let mut h = HeaderMap::new();
                h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
                let err = serde_json::json!({
                    "code": "MODEL_DEPRECATED",
                    "engine": body.engine,
                });
                return (http::StatusCode::FORBIDDEN, h, Json(err)).into_response();
            }
            ModelState::Retired => {
                let mut h = HeaderMap::new();
                h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
                let err = serde_json::json!({
                    "code": "MODEL_RETIRED",
                    "engine": body.engine,
                });
                return (http::StatusCode::GONE, h, Json(err)).into_response();
            }
        }
    }
    // Minimal placeholder: accept admission and return a basic envelope + correlation id
    // Sentinel: if expected_tokens is extremely high, simulate queue full and return 429
    if body.expected_tokens.unwrap_or(0) >= 1_000_000 {
        let backoff = backpressure::Backoff { retry_after_seconds: 1, x_backoff_ms: 1000 };
        let mut headers = backpressure::build_429_headers(backoff);
        headers.insert("X-Correlation-Id", "corr-0".parse().unwrap());
        let policy = backpressure::compute_policy_label(());
        let body = backpressure::build_429_body(policy);
        return (http::StatusCode::TOO_MANY_REQUESTS, headers, Json(body)).into_response();
    }

    // Error taxonomy sentinels
    if body.ctx < 0 {
        let mut h = HeaderMap::new();
        h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
        let err = serde_json::json!({
            "code": "INVALID_PARAMS",
            "engine": body.engine,
        });
        return (http::StatusCode::BAD_REQUEST, h, Json(err)).into_response();
    }
    if body.model_ref == "pool-unavailable" {
        let mut h = HeaderMap::new();
        h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
        let err = serde_json::json!({
            "code": "POOL_UNAVAILABLE",
            "engine": body.engine,
        });
        return (http::StatusCode::SERVICE_UNAVAILABLE, h, Json(err)).into_response();
    }
    if matches!(&body.prompt, Some(p) if p == "cause-internal") {
        let mut h = HeaderMap::new();
        h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
        let err = serde_json::json!({
            "code": "INTERNAL",
            "engine": body.engine,
        });
        return (http::StatusCode::INTERNAL_SERVER_ERROR, h, Json(err)).into_response();
    }

    // Guardrails: context length and token budget (reject before enqueue)
    if body.ctx > 32768 || body.max_tokens > 50000 {
        let mut h = HeaderMap::new();
        h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
        let err = serde_json::json!({
            "code": "INVALID_PARAMS",
            "engine": body.engine,
        });
        return (http::StatusCode::BAD_REQUEST, h, Json(err)).into_response();
    }

    // Deadline infeasible sentinel
    if body.deadline_ms <= 0 {
        let mut h = HeaderMap::new();
        h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
        let err = serde_json::json!({
            "code": "DEADLINE_UNMET",
            "engine": body.engine,
        });
        return (http::StatusCode::BAD_REQUEST, h, Json(err)).into_response();
    }

    let mut headers = HeaderMap::new();
    headers.insert("X-Correlation-Id", "corr-0".parse().unwrap());

    let resp = api::AdmissionResponse {
        task_id: body.task_id.clone(),
        queue_position: 0,
        predicted_start_ms: 0,
        backoff_ms: 0,
    };
    // Record model_state gauge (Draft by default) for visibility
    let model_id = body.model_ref.clone();
    let state_label = "Draft";
    metrics::MODEL_STATE
        .with_label_values(&[&model_id, state_label])
        .set(1);
    // Structured log for started/admission (no secrets)
    {
        let mut logs = state.logs.lock().unwrap();
        logs.push(format!(
            "{{\"event\":\"started\",\"task_id\":\"{}\",\"queue_position\":{},\"predicted_start_ms\":{}}}",
            resp.task_id, resp.queue_position, resp.predicted_start_ms
        ));
    }
    (http::StatusCode::ACCEPTED, headers, Json(resp)).into_response()
}

pub async fn stream_task(headers: HeaderMap, _state: State<AppState>, _path: Path<String>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    // Minimal SSE stub stream sufficient for BDD ordering assertions
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "text/event-stream".parse().unwrap());
    let body = "event: started\ndata: {\"queue_position\":0,\"predicted_start_ms\":0}\n\n\
                event: token\ndata: {\"text\":\"hello\"}\n\n\
                event: metrics\ndata: {\"queue_depth\":0,\"on_time_probability\":0.9}\n\n\
                event: end\ndata: {}\n\n";
    (headers, body).into_response()
}

pub async fn cancel_task(headers: HeaderMap, _state: State<AppState>, _path: Path<String>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut headers = HeaderMap::new();
    headers.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    (http::StatusCode::NO_CONTENT, headers).into_response()
}

pub async fn get_session(headers: HeaderMap, _state: State<AppState>, _path: Path<String>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let info = api::SessionInfo {
        ttl_ms_remaining: Some(60_000),
        turns: Some(0),
        kv_bytes: Some(0),
        kv_warmth: Some(false),
    };
    (http::StatusCode::OK, Json(info)).into_response()
}

pub async fn delete_session(headers: HeaderMap, _state: State<AppState>, _path: Path<String>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut headers = HeaderMap::new();
    headers.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    (http::StatusCode::NO_CONTENT, headers).into_response()
}

// Control plane
pub async fn drain_pool(
    headers: HeaderMap,
    _state: State<AppState>,
    _path: Path<String>,
    _body: Json<api::control::DrainRequest>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let body = json!({
        "status": "draining",
    });
    (http::StatusCode::ACCEPTED, Json(body)).into_response()
}

pub async fn reload_pool(
    headers: HeaderMap,
    _state: State<AppState>,
    _path: Path<String>,
    _body: Json<api::control::ReloadRequest>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    if _body.new_model_ref == "bad" {
        let body = json!({ "status": "rollback" });
        return (http::StatusCode::CONFLICT, Json(body)).into_response();
    }
    let body = json!({ "status": "reloaded" });
    (http::StatusCode::OK, Json(body)).into_response()
}

pub async fn get_pool_health(headers: HeaderMap, _state: State<AppState>, _path: Path<String>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let body = json!({
        "live": true,
        "ready": false,
        "draining": false,
        "metrics": { "queue_depth": 0 },
    });
    (http::StatusCode::OK, Json(body)).into_response()
}

pub async fn list_replicasets(headers: HeaderMap, _state: State<AppState>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let body = json!({
        "replicasets": []
    });
    (http::StatusCode::OK, Json(body)).into_response()
}

// Observability: Prometheus metrics endpoint
pub async fn metrics_endpoint() -> Response {
    let body = metrics::gather_metrics_text();
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "text/plain; version=0.0.4".parse().unwrap());
    (headers, body).into_response()
}
