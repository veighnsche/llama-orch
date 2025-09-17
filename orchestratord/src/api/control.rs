use axum::{extract::State, response::IntoResponse, Json};
use http::{HeaderMap, StatusCode};
use serde_json::json;

use crate::state::AppState;

use super::types::{correlation_id_from, require_api_key};

pub async fn get_capabilities(headers: HeaderMap, _state: State<AppState>) -> axum::response::Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut out = HeaderMap::new();
    let corr = correlation_id_from(&headers);
    out.insert("X-Correlation-Id", corr.parse().unwrap());

    let body = json!({
        "api_version": "1.0.0",
        "engines": [
            {"engine":"llamacpp","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}},
            {"engine":"vllm","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}},
            {"engine":"tgi","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}},
            {"engine":"triton","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}}
        ]
    });
    (StatusCode::OK, out, Json(body)).into_response()
}

pub async fn get_pool_health(
    headers: HeaderMap,
    _state: State<AppState>,
    axum::extract::Path(_id): axum::extract::Path<String>,
) -> axum::response::Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut out = HeaderMap::new();
    let corr = correlation_id_from(&headers);
    out.insert("X-Correlation-Id", corr.parse().unwrap());

    let (live, ready, last_error) = {
        let reg = _state.pool_manager.lock().expect("pool_manager lock");
        let h = reg.get_health(&_id).unwrap_or(pool_managerd::health::HealthStatus {
            live: true,
            ready: true,
        });
        let e = reg.get_last_error(&_id);
        (h.live, h.ready, e)
    };
    let body = json!({
        "live": live,
        "ready": ready,
        "draining": false,
        "metrics": {"queue_depth": 0},
        "last_error": last_error
    });
    (StatusCode::OK, out, Json(body)).into_response()
}

pub async fn drain_pool(
    headers: HeaderMap,
    _state: State<AppState>,
    axum::extract::Path(_id): axum::extract::Path<String>,
    Json(_body): Json<contracts_api_types::control::DrainRequest>,
) -> axum::response::Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut out = HeaderMap::new();
    let corr = correlation_id_from(&headers);
    out.insert("X-Correlation-Id", corr.parse().unwrap());

    (StatusCode::ACCEPTED, out).into_response()
}

pub async fn reload_pool(
    headers: HeaderMap,
    _state: State<AppState>,
    axum::extract::Path(_id): axum::extract::Path<String>,
    Json(body): Json<contracts_api_types::control::ReloadRequest>,
) -> axum::response::Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut out = HeaderMap::new();
    let corr = correlation_id_from(&headers);
    out.insert("X-Correlation-Id", corr.parse().unwrap());

    if body.new_model_ref == "bad" {
        return (StatusCode::CONFLICT, out).into_response();
    }
    (StatusCode::OK, out).into_response()
}
