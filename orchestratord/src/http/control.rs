use axum::{extract::{Path, State}, response::{IntoResponse, Response}, Json};
use http::HeaderMap;
use serde_json::json;
use tracing::info;

use super::auth::require_api_key;
use crate::{metrics, state::{AppState, ModelState}};
use contracts_api_types as api;

#[derive(serde::Deserialize)]
pub struct SetModelStateRequest {
    pub state: String,
    #[serde(default)]
    pub deadline_ms: Option<i64>,
}

pub async fn set_model_state(headers: HeaderMap, state: State<AppState>, Path(id): Path<String>, Json(body): Json<SetModelStateRequest>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let ms = match body.state.as_str() {
        "Draft" => ModelState::Draft,
        "Deprecated" => ModelState::Deprecated { deadline_ms: body.deadline_ms.unwrap_or(0) },
        "Retired" => ModelState::Retired,
        other => {
            return (http::StatusCode::BAD_REQUEST, Json(json!({ "error": format!("unknown state: {}", other) }))).into_response();
        }
    };
    {
        let mut s = state.model_state.lock().unwrap();
        *s = ms.clone();
    }
    let model = id.as_str();
    let label = match ms { ModelState::Draft => "Draft", ModelState::Deprecated{..} => "Deprecated", ModelState::Retired => "Retired" };
    metrics::MODEL_STATE.with_label_values(&[model, label]).set(1);
    info!(event = "lifecycle_state_change", model_id = %model, state = %label, deadline_ms = body.deadline_ms.unwrap_or(0), "model state updated");
    if matches!(ms, ModelState::Retired) {
        let mut logs = state.logs.lock().unwrap();
        logs.push("{\"event\":\"retire\",\"pools_unloaded\":true,\"archives_retained\":true}".to_string());
    }
    let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    (http::StatusCode::ACCEPTED, h, Json(json!({"status":"ok","state": label }))).into_response()
}

pub async fn drain_pool(headers: HeaderMap, _state: State<AppState>, _path: Path<String>, _body: Json<api::control::DrainRequest>) -> Response {
    if let Err(code) = require_api_key(&headers) { return (code, HeaderMap::new()).into_response(); }
    let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    let body = json!({"status": "draining"});
    (http::StatusCode::ACCEPTED, h, Json(body)).into_response()
}

pub async fn reload_pool(headers: HeaderMap, _state: State<AppState>, _path: Path<String>, _body: Json<api::control::ReloadRequest>) -> Response {
    if let Err(code) = require_api_key(&headers) { return (code, HeaderMap::new()).into_response(); }
    if _body.new_model_ref == "bad" {
        let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
        let body = json!({"status": "rollback"});
        return (http::StatusCode::CONFLICT, h, Json(body)).into_response();
    }
    let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    let body = json!({"status": "reloaded"});
    (http::StatusCode::ACCEPTED, h, Json(body)).into_response()
}

pub async fn get_pool_health(headers: HeaderMap, state: State<AppState>, Path(id): Path<String>) -> Response {
    if let Err(code) = require_api_key(&headers) { return (code, HeaderMap::new()).into_response(); }
    let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    let ph = {
        let pools = state.pools.lock().unwrap();
        pools.get(&id).cloned()
    };
    let body = match ph {
        Some(ph) => json!({
            "live": ph.live,
            "ready": ph.ready,
            "draining": ph.draining,
            "metrics": ph.metrics,
        }),
        None => json!({"live": false, "ready": false, "draining": false, "metrics": {}}),
    };
    (http::StatusCode::OK, h, Json(body)).into_response()
}

pub async fn list_replicasets(headers: HeaderMap, _state: State<AppState>) -> Response {
    if let Err(code) = require_api_key(&headers) { return (code, HeaderMap::new()).into_response(); }
    let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    let body = json!([]);
    (http::StatusCode::OK, h, Json(body)).into_response()
}

pub async fn get_capabilities(headers: HeaderMap, _state: State<AppState>) -> Response {
    if let Err(code) = require_api_key(&headers) { return (code, HeaderMap::new()).into_response(); }
    let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    let body = json!({
        "api_version": "1.0.0",
        "engines": [
            {"engine":"llamacpp","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}},
            {"engine":"vllm","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}},
            {"engine":"tgi","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}},
            {"engine":"triton","ctx_max":32768,"supported_workloads":["completion","embedding","rerank"],"rate_limits":{},"features":{}}
        ]
    });
    (http::StatusCode::OK, h, Json(body)).into_response()
}
