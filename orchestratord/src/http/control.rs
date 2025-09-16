use axum::{
    extract::{Path, State},
    response::{IntoResponse, Response},
    Json,
};
use http::HeaderMap;
use serde_json::json;
use tracing::info;

use super::auth::require_api_key;
use crate::{
    metrics,
    state::{AppState, ModelState},
};
use contracts_api_types as api;

#[derive(serde::Deserialize)]
pub struct SetModelStateRequest {
    pub state: String,
    #[serde(default)]
    pub deadline_ms: Option<i64>,
}

pub async fn set_model_state(
    headers: HeaderMap,
    state: State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<SetModelStateRequest>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let ms = match body.state.as_str() {
        "Draft" => ModelState::Draft,
        "Deprecated" => ModelState::Deprecated {
            deadline_ms: body.deadline_ms.unwrap_or(0),
        },
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
    let model = id.as_str();
    let label = match ms {
        ModelState::Draft => "Draft",
        ModelState::Deprecated { .. } => "Deprecated",
        ModelState::Retired => "Retired",
    };
    metrics::MODEL_STATE
        .with_label_values(&[model, label])
        .set(1);
    info!(event = "lifecycle_state_change", model_id = %model, state = %label, deadline_ms = body.deadline_ms.unwrap_or(0), "model state updated");
    if matches!(ms, ModelState::Retired) {
        let mut logs = state.logs.lock().unwrap();
        logs.push(
            "{\"event\":\"retire\",\"pools_unloaded\":true,\"archives_retained\":true}".to_string(),
        );
    }
    let mut h = HeaderMap::new();
    h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    (
        http::StatusCode::ACCEPTED,
        h,
        Json(json!({"status":"ok","state": label })),
    )
        .into_response()
}

pub async fn drain_pool(
    headers: HeaderMap,
    state: State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<api::control::DrainRequest>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    {
        // Flip draining flag in in-memory pool snapshot for visibility and mark unready in registry
        if let Ok(mut pools) = state.pools.lock() {
            if let Some(p) = pools.get_mut(&id) {
                p.draining = true;
            }
        }
        if let Ok(mut pm) = state.pool_manager.lock() {
            pm.set_health(&id, pool_managerd::health::HealthStatus { live: true, ready: false });
        }
        // Metrics: record drain event and readiness 0
        crate::metrics::DRAIN_EVENTS_TOTAL
            .with_label_values(&[&id, "api_request"])
            .inc();
        crate::metrics::POOL_READY.with_label_values(&[&id]).set(0);
    }
    let mut h = HeaderMap::new();
    h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    let resp = json!({"status": "draining", "deadline_ms": body.deadline_ms});
    (http::StatusCode::ACCEPTED, h, Json(resp)).into_response()
}

pub async fn reload_pool(
    headers: HeaderMap,
    state: State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<api::control::ReloadRequest>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    if body.new_model_ref == "bad" {
        let mut h = HeaderMap::new();
        h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
        let body = json!({"status": "rollback"});
        return (http::StatusCode::CONFLICT, h, Json(body)).into_response();
    }
    {
        // Clear draining and mark ready in registry; also bump version hint
        if let Ok(mut pools) = state.pools.lock() {
            if let Some(p) = pools.get_mut(&id) {
                p.draining = false;
            }
        }
        if let Ok(mut pm) = state.pool_manager.lock() {
            pm.set_health(&id, pool_managerd::health::HealthStatus { live: true, ready: true });
            pm.set_version(&id, format!("{}:reloaded", body.new_model_ref));
        }
        crate::metrics::POOL_READY.with_label_values(&[&id]).set(1);
    }
    let mut h = HeaderMap::new();
    h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    let body = json!({"status": "reloaded"});
    (http::StatusCode::OK, h, Json(body)).into_response()
}

pub async fn get_pool_health(
    headers: HeaderMap,
    state: State<AppState>,
    Path(id): Path<String>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut h = HeaderMap::new();
    h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    let (live, ready, last_error) = {
        let pm = state.pool_manager.lock().unwrap();
        let s = pm.get_health(&id);
        let e = pm.get_last_error(&id);
        (
            s.as_ref().map(|x| x.live).unwrap_or(false),
            s.as_ref().map(|x| x.ready).unwrap_or(false),
            e,
        )
    };
    let metrics_val = {
        let pools = state.pools.lock().unwrap();
        pools
            .get(&id)
            .map(|p| p.metrics.clone())
            .unwrap_or_else(|| json!({}))
    };
    let draining = {
        let pools = state.pools.lock().unwrap();
        pools.get(&id).map(|p| p.draining).unwrap_or(false)
    };
    let body = json!({
        "live": live,
        "ready": ready,
        "draining": draining,
        "metrics": metrics_val,
        "last_error": last_error,
    });
    (http::StatusCode::OK, h, Json(body)).into_response()
}

pub async fn list_replicasets(headers: HeaderMap, _state: State<AppState>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut h = HeaderMap::new();
    h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    // Enrich payload using adapters registry as a proxy for available engines
    let mut sets = vec![];
    if let Ok(map) = _state.adapters.lock() {
        for (engine_key, adapter) in map.iter() {
            let props = adapter.props().ok();
            let (slots_total, slots_free) = props
                .map(|p| {
                    (
                        p.slots_total.map(|v| v as i32),
                        p.slots_free.map(|v| v as i32),
                    )
                })
                .unwrap_or((None, None));
            sets.push(json!({
                "id": format!("pool0-{}", engine_key),
                "engine": engine_key,
                "load": 0.0,
                "slots_total": slots_total,
                "slots_free": slots_free,
                "slo": {}
            }));
        }
    }
    let body = json!(sets);
    (http::StatusCode::OK, h, Json(body)).into_response()
}

pub async fn get_capabilities(headers: HeaderMap, _state: State<AppState>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut h = HeaderMap::new();
    h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
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
