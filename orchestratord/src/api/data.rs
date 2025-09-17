use axum::{extract::State, response::IntoResponse, Json};
use http::{HeaderMap, StatusCode};
use serde_json::json;

use crate::state::AppState;
use contracts_api_types as api;

use super::types::{correlation_id_from, require_api_key};

pub async fn get_session(
    headers: HeaderMap,
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut out = HeaderMap::new();
    let corr = correlation_id_from(&headers);
    out.insert("X-Correlation-Id", corr.parse().unwrap());

    let mut guard = state.sessions.lock().unwrap();
    let entry = guard.entry(id).or_insert_with(|| crate::state::SessionInfo {
        ttl_ms_remaining: 600_000,
        turns: 1,
        kv_bytes: 0,
        kv_warmth: false,
    });

    let body = json!({
        "ttl_ms_remaining": entry.ttl_ms_remaining,
        "turns": entry.turns,
        "kv_bytes": entry.kv_bytes,
        "kv_warmth": entry.kv_warmth,
    });
    (StatusCode::OK, out, Json(body)).into_response()
}

pub async fn delete_session(
    headers: HeaderMap,
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> axum::response::Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut out = HeaderMap::new();
    let corr = correlation_id_from(&headers);
    out.insert("X-Correlation-Id", corr.parse().unwrap());

    let mut guard = state.sessions.lock().unwrap();
    guard.remove(&id);
    (StatusCode::NO_CONTENT, out).into_response()
}

pub async fn create_task(
    headers: HeaderMap,
    state: State<AppState>,
    Json(body): Json<api::TaskRequest>,
) -> axum::response::Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut out = HeaderMap::new();
    let corr = correlation_id_from(&headers);
    out.insert("X-Correlation-Id", corr.parse().unwrap());

    // Error taxonomy sentinels
    if body.ctx < 0 {
        let env = api::ErrorEnvelope {
            code: api::ErrorKind::InvalidParams,
            message: Some("ctx must be >= 0".into()),
            engine: Some(body.engine),
            retriable: None,
            retry_after_ms: None,
            policy_label: None,
        };
        return (StatusCode::BAD_REQUEST, out, Json(env)).into_response();
    }
    if body.deadline_ms <= 0 {
        let env = api::ErrorEnvelope {
            code: api::ErrorKind::DeadlineUnmet,
            message: Some("deadline_ms must be > 0".into()),
            engine: Some(body.engine),
            retriable: None,
            retry_after_ms: None,
            policy_label: None,
        };
        return (StatusCode::BAD_REQUEST, out, Json(env)).into_response();
    }
    if body.model_ref == "pool-unavailable" {
        let env = api::ErrorEnvelope {
            code: api::ErrorKind::PoolUnavailable,
            message: Some("pool unavailable".into()),
            engine: Some(body.engine),
            retriable: Some(true),
            retry_after_ms: Some(1000),
            policy_label: Some("retry".into()),
        };
        return (StatusCode::SERVICE_UNAVAILABLE, out, Json(env)).into_response();
    }
    if body.prompt.as_deref() == Some("cause-internal") {
        let env = api::ErrorEnvelope {
            code: api::ErrorKind::Internal,
            message: Some("internal error".into()),
            engine: Some(body.engine),
            retriable: None,
            retry_after_ms: None,
            policy_label: None,
        };
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            out,
            Json(env),
        )
            .into_response();
    }
    if body.expected_tokens.unwrap_or(0) >= 1_000_000 {
        out.insert("Retry-After", "1".parse().unwrap());
        out.insert("X-Backoff-Ms", "1000".parse().unwrap());
        let env = api::ErrorEnvelope {
            code: api::ErrorKind::AdmissionReject,
            message: Some("queue full policies applied".into()),
            engine: Some(body.engine),
            retriable: Some(true),
            retry_after_ms: Some(1000),
            policy_label: Some("reject".into()),
        };
        return (StatusCode::TOO_MANY_REQUESTS, out, Json(env)).into_response();
    }

    // Success path (stub ETA/position)
    let admission = api::AdmissionResponse {
        task_id: body.task_id.clone(),
        queue_position: 3,
        predicted_start_ms: 420,
        backoff_ms: 0,
    };

    // Emit a simple log line for BDD assertions
    let mut lg = state.logs.lock().unwrap();
    lg.push(format!(
        "{{\"queue_position\":{},\"predicted_start_ms\":{}}}",
        admission.queue_position, admission.predicted_start_ms
    ));

    (StatusCode::ACCEPTED, out, Json(admission)).into_response()
}

pub async fn stream_task(
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
    out.insert("Content-Type", "text/event-stream".parse().unwrap());

    let sse = [
        "event: started",
        &format!("data: {{\"queue_position\":{},\"predicted_start_ms\":{}}}", 3, 420),
        "",
        "event: token",
        "data: {\"t\":\"Hello\",\"i\":0}",
        "",
        "event: metrics",
        "data: {\"queue_depth\":1,\"on_time_probability\":0.99,\"kv_warmth\":false,\"tokens_budget_remaining\":0,\"time_budget_remaining_ms\":600000,\"cost_budget_remaining\":0.0}",
        "",
        "event: end",
        "data: {\"tokens_out\":1,\"decode_ms\":5}",
        "",
    ]
    .join("\n");

    (StatusCode::OK, out, sse).into_response()
}

pub async fn cancel_task(
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
    (StatusCode::NO_CONTENT, out).into_response()
}
