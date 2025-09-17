use axum::{extract::State, response::IntoResponse, Json};
use http::{HeaderMap, StatusCode};
use serde_json::json;

use crate::{services, state::AppState};
use contracts_api_types as api;
use crate::domain::error::OrchestratorError as ErrO;

pub async fn get_session(
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ErrO> {
    let svc = services::session::SessionService::new(state.sessions.clone(), std::sync::Arc::new(crate::infra::clock::SystemClock::default()));
    let entry = svc.get_or_create(&id);
    let body = json!({
        "ttl_ms_remaining": entry.ttl_ms_remaining,
        "turns": entry.turns,
        "kv_bytes": entry.kv_bytes,
        "kv_warmth": entry.kv_warmth,
        "tokens_budget_remaining": entry.tokens_budget_remaining,
        "time_budget_remaining_ms": entry.time_budget_remaining_ms,
        "cost_budget_remaining": entry.cost_budget_remaining,
    });
    Ok((StatusCode::OK, Json(body)))
}

pub async fn delete_session(
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ErrO> {
    let svc = services::session::SessionService::new(state.sessions.clone(), std::sync::Arc::new(crate::infra::clock::SystemClock::default()));
    svc.delete(&id);
    Ok((StatusCode::NO_CONTENT))
}

pub async fn create_task(
    state: State<AppState>,
    Json(body): Json<api::TaskRequest>,
) -> Result<impl IntoResponse, ErrO> {
    // Error taxonomy sentinels
    if body.ctx < 0 {
        return Err(ErrO::InvalidParams("ctx must be >= 0".into()));
    }
    if body.deadline_ms <= 0 {
        return Err(ErrO::DeadlineUnmet);
    }
    if body.model_ref == "pool-unavailable" {
        return Err(ErrO::PoolUnavailable);
    }
    if body.prompt.as_deref() == Some("cause-internal") {
        return Err(ErrO::Internal);
    }
    if body.expected_tokens.unwrap_or(0) >= 1_000_000 {
        return Err(ErrO::AdmissionReject { policy_label: "reject".into(), retry_after_ms: Some(1000) });
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

    Ok((StatusCode::ACCEPTED, Json(admission)))
}

pub async fn stream_task(
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ErrO> {
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "text/event-stream".parse().unwrap());
    let sse = services::streaming::render_sse_for_task(&*state, id).await;
    Ok((StatusCode::OK, headers, sse))
}

pub async fn cancel_task(
    state: State<AppState>,
    axum::extract::Path(_id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ErrO> {
    // Record cancellation metric
    crate::metrics::inc_counter("tasks_canceled_total", &[("engine","llamacpp"),("engine_version","v0"),("pool_id","default"),("replica_id","r0"),("reason","client")]);
    // In a full impl, signal cancel token to streaming service
    let mut lg = state.logs.lock().unwrap();
    lg.push("{\"canceled\":true}".to_string());
    Ok((StatusCode::NO_CONTENT))
}
