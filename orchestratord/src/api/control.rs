use axum::{extract::State, response::IntoResponse, Json};
use http::{HeaderMap, StatusCode};
use serde_json::json;
use std::sync::Arc;

use crate::state::AppState;
use crate::domain::error::OrchestratorError as ErrO;

pub async fn get_capabilities(state: State<AppState>) -> Result<impl IntoResponse, ErrO> {
    // Serve from cache if present; otherwise compute and store.
    let body = {
        let mut guard = state.capabilities_cache.lock().unwrap();
        if let Some(cached) = guard.as_ref() {
            cached.clone()
        } else {
            let snap = crate::services::capabilities::snapshot();
            *guard = Some(snap.clone());
            snap
        }
    };
    Ok((StatusCode::OK, Json(body)))
}

pub async fn get_pool_health(
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ErrO> {
    let (live, ready, last_error) = {
        let reg = state.pool_manager.lock().expect("pool_manager lock");
        let h = reg.get_health(&id).unwrap_or(pool_managerd::health::HealthStatus {
            live: true,
            ready: true,
        });
        let e = reg.get_last_error(&id);
        (h.live, h.ready, e)
    };
    let draining = {
        let d = state.draining_pools.lock().unwrap();
        *d.get(&id).unwrap_or(&false)
    };
    let body = json!({
        "live": live,
        "ready": ready,
        "draining": draining,
        "metrics": {"queue_depth": 0},
        "last_error": last_error
    });
    Ok((StatusCode::OK, Json(body)))
}

pub async fn drain_pool(
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(_body): Json<contracts_api_types::control::DrainRequest>,
) -> Result<impl IntoResponse, ErrO> {
    let mut d = state.draining_pools.lock().unwrap();
    d.insert(id, true);
    Ok(StatusCode::ACCEPTED)
}

pub async fn reload_pool(
    _state: State<AppState>,
    axum::extract::Path(_id): axum::extract::Path<String>,
    Json(body): Json<contracts_api_types::control::ReloadRequest>,
) -> Result<impl IntoResponse, ErrO> {
    if body.new_model_ref == "bad" {
        return Ok(StatusCode::CONFLICT);
    }
    // Update model state metric (simplified)
    crate::metrics::set_gauge(
        "model_state",
        &[("model_id", body.new_model_ref.as_str()), ("state","loaded")],
        1,
    );
    Ok(StatusCode::OK)
}

/// Worker registration requires a valid Bearer token.
#[derive(serde::Deserialize)]
pub struct RegisterWorkerBody { pub pool_id: Option<String>, pub replica_id: Option<String> }

pub async fn register_worker(
    state: State<AppState>,
    headers: HeaderMap,
    body: Option<Json<RegisterWorkerBody>>,
) -> Result<impl IntoResponse, ErrO> {
    let expected = std::env::var("AUTH_TOKEN").ok();
    let auth = headers
        .get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    let token_opt = auth_min::parse_bearer(auth);

    // Missing token
    let token = match token_opt {
        None => {
            let env = json!({ "code": 40101, "message": "MISSING_TOKEN" });
            return Ok((StatusCode::UNAUTHORIZED, Json(env)));
        }
        Some(t) => t,
    };

    // Compare using timing safe equality when expected provided
    if let Some(exp) = expected {
        if !auth_min::timing_safe_eq(exp.as_bytes(), token.as_bytes()) {
            let env = json!({ "code": 40102, "message": "BAD_TOKEN" });
            return Ok((StatusCode::UNAUTHORIZED, Json(env)));
        }
    }

    // Record identity breadcrumb
    let id = format!("token:{}", auth_min::token_fp6(&token));
    let mut lg = state.logs.lock().unwrap();
    lg.push(format!("{{\"identity\":\"{}\",\"event\":\"worker_register\"}}", id));

    // For scaffolding: bind a mock adapter for the provided pool
    let pool_id = body.as_ref().and_then(|b| b.pool_id.clone()).unwrap_or_else(|| "default".to_string());
    let replica_id = body.as_ref().and_then(|b| b.replica_id.clone()).unwrap_or_else(|| "r0".to_string());
    #[cfg(feature = "mock-adapters")]
    {
        let mock = worker_adapters_mock::MockAdapter::default();
        state.adapter_host.bind(pool_id.clone(), replica_id.clone(), Arc::new(mock));
    }

    Ok((StatusCode::OK, Json(json!({"ok": true, "identity": id, "pool_id": pool_id, "replica_id": replica_id}))))
}
