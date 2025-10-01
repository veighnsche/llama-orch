use axum::{extract::State, response::IntoResponse, Json};
use http::{HeaderMap, StatusCode};
use serde_json::json;
use std::sync::Arc;

use crate::domain::error::OrchestratorError as ErrO;
use crate::state::AppState;

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
    // Call pool-managerd HTTP API
    let status = state.pool_manager.get_pool_status(&id).await.unwrap_or_else(|_| {
        crate::clients::pool_manager::PoolStatus {
            pool_id: id.clone(),
            live: false,
            ready: false,
            active_leases: 0,
            slots_total: 0,
            slots_free: 0,
        }
    });
    let (live, ready) = (status.live, status.ready);
    let last_error: Option<String> = None; // TODO: Get from daemon if needed
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
        &[("model_id", body.new_model_ref.as_str()), ("state", "loaded")],
        1,
    );
    Ok(StatusCode::OK)
}

/// Worker registration endpoint.
///
/// Authentication is enforced by bearer_auth_middleware.
/// Middleware has already validated the Bearer token.
#[derive(serde::Deserialize)]
pub struct RegisterWorkerBody {
    pub pool_id: Option<String>,
    pub replica_id: Option<String>,
}

pub async fn register_worker(
    state: State<AppState>,
    body: Option<Json<RegisterWorkerBody>>,
) -> Result<impl IntoResponse, ErrO> {
    // Authentication already handled by bearer_auth_middleware
    // Log the registration event
    let mut lg = state.logs.lock().unwrap();
    lg.push(format!("{{\"event\":\"worker_register\"}}"));

    // For scaffolding: bind a mock adapter for the provided pool
    let pool_id =
        body.as_ref().and_then(|b| b.pool_id.clone()).unwrap_or_else(|| "default".to_string());
    let replica_id =
        body.as_ref().and_then(|b| b.replica_id.clone()).unwrap_or_else(|| "r0".to_string());
    #[cfg(feature = "mock-adapters")]
    {
        let mock = worker_adapters_mock::MockAdapter::default();
        state.adapter_host.bind(pool_id.clone(), replica_id.clone(), Arc::new(mock));
    }

    Ok((StatusCode::OK, Json(json!({"ok": true, "pool_id": pool_id, "replica_id": replica_id}))))
}

/// v2 purge endpoint (stub): accept request and return 202 to indicate purge scheduled.
/// Pre-1.0: semantics subject to change; body is intentionally untyped here to avoid
/// tight coupling with evolving contracts_api_types.
///
/// TODO(ARCH-CHANGE): This is a stub implementation. Per ARCHITECTURE_CHANGE_PLAN.md:
/// - Implement actual purge logic via pool-managerd API
/// - Add authentication/authorization checks
/// - Implement proper async job tracking
/// - Add audit logging for purge operations
/// - Return job_id for tracking purge progress
/// See: SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #8 (pool-managerd auth)
pub async fn purge_pool_v2(
    _state: State<AppState>,
    axum::extract::Path(_id): axum::extract::Path<String>,
    _body: Option<Json<serde_json::Value>>,
) -> Result<impl IntoResponse, ErrO> {
    Ok(StatusCode::ACCEPTED)
}
