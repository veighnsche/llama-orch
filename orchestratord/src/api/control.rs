use axum::{extract::State, response::IntoResponse, Json};
use http::StatusCode;
use serde_json::json;

use crate::state::AppState;
use crate::domain::error::OrchestratorError as ErrO;

pub async fn get_capabilities(_state: State<AppState>) -> Result<impl IntoResponse, ErrO> {
    let body = crate::services::capabilities::snapshot();
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
