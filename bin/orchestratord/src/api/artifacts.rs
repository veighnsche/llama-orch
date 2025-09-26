use axum::{extract::State, response::IntoResponse, Json};
use http::StatusCode;
use serde_json::json;

use crate::domain::error::OrchestratorError as ErrO;
use crate::state::AppState;

pub async fn create_artifact(
    state: State<AppState>,
    Json(doc): Json<serde_json::Value>,
) -> Result<impl IntoResponse, ErrO> {
    let id = crate::services::artifacts::put(&*state, doc).map_err(|_| ErrO::Internal)?;
    let body = json!({
        "id": id,
        "kind": "doc",
    });
    Ok((StatusCode::CREATED, Json(body)).into_response())
}

pub async fn get_artifact(
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ErrO> {
    if let Ok(Some(doc)) = crate::services::artifacts::get(&*state, &id) {
        let resp = (StatusCode::OK, Json(doc)).into_response();
        return Ok(resp);
    }
    let resp = StatusCode::NOT_FOUND.into_response();
    Ok(resp)
}
