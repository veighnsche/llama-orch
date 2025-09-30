use axum::{extract::State, response::IntoResponse, Json};
use http::StatusCode;
use serde_json::json;

use crate::domain::error::OrchestratorError as ErrO;
use crate::state::AppState;

//! Artifact storage API endpoints
//
// TODO(SECURITY): Add authentication to artifact endpoints using auth-min
//
// Artifact endpoints allow uploading and downloading artifacts (plans, diffs, traces).
// These should require authentication to prevent unauthorized access.
//
// Endpoints requiring auth:
// - POST /v2/artifacts - Upload artifact
// - GET /v2/artifacts/{id} - Download artifact
//
// Should use auth-min based middleware for Bearer token validation.
//
// See: .docs/PHASE5_FIX_CHECKLIST.md Task 11
// See: .specs/12_auth-min-hardening.md (SEC-AUTH-3001)
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
