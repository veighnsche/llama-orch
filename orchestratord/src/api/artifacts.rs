use axum::{extract::State, response::IntoResponse, Json};
use http::StatusCode;
use serde_json::json;

use crate::state::AppState;
use crate::domain::error::OrchestratorError as ErrO;

pub async fn create_artifact(
    state: State<AppState>,
    Json(doc): Json<serde_json::Value>,
) -> Result<impl IntoResponse, ErrO> {
    let id = format!("sha256:{}", sha256::digest(doc.to_string()));
    {
        let mut guard = state.artifacts.lock().unwrap();
        guard.insert(id.clone(), doc);
    }
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
    let guard = state.artifacts.lock().unwrap();
    if let Some(doc) = guard.get(&id) {
        let resp = (StatusCode::OK, Json(doc.clone())).into_response();
        return Ok(resp);
    }
    let resp = StatusCode::NOT_FOUND.into_response();
    Ok(resp)
}

mod sha256 {
    use sha2::{Digest, Sha256};
    pub fn digest(s: String) -> String {
        let mut hasher = Sha256::new();
        hasher.update(s.as_bytes());
        let bytes = hasher.finalize();
        hex::encode(bytes)
    }
}
