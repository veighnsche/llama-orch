use axum::{extract::State, response::IntoResponse, Json};
use http::StatusCode;
use serde_json::json;

use crate::state::AppState;
use crate::domain::error::OrchestratorError as ErrO;

use super::types::{correlation_id_from, require_api_key};

pub async fn create_artifact(
    state: State<AppState>,
    Json(doc): Json<serde_json::Value>,
) -> Result<impl IntoResponse, ErrO> {
    if let Err(code) = require_api_key(&state.headers) {
        return Err(code);
    }
    let id = format!("sha256:{}", sha256::digest(doc.to_string()));
    {
        let mut guard = state.artifacts.lock().unwrap();
        guard.insert(id.clone(), doc);
    }
    let body = json!({
        "id": id,
        "kind": "doc",
    });
    Ok((StatusCode::CREATED, Json(body)))
}

pub async fn get_artifact(
    state: State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ErrO> {
    if let Err(code) = require_api_key(&state.headers) {
        return Err(code);
    }
    let guard = state.artifacts.lock().unwrap();
    if let Some(doc) = guard.get(&id) {
        return Ok((StatusCode::OK, Json(doc.clone())));
    }
    Ok((StatusCode::NOT_FOUND))
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
