use axum::{extract::State, response::IntoResponse, Json};
use http::{HeaderMap, StatusCode};
use serde_json::json;

use crate::state::AppState;

use super::types::{correlation_id_from, require_api_key};

pub async fn create_artifact(
    headers: HeaderMap,
    state: State<AppState>,
    Json(doc): Json<serde_json::Value>,
) -> axum::response::Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let mut out = HeaderMap::new();
    let corr = correlation_id_from(&headers);
    out.insert("X-Correlation-Id", corr.parse().unwrap());

    let id = format!("sha256:{}", sha256::digest(doc.to_string()));
    {
        let mut guard = state.artifacts.lock().unwrap();
        guard.insert(id.clone(), doc);
    }
    let body = json!({
        "id": id,
        "kind": "doc",
    });
    (StatusCode::CREATED, out, Json(body)).into_response()
}

pub async fn get_artifact(
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

    let guard = state.artifacts.lock().unwrap();
    if let Some(doc) = guard.get(&id) {
        return (StatusCode::OK, out, Json(doc.clone())).into_response();
    }
    (StatusCode::NOT_FOUND, out).into_response()
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
