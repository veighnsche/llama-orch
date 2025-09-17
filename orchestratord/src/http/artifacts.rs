use axum::{extract::{Path, State}, response::{IntoResponse, Response}, Json};
use http::HeaderMap;
use serde_json::json;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::auth::require_api_key;
use crate::state::AppState;

fn correlation_id_from(headers: &HeaderMap) -> String {
    headers
        .get("X-Correlation-Id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    fn ok_headers() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("X-API-Key", "valid".parse().unwrap());
        h
    }

    #[tokio::test]
    async fn artifact_roundtrip_and_404() {
        let state = crate::state::default_state();
        // Create
        let body = json!({
            "kind": "plan",
            "content": {"steps":["a","b"]},
            "tags": ["test"],
            "metadata": {"author":"tester"}
        });
        let resp = create_artifact(ok_headers(), State(state.clone()), Json(body.clone())).await;
        assert_eq!(resp.status(), http::StatusCode::CREATED);
        let corr = resp.headers().get("X-Correlation-Id");
        assert!(corr.is_some(), "missing correlation id");
        let body_bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        let id = v["id"].as_str().expect("id").to_string();

        // Get
        let resp = get_artifact(ok_headers(), State(state.clone()), Path(id.clone())).await;
        assert_eq!(resp.status(), http::StatusCode::OK);
        let body_bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let doc: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(doc["kind"], json!("plan"));
        assert_eq!(doc["tags"], json!(["test"]));

        // 404
        let resp = get_artifact(ok_headers(), State(state.clone()), Path("nope".to_string())).await;
        assert_eq!(resp.status(), http::StatusCode::NOT_FOUND);
    }
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

pub async fn create_artifact(
    headers: HeaderMap,
    state: State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let req_corr = correlation_id_from(&headers);

    // Compute digest of the request body (content-addressed id)
    let body_bytes = serde_json::to_vec(&body).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(&body_bytes);
    let digest_raw = hasher.finalize();
    let digest_hex = to_hex(&digest_raw);
    let id = digest_hex.clone();
    let created_ms = chrono::Utc::now().timestamp_millis();

    // Extract fields from body per contract
    let kind = body.get("kind").cloned().unwrap_or(json!("other"));
    let tags = body.get("tags").cloned().unwrap_or(json!([]));
    let content = body.get("content").cloned().unwrap_or(json!({}));
    let parent = body.get("parent").cloned();
    let metadata = body.get("metadata").cloned().unwrap_or(json!({}));

    let stored = json!({
        "id": id,
        "kind": kind,
        "digest": format!("sha256:{}", digest_hex),
        "created_ms": created_ms,
        "tags": tags,
        "content": content,
        "parent": parent,
        "metadata": metadata,
    });

    {
        let mut map = state.artifacts.lock().unwrap();
        map.insert(stored["id"].as_str().unwrap().to_string(), stored.clone());
    }

    let mut h = HeaderMap::new();
    h.insert("X-Correlation-Id", req_corr.parse().unwrap());
    let resp = json!({
        "id": stored["id"],
        "kind": stored["kind"],
        "digest": stored["digest"],
        "created_ms": stored["created_ms"],
        "tags": stored["tags"],
    });
    (http::StatusCode::CREATED, h, Json(resp)).into_response()
}

pub async fn get_artifact(headers: HeaderMap, state: State<AppState>, Path(id): Path<String>) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let req_corr = correlation_id_from(&headers);
    let maybe = {
        let map = state.artifacts.lock().unwrap();
        map.get(&id).cloned()
    };
    let mut h = HeaderMap::new();
    h.insert("X-Correlation-Id", req_corr.parse().unwrap());
    match maybe {
        Some(doc) => (http::StatusCode::OK, h, Json(doc)).into_response(),
        None => {
            let err = json!({"code":"NOT_FOUND","message":"artifact not found"});
            (http::StatusCode::NOT_FOUND, h, Json(err)).into_response()
        }
    }
}
