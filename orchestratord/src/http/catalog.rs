use super::auth::require_api_key;
use crate::state::AppState;
use axum::{
    extract::{Path, State},
    response::{IntoResponse, Response},
    Json,
};
use http::HeaderMap;
use serde::Deserialize;
use serde_json::json;

fn correlation_id_from(headers: &HeaderMap) -> String {
    headers
        .get("X-Correlation-Id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "corr-0".to_string())
}

#[derive(Deserialize)]
pub struct CatalogModelReq {
    pub id: String,
    #[serde(default)]
    pub signed: Option<bool>,
}

pub async fn create_catalog_model(
    headers: HeaderMap,
    _state: State<AppState>,
    Json(body): Json<CatalogModelReq>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let strict = headers.get("X-Trust-Policy").and_then(|v| v.to_str().ok()) == Some("strict");
    let signed = body.signed.unwrap_or(true);
    if strict && !signed {
        let mut h = HeaderMap::new();
        let req_corr = correlation_id_from(&headers);
        h.insert("X-Correlation-Id", req_corr.parse().unwrap());
        let err = json!({ "code": "UNTRUSTED_ARTIFACT" });
        return (http::StatusCode::BAD_REQUEST, h, Json(err)).into_response();
    }
    let resp = json!({ "id": body.id, "signatures": true, "sbom": true });
    let mut h = HeaderMap::new();
    let req_corr = correlation_id_from(&headers);
    h.insert("X-Correlation-Id", req_corr.parse().unwrap());
    (http::StatusCode::CREATED, h, Json(resp)).into_response()
}

pub async fn get_catalog_model(
    headers: HeaderMap,
    _state: State<AppState>,
    Path(id): Path<String>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let resp = json!({ "id": id, "signatures": true, "sbom": true });
    let mut h = HeaderMap::new();
    let req_corr = correlation_id_from(&headers);
    h.insert("X-Correlation-Id", req_corr.parse().unwrap());
    (http::StatusCode::OK, h, Json(resp)).into_response()
}

pub async fn verify_catalog_model(
    headers: HeaderMap,
    _state: State<AppState>,
    Path(_id): Path<String>,
) -> Response {
    if let Err(code) = require_api_key(&headers) {
        return (code, HeaderMap::new()).into_response();
    }
    let resp = json!({ "status": "started" });
    let mut h = HeaderMap::new();
    let req_corr = correlation_id_from(&headers);
    h.insert("X-Correlation-Id", req_corr.parse().unwrap());
    (http::StatusCode::ACCEPTED, h, Json(resp)).into_response()
}
