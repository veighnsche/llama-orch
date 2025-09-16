use axum::{extract::{Path, State}, response::{IntoResponse, Response}, Json};
use http::HeaderMap;
use serde::Deserialize;
use serde_json::json;
use super::auth::require_api_key;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct CatalogModelReq { pub id: String, #[serde(default)] pub signed: Option<bool> }

pub async fn create_catalog_model(headers: HeaderMap, _state: State<AppState>, Json(body): Json<CatalogModelReq>) -> Response {
    if let Err(code) = require_api_key(&headers) { return (code, HeaderMap::new()).into_response(); }
    let strict = headers.get("X-Trust-Policy").and_then(|v| v.to_str().ok()) == Some("strict");
    let signed = body.signed.unwrap_or(true);
    if strict && !signed {
        let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
        let err = json!({ "code": "UNTRUSTED_ARTIFACT" });
        return (http::StatusCode::BAD_REQUEST, h, Json(err)).into_response();
    }
    let resp = json!({ "id": body.id, "signatures": true, "sbom": true });
    let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    (http::StatusCode::CREATED, h, Json(resp)).into_response()
}

pub async fn get_catalog_model(headers: HeaderMap, _state: State<AppState>, Path(id): Path<String>) -> Response {
    if let Err(code) = require_api_key(&headers) { return (code, HeaderMap::new()).into_response(); }
    let resp = json!({ "id": id, "signatures": true, "sbom": true });
    let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    (http::StatusCode::OK, h, Json(resp)).into_response()
}

pub async fn verify_catalog_model(headers: HeaderMap, _state: State<AppState>, Path(_id): Path<String>) -> Response {
    if let Err(code) = require_api_key(&headers) { return (code, HeaderMap::new()).into_response(); }
    let resp = json!({ "status": "started" });
    let mut h = HeaderMap::new(); h.insert("X-Correlation-Id", "corr-0".parse().unwrap());
    (http::StatusCode::ACCEPTED, h, Json(resp)).into_response()
}
