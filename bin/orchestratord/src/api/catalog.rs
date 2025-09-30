use axum::{
    extract::{Path, State},
    Json,
};
use axum::extract::BearerToken;
use axum::middleware::Next;
use axum::response::IntoResponse;
use http::StatusCode;
use serde_json::json;

use crate::domain::error::OrchestratorError as ErrO;
use crate::state::AppState;

// TODO(SECURITY): Add authentication to catalog endpoints using auth-min
//
// Catalog endpoints allow model registration and state changes.
// These should require authentication to prevent unauthorized catalog modifications.
//
// Endpoints requiring auth:
// - POST /v2/catalog/models - Register model
// - GET /v2/catalog/models/{id} - Get model
// - POST /v2/catalog/models/{id}/verify - Verify model
// - POST /v2/catalog/models/{id}/state - Set state (Active/Retired)
//
// Should use auth-min based middleware for Bearer token validation.
//
// See: .docs/PHASE5_FIX_CHECKLIST.md Task 11
// See: .specs/12_auth-min-hardening.md (SEC-AUTH-3001)
//! Catalog API endpoints for model management, FsCatalog, LifecycleState};
fn open_catalog() -> anyhow::Result<FsCatalog> {
    let root = catalog_core::default_model_cache_dir();
    Ok(FsCatalog::new(root)?)
}

pub async fn create_model(
    _state: State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<(StatusCode, Json<serde_json::Value>), ErrO> {
    let id = body.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if id.is_empty() {
        return Ok((StatusCode::BAD_REQUEST, Json(json!({"error":"id required"}))));
    }
    let digest = body.get("digest").and_then(|v| v.as_str()).map(|s| s.to_string());
    let digest_parsed = digest
        .as_ref()
        .and_then(|s| s.split_once(":"))
        .map(|(algo, val)| Digest { algo: algo.to_string(), value: val.to_string() });

    let cat = open_catalog().map_err(|_e| ErrO::Internal)?;
    // Use a placeholder local path keyed by id; actual path will be set when provisioners ensure models.
    let entry = catalog_core::CatalogEntry {
        id: id.clone(),
        local_path: catalog_core::default_model_cache_dir().join(id.clone()),
        lifecycle: LifecycleState::Active,
        digest: digest_parsed,
        last_verified_ms: None,
    };
    cat.put(&entry).map_err(|_e| ErrO::Internal)?;

    let out = json!({
        "id": entry.id,
        "digest": entry.digest.as_ref().map(|d| format!("{}:{}", d.algo, d.value)),
        "source_url": body.get("source_url").cloned().unwrap_or(json!(null)),
        "manifests": body.get("manifests").cloned().unwrap_or(json!([])),
        "signatures": body.get("signatures").cloned().unwrap_or(json!([])),
        "sbom": body.get("sbom").cloned().unwrap_or(json!(null)),
        "trust_policy": body.get("trust_policy").cloned().unwrap_or(json!(null)),
    });
    Ok((StatusCode::CREATED, Json(out)))
}

pub async fn get_model(
    _state: State<AppState>,
    Path(id): Path<String>,
) -> Result<(StatusCode, Json<serde_json::Value>), ErrO> {
    let cat = open_catalog().map_err(|_e| ErrO::Internal)?;
    if let Some(entry) = cat.get(&id).map_err(|_e| ErrO::Internal)? {
        // Return full CatalogEntry as JSON (includes id, digest, state, timestamps, etc.)
        let out = serde_json::to_value(&entry).map_err(|_| ErrO::Internal)?;
        Ok((StatusCode::OK, Json(out)))
    } else {
        Ok((StatusCode::NOT_FOUND, Json(json!({"error":"not found"}))))
    }
}

pub async fn verify_model(
    _state: State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ErrO> {
    let cat = open_catalog().map_err(|_e| ErrO::Internal)?;
    if let Some(mut entry) = cat.get(&id).map_err(|_e| ErrO::Internal)? {
        entry.last_verified_ms = Some(
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()
                as u64,
        );
        cat.put(&entry).map_err(|_e| ErrO::Internal)?;
        Ok(StatusCode::ACCEPTED)
    } else {
        Ok(StatusCode::NOT_FOUND)
    }
}

#[derive(serde::Deserialize)]
pub struct SetStateRequest {
    pub state: String,
    pub deadline_ms: Option<i64>,
}

pub async fn set_model_state(
    _state: State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<SetStateRequest>,
) -> Result<StatusCode, ErrO> {
    let state = match req.state.as_str() {
        "Active" => LifecycleState::Active,
        "Retired" => LifecycleState::Retired,
        _ => LifecycleState::Active,
    };
    let cat = open_catalog().map_err(|_e| ErrO::Internal)?;
    cat.set_state(&id, state).map_err(|_e| ErrO::Internal)?;
    Ok(StatusCode::ACCEPTED)
}

pub async fn delete_model(
    _state: State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ErrO> {
    let cat = open_catalog().map_err(|_e| ErrO::Internal)?;
    let deleted = cat.delete(&id).map_err(|_e| ErrO::Internal)?;
    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Ok(StatusCode::NOT_FOUND)
    }
}
