//! Model management endpoints
//!
//! Per test-001-mvp.md Phase 3: Model Provisioning
//! - POST /v1/models/download - Download a model
//! - GET /v1/models/download/progress - SSE progress stream
//!
//! Created by: TEAM-026

use axum::{http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use tracing::info;

/// Download model request
#[derive(Debug, Deserialize)]
pub struct DownloadModelRequest {
    /// Model reference (e.g., "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    pub model_ref: String,
}

/// Download model response
#[derive(Debug, Serialize)]
pub struct DownloadModelResponse {
    /// Download ID (for progress tracking)
    pub download_id: String,
    /// Local path (if already downloaded)
    pub local_path: Option<String>,
}

/// Handle POST /v1/models/download
///
/// TODO: Implement model download with progress tracking
pub async fn handle_download_model(
    Json(request): Json<DownloadModelRequest>,
) -> Result<Json<DownloadModelResponse>, (StatusCode, String)> {
    info!(model_ref = %request.model_ref, "Model download requested");

    // TODO: Implement actual download logic
    // For now, return a placeholder response
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "Model download not yet implemented".to_string(),
    ))
}

/// Handle GET /v1/models/download/progress
///
/// TODO: Implement SSE progress stream
pub async fn handle_download_progress() -> Result<String, (StatusCode, String)> {
    // TODO: Implement SSE streaming
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "Download progress not yet implemented".to_string(),
    ))
}
