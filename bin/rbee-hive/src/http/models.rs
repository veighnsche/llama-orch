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

// TEAM-031: Unit tests for models module
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_model_request_deserialization() {
        let json = r#"{"model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"}"#;
        let request: DownloadModelRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model_ref, "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF");
    }

    #[test]
    fn test_download_model_response_serialization() {
        let response = DownloadModelResponse {
            download_id: "download-123".to_string(),
            local_path: Some("/models/tinyllama.gguf".to_string()),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["download_id"], "download-123");
        assert_eq!(json["local_path"], "/models/tinyllama.gguf");
    }

    #[test]
    fn test_download_model_response_serialization_no_path() {
        let response = DownloadModelResponse {
            download_id: "download-123".to_string(),
            local_path: None,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["download_id"], "download-123");
        assert!(json["local_path"].is_null());
    }
}
