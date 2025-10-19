// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Error types with HTTP responses

//! Worker error types

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

/// Worker-level errors
#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Inference timeout")]
    Timeout,

    #[error("Worker unhealthy: {0}")]
    Unhealthy(String),

    #[error("Internal error: {0}")]
    Internal(String),

    /// TEAM-130: Insufficient resources (RAM, CPU, etc.)
    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),

    /// TEAM-130: Insufficient VRAM for model
    #[error("Insufficient VRAM: {0}")]
    InsufficientVram(String),
}

impl WorkerError {
    /// Get stable error code for API responses
    pub fn code(&self) -> &'static str {
        match self {
            Self::Cuda(_) => "CUDA_ERROR",
            Self::InvalidRequest(_) => "INVALID_REQUEST",
            Self::Timeout => "INFERENCE_TIMEOUT",
            Self::Unhealthy(_) => "WORKER_UNHEALTHY",
            Self::Internal(_) => "INTERNAL",
            Self::InsufficientResources(_) => "INSUFFICIENT_RESOURCES",
            Self::InsufficientVram(_) => "INSUFFICIENT_VRAM",
        }
    }

    /// Check if error is retriable by orchestrator
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            Self::Cuda(_) | Self::Timeout | Self::Internal(_) | Self::InsufficientResources(_) | Self::InsufficientVram(_)
        )
    }

    /// Get HTTP status code
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::Cuda(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            Self::Timeout => StatusCode::REQUEST_TIMEOUT,
            Self::Unhealthy(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::InsufficientResources(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::InsufficientVram(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

#[derive(Serialize)]
struct ErrorResponse {
    code: &'static str,
    message: String,
    retriable: bool,
}

impl IntoResponse for WorkerError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = ErrorResponse {
            code: self.code(),
            message: self.to_string(),
            retriable: self.is_retriable(),
        };

        (status, Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_error_properties() {
        let err = WorkerError::Cuda("out of memory".to_string());
        assert_eq!(err.code(), "CUDA_ERROR");
        assert!(err.is_retriable());
        assert_eq!(err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(err.to_string().contains("CUDA error"));
        assert!(err.to_string().contains("out of memory"));
    }

    #[test]
    fn test_invalid_request_error_properties() {
        let err = WorkerError::InvalidRequest("missing prompt".to_string());
        assert_eq!(err.code(), "INVALID_REQUEST");
        assert!(!err.is_retriable());
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
        assert!(err.to_string().contains("Invalid request"));
        assert!(err.to_string().contains("missing prompt"));
    }

    #[test]
    fn test_timeout_error_properties() {
        let err = WorkerError::Timeout;
        assert_eq!(err.code(), "INFERENCE_TIMEOUT");
        assert!(err.is_retriable());
        assert_eq!(err.status_code(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(err.to_string(), "Inference timeout");
    }

    #[test]
    fn test_unhealthy_error_properties() {
        let err = WorkerError::Unhealthy("model not loaded".to_string());
        assert_eq!(err.code(), "WORKER_UNHEALTHY");
        assert!(!err.is_retriable());
        assert_eq!(err.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(err.to_string().contains("Worker unhealthy"));
        assert!(err.to_string().contains("model not loaded"));
    }

    #[test]
    fn test_internal_error_properties() {
        let err = WorkerError::Internal("panic in decoder".to_string());
        assert_eq!(err.code(), "INTERNAL");
        assert!(err.is_retriable());
        assert_eq!(err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(err.to_string().contains("Internal error"));
        assert!(err.to_string().contains("panic in decoder"));
    }

    #[test]
    fn test_retriability_classification() {
        // Retriable errors
        assert!(WorkerError::Cuda("test".to_string()).is_retriable());
        assert!(WorkerError::Timeout.is_retriable());
        assert!(WorkerError::Internal("test".to_string()).is_retriable());

        // Non-retriable errors
        assert!(!WorkerError::InvalidRequest("test".to_string()).is_retriable());
        assert!(!WorkerError::Unhealthy("test".to_string()).is_retriable());
    }

    #[test]
    fn test_status_code_mapping() {
        assert_eq!(
            WorkerError::Cuda("test".to_string()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            WorkerError::InvalidRequest("test".to_string()).status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(WorkerError::Timeout.status_code(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(
            WorkerError::Unhealthy("test".to_string()).status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            WorkerError::Internal("test".to_string()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn test_error_code_stability() {
        // Error codes must be stable for API compatibility
        assert_eq!(WorkerError::Cuda("".to_string()).code(), "CUDA_ERROR");
        assert_eq!(WorkerError::InvalidRequest("".to_string()).code(), "INVALID_REQUEST");
        assert_eq!(WorkerError::Timeout.code(), "INFERENCE_TIMEOUT");
        assert_eq!(WorkerError::Unhealthy("".to_string()).code(), "WORKER_UNHEALTHY");
        assert_eq!(WorkerError::Internal("".to_string()).code(), "INTERNAL");
        // TEAM-130: New resource error codes
        assert_eq!(WorkerError::InsufficientResources("".to_string()).code(), "INSUFFICIENT_RESOURCES");
        assert_eq!(WorkerError::InsufficientVram("".to_string()).code(), "INSUFFICIENT_VRAM");
    }

    #[tokio::test]
    async fn test_into_response_structure() {
        let err = WorkerError::Timeout;
        let response = err.into_response();

        // Verify status code
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);

        // Extract and verify body
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("Failed to read response body");
        let body_str = std::str::from_utf8(&body_bytes).expect("Invalid UTF-8");
        let json: serde_json::Value = serde_json::from_str(body_str).expect("Invalid JSON");

        assert_eq!(json["code"], "INFERENCE_TIMEOUT");
        assert_eq!(json["message"], "Inference timeout");
        assert_eq!(json["retriable"], true);
    }

    #[tokio::test]
    async fn test_into_response_invalid_request() {
        let err = WorkerError::InvalidRequest("prompt too long".to_string());
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("Failed to read response body");
        let body_str = std::str::from_utf8(&body_bytes).expect("Invalid UTF-8");
        let json: serde_json::Value = serde_json::from_str(body_str).expect("Invalid JSON");

        assert_eq!(json["code"], "INVALID_REQUEST");
        assert!(json["message"].as_str().unwrap().contains("prompt too long"));
        assert_eq!(json["retriable"], false);
    }

    #[tokio::test]
    async fn test_into_response_cuda_error() {
        let err = WorkerError::Cuda("device 0 not found".to_string());
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("Failed to read response body");
        let body_str = std::str::from_utf8(&body_bytes).expect("Invalid UTF-8");
        let json: serde_json::Value = serde_json::from_str(body_str).expect("Invalid JSON");

        assert_eq!(json["code"], "CUDA_ERROR");
        assert!(json["message"].as_str().unwrap().contains("device 0 not found"));
        assert_eq!(json["retriable"], true);
    }

    #[test]
    fn test_error_message_formatting() {
        // Verify error messages are properly formatted
        let cuda_err = WorkerError::Cuda("OOM".to_string());
        assert_eq!(cuda_err.to_string(), "CUDA error: OOM");

        let invalid_err = WorkerError::InvalidRequest("bad param".to_string());
        assert_eq!(invalid_err.to_string(), "Invalid request: bad param");

        let timeout_err = WorkerError::Timeout;
        assert_eq!(timeout_err.to_string(), "Inference timeout");

        let unhealthy_err = WorkerError::Unhealthy("not ready".to_string());
        assert_eq!(unhealthy_err.to_string(), "Worker unhealthy: not ready");

        let internal_err = WorkerError::Internal("crash".to_string());
        assert_eq!(internal_err.to_string(), "Internal error: crash");

        // TEAM-130: New resource error messages
        let resources_err = WorkerError::InsufficientResources("not enough RAM".to_string());
        assert_eq!(resources_err.to_string(), "Insufficient resources: not enough RAM");

        let vram_err = WorkerError::InsufficientVram("need 8GB, have 2GB".to_string());
        assert_eq!(vram_err.to_string(), "Insufficient VRAM: need 8GB, have 2GB");
    }

    // TEAM-130: Tests for new resource error types
    #[test]
    fn test_insufficient_resources_error() {
        let err = WorkerError::InsufficientResources("not enough RAM".to_string());
        assert_eq!(err.code(), "INSUFFICIENT_RESOURCES");
        assert!(err.is_retriable());
        assert_eq!(err.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(err.to_string().contains("Insufficient resources"));
        assert!(err.to_string().contains("not enough RAM"));
    }

    #[test]
    fn test_insufficient_vram_error() {
        let err = WorkerError::InsufficientVram("need 8GB, have 2GB".to_string());
        assert_eq!(err.code(), "INSUFFICIENT_VRAM");
        assert!(err.is_retriable());
        assert_eq!(err.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(err.to_string().contains("Insufficient VRAM"));
        assert!(err.to_string().contains("need 8GB, have 2GB"));
    }

    #[tokio::test]
    async fn test_into_response_insufficient_vram() {
        let err = WorkerError::InsufficientVram("required: 8GB, available: 2GB".to_string());
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("Failed to read response body");
        let body_str = std::str::from_utf8(&body_bytes).expect("Invalid UTF-8");
        let json: serde_json::Value = serde_json::from_str(body_str).expect("Invalid JSON");

        assert_eq!(json["code"], "INSUFFICIENT_VRAM");
        assert!(json["message"].as_str().unwrap().contains("required: 8GB"));
        assert_eq!(json["retriable"], true);
    }

    #[tokio::test]
    async fn test_into_response_insufficient_resources() {
        let err = WorkerError::InsufficientResources("RAM exhausted".to_string());
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("Failed to read response body");
        let body_str = std::str::from_utf8(&body_bytes).expect("Invalid UTF-8");
        let json: serde_json::Value = serde_json::from_str(body_str).expect("Invalid JSON");

        assert_eq!(json["code"], "INSUFFICIENT_RESOURCES");
        assert!(json["message"].as_str().unwrap().contains("RAM exhausted"));
        assert_eq!(json["retriable"], true);
    }
}

// ---
// Verified by Testing Team 🔍
