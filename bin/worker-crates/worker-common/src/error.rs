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
        }
    }

    /// Check if error is retriable by orchestrator
    pub fn is_retriable(&self) -> bool {
        match self {
            Self::Cuda(_) => true,
            Self::Timeout | Self::Internal(_) => true,
            _ => false,
        }
    }

    /// Get HTTP status code
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::Cuda(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            Self::Timeout => StatusCode::REQUEST_TIMEOUT,
            Self::Unhealthy(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
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
