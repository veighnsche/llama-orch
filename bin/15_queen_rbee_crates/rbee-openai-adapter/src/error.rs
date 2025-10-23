//! Error types for OpenAI adapter

use axum::{http::StatusCode, response::{IntoResponse, Response}, Json};
use crate::types::{ErrorResponse, ErrorDetail};

/// OpenAI adapter error
#[derive(Debug, thiserror::Error)]
pub enum OpenAIError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for OpenAIError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match self {
            OpenAIError::InvalidRequest(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_request_error", msg)
            }
            OpenAIError::ModelNotFound(msg) => {
                (StatusCode::NOT_FOUND, "model_not_found", msg)
            }
            OpenAIError::Internal(msg) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg)
            }
        };
        
        let error_response = ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: error_type.to_string(),
                code: None,
            },
        };
        
        (status, Json(error_response)).into_response()
    }
}
