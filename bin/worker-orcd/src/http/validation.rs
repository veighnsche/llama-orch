//! Request validation for worker-orcd HTTP endpoints
//!
//! This module provides validation logic for inference requests,
//! ensuring all parameters meet spec requirements before processing.
//!
//! # Spec References
//! - M0-W-1302: Request validation requirements

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

/// Execute request parameters
#[derive(Debug, Deserialize, Clone)]
pub struct ExecuteRequest {
    /// Job ID (must be non-empty)
    pub job_id: String,

    /// Prompt text (1-32768 characters)
    pub prompt: String,

    /// Maximum tokens to generate (1-2048)
    pub max_tokens: u32,

    /// Temperature for sampling (0.0-2.0)
    pub temperature: f32,

    /// Random seed for reproducibility
    pub seed: u64,
}

/// Validation error response
#[derive(Debug, Serialize)]
pub struct ValidationError {
    /// Field that failed validation
    pub field: String,

    /// Human-readable error message
    pub message: String,
}

impl IntoResponse for ValidationError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(self)).into_response()
    }
}

impl ExecuteRequest {
    /// Validate request parameters
    ///
    /// Returns `Ok(())` if all parameters are valid, or `Err(ValidationError)`
    /// with details about the first validation failure.
    ///
    /// # Validation Rules
    /// - `job_id`: Must be non-empty
    /// - `prompt`: Must be 1-32768 characters
    /// - `max_tokens`: Must be 1-2048
    /// - `temperature`: Must be 0.0-2.0 (inclusive)
    /// - `seed`: No validation (all u64 values valid)
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Validate job_id
        if self.job_id.is_empty() {
            return Err(ValidationError {
                field: "job_id".to_string(),
                message: "must not be empty".to_string(),
            });
        }

        // Validate prompt length
        let prompt_len = self.prompt.chars().count();
        if prompt_len == 0 {
            return Err(ValidationError {
                field: "prompt".to_string(),
                message: "must not be empty".to_string(),
            });
        }
        if prompt_len > 32768 {
            return Err(ValidationError {
                field: "prompt".to_string(),
                message: format!("must be at most 32768 characters (got {})", prompt_len),
            });
        }

        // Validate max_tokens
        if self.max_tokens < 1 {
            return Err(ValidationError {
                field: "max_tokens".to_string(),
                message: "must be at least 1".to_string(),
            });
        }
        if self.max_tokens > 2048 {
            return Err(ValidationError {
                field: "max_tokens".to_string(),
                message: format!("must be at most 2048 (got {})", self.max_tokens),
            });
        }

        // Validate temperature
        if self.temperature < 0.0 {
            return Err(ValidationError {
                field: "temperature".to_string(),
                message: format!("must be at least 0.0 (got {})", self.temperature),
            });
        }
        if self.temperature > 2.0 {
            return Err(ValidationError {
                field: "temperature".to_string(),
                message: format!("must be at most 2.0 (got {})", self.temperature),
            });
        }

        // Seed: no validation needed (all u64 values valid)

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_request() -> ExecuteRequest {
        ExecuteRequest {
            job_id: "test-job-123".to_string(),
            prompt: "Hello, world!".to_string(),
            max_tokens: 100,
            temperature: 0.7,
            seed: 42,
        }
    }

    #[test]
    fn test_valid_request() {
        let req = valid_request();
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_empty_job_id() {
        let mut req = valid_request();
        req.job_id = "".to_string();

        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "job_id");
        assert!(err.message.contains("empty"));
    }

    #[test]
    fn test_empty_prompt() {
        let mut req = valid_request();
        req.prompt = "".to_string();

        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "prompt");
        assert!(err.message.contains("empty"));
    }

    #[test]
    fn test_prompt_too_long() {
        let mut req = valid_request();
        req.prompt = "x".repeat(32769);

        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "prompt");
        assert!(err.message.contains("32768"));
    }

    #[test]
    fn test_prompt_exactly_32768() {
        let mut req = valid_request();
        req.prompt = "x".repeat(32768);

        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_max_tokens_too_small() {
        let mut req = valid_request();
        req.max_tokens = 0;

        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "max_tokens");
        assert!(err.message.contains("at least 1"));
    }

    #[test]
    fn test_max_tokens_too_large() {
        let mut req = valid_request();
        req.max_tokens = 2049;

        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "max_tokens");
        assert!(err.message.contains("2048"));
    }

    #[test]
    fn test_max_tokens_boundaries() {
        let mut req = valid_request();

        req.max_tokens = 1;
        assert!(req.validate().is_ok());

        req.max_tokens = 2048;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_temperature_too_low() {
        let mut req = valid_request();
        req.temperature = -0.1;

        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "temperature");
        assert!(err.message.contains("0.0"));
    }

    #[test]
    fn test_temperature_too_high() {
        let mut req = valid_request();
        req.temperature = 2.1;

        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "temperature");
        assert!(err.message.contains("2.0"));
    }

    #[test]
    fn test_temperature_boundaries() {
        let mut req = valid_request();

        req.temperature = 0.0;
        assert!(req.validate().is_ok());

        req.temperature = 2.0;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_seed_all_values_valid() {
        let mut req = valid_request();

        req.seed = 0;
        assert!(req.validate().is_ok());

        req.seed = u64::MAX;
        assert!(req.validate().is_ok());

        req.seed = 12345678901234567890;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_validation_error_serialization() {
        let err = ValidationError {
            field: "test_field".to_string(),
            message: "test message".to_string(),
        };

        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("test_field"));
        assert!(json.contains("test message"));
    }
}

// ---
// Built by Foundation-Alpha 🏗️
