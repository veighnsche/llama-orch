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

/// Single field validation error
#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct FieldError {
    /// Field that failed validation
    pub field: String,

    /// Constraint that was violated
    pub constraint: String,

    /// Human-readable error message
    pub message: String,

    /// Provided value (optional, omitted for sensitive fields)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

/// Validation error response (single error for backward compatibility)
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

/// Multiple validation errors response
#[derive(Debug, Serialize)]
pub struct ValidationErrorResponse {
    /// List of validation errors
    pub errors: Vec<FieldError>,
}

impl IntoResponse for ValidationErrorResponse {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(self)).into_response()
    }
}

impl ExecuteRequest {
    /// Validate request parameters (returns first error only)
    ///
    /// Returns `Ok(())` if all parameters are valid, or `Err(ValidationError)`
    /// with details about the first validation failure.
    ///
    /// For collecting all errors, use `validate_all()`.
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

    /// Validate request parameters and collect ALL errors
    ///
    /// Returns `Ok(())` if all parameters are valid, or `Err(ValidationErrorResponse)`
    /// with details about ALL validation failures (not just the first).
    ///
    /// # Validation Rules
    /// Same as `validate()`, but collects all errors before returning.
    pub fn validate_all(&self) -> Result<(), ValidationErrorResponse> {
        let mut errors = Vec::new();

        // Validate job_id
        if self.job_id.is_empty() {
            errors.push(FieldError {
                field: "job_id".to_string(),
                constraint: "non_empty".to_string(),
                message: "job_id must not be empty".to_string(),
                value: Some(self.job_id.clone()),
            });
        }

        // Validate prompt length
        let prompt_len = self.prompt.chars().count();
        if prompt_len == 0 {
            errors.push(FieldError {
                field: "prompt".to_string(),
                constraint: "length(1..=32768)".to_string(),
                message: "prompt must not be empty".to_string(),
                value: None, // Don't include prompt in error
            });
        } else if prompt_len > 32768 {
            errors.push(FieldError {
                field: "prompt".to_string(),
                constraint: "length(1..=32768)".to_string(),
                message: format!("prompt must be at most 32768 characters (got {})", prompt_len),
                value: None, // Don't include prompt in error
            });
        }

        // Validate max_tokens
        if self.max_tokens < 1 {
            errors.push(FieldError {
                field: "max_tokens".to_string(),
                constraint: "range(1..=2048)".to_string(),
                message: "max_tokens must be at least 1".to_string(),
                value: Some(self.max_tokens.to_string()),
            });
        } else if self.max_tokens > 2048 {
            errors.push(FieldError {
                field: "max_tokens".to_string(),
                constraint: "range(1..=2048)".to_string(),
                message: format!("max_tokens must be at most 2048 (got {})", self.max_tokens),
                value: Some(self.max_tokens.to_string()),
            });
        }

        // Validate temperature
        if self.temperature < 0.0 {
            errors.push(FieldError {
                field: "temperature".to_string(),
                constraint: "range(0.0..=2.0)".to_string(),
                message: format!("temperature must be at least 0.0 (got {})", self.temperature),
                value: Some(self.temperature.to_string()),
            });
        } else if self.temperature > 2.0 {
            errors.push(FieldError {
                field: "temperature".to_string(),
                constraint: "range(0.0..=2.0)".to_string(),
                message: format!("temperature must be at most 2.0 (got {})", self.temperature),
                value: Some(self.temperature.to_string()),
            });
        }

        // Seed: no validation needed (all u64 values valid)

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ValidationErrorResponse { errors })
        }
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

    #[test]
    fn test_validate_all_collects_multiple_errors() {
        let req = ExecuteRequest {
            job_id: "".to_string(), // Invalid: empty
            prompt: "".to_string(), // Invalid: empty
            max_tokens: 0,          // Invalid: too small
            temperature: 3.0,       // Invalid: too high
            seed: 42,
        };

        let result = req.validate_all();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.errors.len(), 4); // All 4 errors collected

        // Verify each error is present
        let fields: Vec<_> = err.errors.iter().map(|e| e.field.as_str()).collect();
        assert!(fields.contains(&"job_id"));
        assert!(fields.contains(&"prompt"));
        assert!(fields.contains(&"max_tokens"));
        assert!(fields.contains(&"temperature"));
    }

    #[test]
    fn test_validate_all_with_valid_request() {
        let req = valid_request();
        assert!(req.validate_all().is_ok());
    }

    #[test]
    fn test_field_error_structure() {
        let err = FieldError {
            field: "test_field".to_string(),
            constraint: "non_empty".to_string(),
            message: "must not be empty".to_string(),
            value: Some("".to_string()),
        };

        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("test_field"));
        assert!(json.contains("non_empty"));
        assert!(json.contains("must not be empty"));
    }

    #[test]
    fn test_field_error_omits_none_value() {
        let err = FieldError {
            field: "prompt".to_string(),
            constraint: "length".to_string(),
            message: "too long".to_string(),
            value: None, // Should be omitted from JSON
        };

        let json = serde_json::to_string(&err).unwrap();
        assert!(!json.contains("\"value\""));
    }

    #[test]
    fn test_validation_error_response_serialization() {
        let response = ValidationErrorResponse {
            errors: vec![
                FieldError {
                    field: "field1".to_string(),
                    constraint: "constraint1".to_string(),
                    message: "message1".to_string(),
                    value: Some("value1".to_string()),
                },
                FieldError {
                    field: "field2".to_string(),
                    constraint: "constraint2".to_string(),
                    message: "message2".to_string(),
                    value: None,
                },
            ],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"errors\""));
        assert!(json.contains("field1"));
        assert!(json.contains("field2"));
    }
}

// ---
// Built by Foundation-Alpha 🏗️
