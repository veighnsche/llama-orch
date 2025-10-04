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
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Random seed for reproducibility
    #[serde(default)]
    pub seed: Option<u64>,

    /// Top-P (nucleus) sampling (0.0-1.0, 1.0 = disabled)
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-K sampling (0 = disabled)
    #[serde(default)]
    pub top_k: u32,

    /// Repetition penalty (1.0 = disabled, >1.0 = penalize)
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// Stop sequences (max 4)
    #[serde(default)]
    pub stop: Vec<String>,

    /// Min-P sampling (0.0 = disabled)
    #[serde(default)]
    pub min_p: f32,
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_repetition_penalty() -> f32 {
    1.0
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
    /// - `seed`: Optional, no validation (all u64 values valid)
    /// - `top_p`: Must be 0.0-1.0 (inclusive)
    /// - `top_k`: No validation (all u32 values valid)
    /// - `repetition_penalty`: Must be 0.0-2.0 (inclusive)
    /// - `stop`: Max 4 sequences, each max 100 chars
    /// - `min_p`: Must be 0.0-1.0 (inclusive)
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

        // Validate top_p
        if self.top_p < 0.0 {
            return Err(ValidationError {
                field: "top_p".to_string(),
                message: format!("must be at least 0.0 (got {})", self.top_p),
            });
        }
        if self.top_p > 1.0 {
            return Err(ValidationError {
                field: "top_p".to_string(),
                message: format!("must be at most 1.0 (got {})", self.top_p),
            });
        }

        // top_k: no validation needed (all u32 values valid)

        // Validate repetition_penalty
        if self.repetition_penalty < 0.0 {
            return Err(ValidationError {
                field: "repetition_penalty".to_string(),
                message: format!("must be at least 0.0 (got {})", self.repetition_penalty),
            });
        }
        if self.repetition_penalty > 2.0 {
            return Err(ValidationError {
                field: "repetition_penalty".to_string(),
                message: format!("must be at most 2.0 (got {})", self.repetition_penalty),
            });
        }

        // Validate stop sequences
        if self.stop.len() > 4 {
            return Err(ValidationError {
                field: "stop".to_string(),
                message: format!("must have at most 4 sequences (got {})", self.stop.len()),
            });
        }
        for (i, seq) in self.stop.iter().enumerate() {
            if seq.is_empty() {
                return Err(ValidationError {
                    field: "stop".to_string(),
                    message: format!("sequence {} must not be empty", i),
                });
            }
            if seq.len() > 100 {
                return Err(ValidationError {
                    field: "stop".to_string(),
                    message: format!("sequence {} must be at most 100 characters (got {})", i, seq.len()),
                });
            }
        }

        // Validate min_p
        if self.min_p < 0.0 {
            return Err(ValidationError {
                field: "min_p".to_string(),
                message: format!("must be at least 0.0 (got {})", self.min_p),
            });
        }
        if self.min_p > 1.0 {
            return Err(ValidationError {
                field: "min_p".to_string(),
                message: format!("must be at most 1.0 (got {})", self.min_p),
            });
        }

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

        // Validate top_p
        if self.top_p < 0.0 {
            errors.push(FieldError {
                field: "top_p".to_string(),
                constraint: "range(0.0..=1.0)".to_string(),
                message: format!("top_p must be at least 0.0 (got {})", self.top_p),
                value: Some(self.top_p.to_string()),
            });
        } else if self.top_p > 1.0 {
            errors.push(FieldError {
                field: "top_p".to_string(),
                constraint: "range(0.0..=1.0)".to_string(),
                message: format!("top_p must be at most 1.0 (got {})", self.top_p),
                value: Some(self.top_p.to_string()),
            });
        }

        // top_k: no validation needed (all u32 values valid)

        // Validate repetition_penalty
        if self.repetition_penalty < 0.0 {
            errors.push(FieldError {
                field: "repetition_penalty".to_string(),
                constraint: "range(0.0..=2.0)".to_string(),
                message: format!("repetition_penalty must be at least 0.0 (got {})", self.repetition_penalty),
                value: Some(self.repetition_penalty.to_string()),
            });
        } else if self.repetition_penalty > 2.0 {
            errors.push(FieldError {
                field: "repetition_penalty".to_string(),
                constraint: "range(0.0..=2.0)".to_string(),
                message: format!("repetition_penalty must be at most 2.0 (got {})", self.repetition_penalty),
                value: Some(self.repetition_penalty.to_string()),
            });
        }

        // Validate stop sequences
        if self.stop.len() > 4 {
            errors.push(FieldError {
                field: "stop".to_string(),
                constraint: "max_count(4)".to_string(),
                message: format!("stop must have at most 4 sequences (got {})", self.stop.len()),
                value: Some(self.stop.len().to_string()),
            });
        }
        for (i, seq) in self.stop.iter().enumerate() {
            if seq.is_empty() {
                errors.push(FieldError {
                    field: "stop".to_string(),
                    constraint: "non_empty".to_string(),
                    message: format!("stop sequence {} must not be empty", i),
                    value: None,
                });
            } else if seq.len() > 100 {
                errors.push(FieldError {
                    field: "stop".to_string(),
                    constraint: "max_length(100)".to_string(),
                    message: format!("stop sequence {} must be at most 100 characters (got {})", i, seq.len()),
                    value: None,
                });
            }
        }

        // Validate min_p
        if self.min_p < 0.0 {
            errors.push(FieldError {
                field: "min_p".to_string(),
                constraint: "range(0.0..=1.0)".to_string(),
                message: format!("min_p must be at least 0.0 (got {})", self.min_p),
                value: Some(self.min_p.to_string()),
            });
        } else if self.min_p > 1.0 {
            errors.push(FieldError {
                field: "min_p".to_string(),
                constraint: "range(0.0..=1.0)".to_string(),
                message: format!("min_p must be at most 1.0 (got {})", self.min_p),
                value: Some(self.min_p.to_string()),
            });
        }

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
            seed: Some(42),
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            stop: vec![],
            min_p: 0.0,
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

        req.seed = Some(0);
        assert!(req.validate().is_ok());

        req.seed = Some(u64::MAX);
        assert!(req.validate().is_ok());

        req.seed = Some(12345678901234567890);
        assert!(req.validate().is_ok());
        
        req.seed = None;
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
            seed: Some(42),
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            stop: vec![],
            min_p: 0.0,
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

    // ========================================================================
    // Advanced Parameter Validation Tests
    // ========================================================================

    #[test]
    fn test_top_p_valid_range() {
        let mut req = valid_request();
        
        req.top_p = 0.0;
        assert!(req.validate().is_ok());
        
        req.top_p = 0.5;
        assert!(req.validate().is_ok());
        
        req.top_p = 1.0;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_top_p_too_low() {
        let mut req = valid_request();
        req.top_p = -0.1;
        
        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "top_p");
        assert!(err.message.contains("0.0"));
    }

    #[test]
    fn test_top_p_too_high() {
        let mut req = valid_request();
        req.top_p = 1.1;
        
        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "top_p");
        assert!(err.message.contains("1.0"));
    }

    #[test]
    fn test_top_k_all_values_valid() {
        let mut req = valid_request();
        
        req.top_k = 0;  // Disabled
        assert!(req.validate().is_ok());
        
        req.top_k = 50;
        assert!(req.validate().is_ok());
        
        req.top_k = 151936;  // Large vocab
        assert!(req.validate().is_ok());
        
        req.top_k = u32::MAX;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_repetition_penalty_valid_range() {
        let mut req = valid_request();
        
        req.repetition_penalty = 0.0;
        assert!(req.validate().is_ok());
        
        req.repetition_penalty = 1.0;  // Disabled
        assert!(req.validate().is_ok());
        
        req.repetition_penalty = 1.5;
        assert!(req.validate().is_ok());
        
        req.repetition_penalty = 2.0;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_repetition_penalty_too_low() {
        let mut req = valid_request();
        req.repetition_penalty = -0.1;
        
        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "repetition_penalty");
        assert!(err.message.contains("0.0"));
    }

    #[test]
    fn test_repetition_penalty_too_high() {
        let mut req = valid_request();
        req.repetition_penalty = 2.1;
        
        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "repetition_penalty");
        assert!(err.message.contains("2.0"));
    }

    #[test]
    fn test_stop_sequences_valid() {
        let mut req = valid_request();
        
        req.stop = vec![];  // Empty is valid
        assert!(req.validate().is_ok());
        
        req.stop = vec!["\\n\\n".to_string()];
        assert!(req.validate().is_ok());
        
        req.stop = vec!["\\n\\n".to_string(), "END".to_string(), "STOP".to_string()];
        assert!(req.validate().is_ok());
        
        req.stop = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_stop_sequences_too_many() {
        let mut req = valid_request();
        req.stop = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string(), "e".to_string()];
        
        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "stop");
        assert!(err.message.contains("4"));
    }

    #[test]
    fn test_stop_sequence_empty() {
        let mut req = valid_request();
        req.stop = vec!["".to_string()];
        
        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "stop");
        assert!(err.message.contains("empty"));
    }

    #[test]
    fn test_stop_sequence_too_long() {
        let mut req = valid_request();
        req.stop = vec!["x".repeat(101)];
        
        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "stop");
        assert!(err.message.contains("100"));
    }

    #[test]
    fn test_min_p_valid_range() {
        let mut req = valid_request();
        
        req.min_p = 0.0;  // Disabled
        assert!(req.validate().is_ok());
        
        req.min_p = 0.05;
        assert!(req.validate().is_ok());
        
        req.min_p = 1.0;
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_min_p_too_low() {
        let mut req = valid_request();
        req.min_p = -0.1;
        
        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "min_p");
        assert!(err.message.contains("0.0"));
    }

    #[test]
    fn test_min_p_too_high() {
        let mut req = valid_request();
        req.min_p = 1.1;
        
        let err = req.validate().unwrap_err();
        assert_eq!(err.field, "min_p");
        assert!(err.message.contains("1.0"));
    }

    #[test]
    fn test_backward_compatibility_old_request_format() {
        // Old request format (Sprint 3) should still work
        let json = r#"{
            "job_id": "test-job",
            "prompt": "Hello",
            "max_tokens": 100,
            "temperature": 0.7,
            "seed": 42
        }"#;
        
        let req: ExecuteRequest = serde_json::from_str(json).unwrap();
        assert!(req.validate().is_ok());
        
        // Verify defaults
        assert_eq!(req.top_p, 1.0);
        assert_eq!(req.top_k, 0);
        assert_eq!(req.repetition_penalty, 1.0);
        assert_eq!(req.stop.len(), 0);
        assert_eq!(req.min_p, 0.0);
    }

    #[test]
    fn test_new_request_format_with_all_parameters() {
        let json = r#"{
            "job_id": "test-job",
            "prompt": "Hello",
            "max_tokens": 100,
            "temperature": 0.7,
            "seed": 42,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop": ["\\n\\n", "END"],
            "min_p": 0.05
        }"#;
        
        let req: ExecuteRequest = serde_json::from_str(json).unwrap();
        assert!(req.validate().is_ok());
        
        assert_eq!(req.top_p, 0.9);
        assert_eq!(req.top_k, 50);
        assert_eq!(req.repetition_penalty, 1.1);
        assert_eq!(req.stop.len(), 2);
        assert_eq!(req.min_p, 0.05);
    }

    #[test]
    fn test_validate_all_collects_advanced_parameter_errors() {
        let req = ExecuteRequest {
            job_id: "test".to_string(),
            prompt: "Hello".to_string(),
            max_tokens: 100,
            temperature: 0.7,
            seed: Some(42),
            top_p: 1.5,  // Invalid
            top_k: 0,
            repetition_penalty: 3.0,  // Invalid
            stop: vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string(), "e".to_string()],  // Invalid: too many
            min_p: -0.1,  // Invalid
        };

        let result = req.validate_all();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.errors.len() >= 4);  // At least 4 errors

        let fields: Vec<_> = err.errors.iter().map(|e| e.field.as_str()).collect();
        assert!(fields.contains(&"top_p"));
        assert!(fields.contains(&"repetition_penalty"));
        assert!(fields.contains(&"stop"));
        assert!(fields.contains(&"min_p"));
    }
}

// ---
// Built by Foundation-Alpha 🏗️
