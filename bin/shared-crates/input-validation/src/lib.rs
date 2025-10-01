//! input-validation — Input sanitization and validation
//!
//! Prevents injection attacks, path traversal, and malformed input from reaching core logic.
//!
//! # Security Properties
//!
//! - Validates model references (no traversal)
//! - Validates filesystem paths (no ../escape)
//! - Validates prompts (length, chars, no null bytes)
//! - Validates parameters (ranges, types)
//!
//! # Example
//!
//! ```rust
//! use input_validation::{validate_model_ref, validate_prompt};
//!
//! // Validate model reference
//! validate_model_ref("hf:meta-llama/Llama-3.1-8B")?;
//! // validate_model_ref("hf:../../../etc/passwd")?; // ERROR: path traversal
//!
//! // Validate prompt
//! validate_prompt("Hello, world!", 100_000)?;
//! // validate_prompt("A\0B", 100)?; // ERROR: null byte
//! ```

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::must_use_candidate)]

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("input too long: {0} > {1}")]
    TooLong(usize, usize),
    #[error("input too short: {0} < {1}")]
    TooShort(usize, usize),
    #[error("invalid characters: {0}")]
    InvalidChars(String),
    #[error("path traversal detected: {0}")]
    PathTraversal(String),
    #[error("null byte detected")]
    NullByte,
    #[error("out of range: {0}")]
    OutOfRange(String),
}

pub type Result<T> = std::result::Result<T, ValidationError>;

/// Validate model reference (hf:org/repo, file:path, etc.)
pub fn validate_model_ref(model_ref: &str) -> Result<()> {
    const MAX_LEN: usize = 512;
    
    if model_ref.is_empty() {
        return Err(ValidationError::TooShort(0, 1));
    }
    
    if model_ref.len() > MAX_LEN {
        return Err(ValidationError::TooLong(model_ref.len(), MAX_LEN));
    }
    
    // Check for null bytes
    if model_ref.contains('\0') {
        return Err(ValidationError::NullByte);
    }
    
    // Check for path traversal
    if model_ref.contains("..") {
        return Err(ValidationError::PathTraversal(model_ref.to_string()));
    }
    
    // Validate characters (alphanumeric + safe punctuation)
    let valid_chars = model_ref.chars().all(|c| {
        c.is_alphanumeric() || matches!(c, '-' | '_' | '/' | ':' | '.')
    });
    
    if !valid_chars {
        return Err(ValidationError::InvalidChars(model_ref.to_string()));
    }
    
    Ok(())
}

/// Validate prompt text
pub fn validate_prompt(prompt: &str, max_len: usize) -> Result<()> {
    if prompt.is_empty() {
        return Err(ValidationError::TooShort(0, 1));
    }
    
    if prompt.len() > max_len {
        return Err(ValidationError::TooLong(prompt.len(), max_len));
    }
    
    // Check for null bytes
    if prompt.contains('\0') {
        return Err(ValidationError::NullByte);
    }
    
    // Validate UTF-8 (already guaranteed by &str, but check boundaries)
    if !prompt.is_char_boundary(0) {
        return Err(ValidationError::InvalidChars("Invalid UTF-8".to_string()));
    }
    
    Ok(())
}

/// Validate integer parameter in range
pub fn validate_int_range(value: i64, min: i64, max: i64, name: &str) -> Result<i64> {
    if value < min || value > max {
        return Err(ValidationError::OutOfRange(
            format!("{} must be between {} and {}, got {}", name, min, max, value)
        ));
    }
    Ok(value)
}

/// Validate float parameter in range
pub fn validate_float_range(value: f64, min: f64, max: f64, name: &str) -> Result<f64> {
    if value < min || value > max || value.is_nan() {
        return Err(ValidationError::OutOfRange(
            format!("{} must be between {} and {}, got {}", name, min, max, value)
        ));
    }
    Ok(value)
}

/// Validate filesystem path (no traversal)
pub fn validate_path(path: &str) -> Result<()> {
    if path.is_empty() {
        return Err(ValidationError::TooShort(0, 1));
    }
    
    // Check for path traversal
    if path.contains("..") {
        return Err(ValidationError::PathTraversal(path.to_string()));
    }
    
    // Check for null bytes
    if path.contains('\0') {
        return Err(ValidationError::NullByte);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_model_ref() {
        // Valid
        assert!(validate_model_ref("hf:meta-llama/Llama-3.1-8B").is_ok());
        assert!(validate_model_ref("file:/models/model.gguf").is_ok());
        
        // Invalid
        assert!(validate_model_ref("hf:../../../etc/passwd").is_err());
        assert!(validate_model_ref("model\0name").is_err());
        assert!(validate_model_ref("").is_err());
    }
    
    #[test]
    fn test_validate_prompt() {
        assert!(validate_prompt("Hello, world!", 100).is_ok());
        assert!(validate_prompt("A\0B", 100).is_err()); // Null byte
        assert!(validate_prompt("", 100).is_err()); // Empty
        assert!(validate_prompt("A".repeat(1000).as_str(), 100).is_err()); // Too long
    }
    
    #[test]
    fn test_validate_ranges() {
        assert!(validate_int_range(50, 0, 100, "value").is_ok());
        assert!(validate_int_range(150, 0, 100, "value").is_err());
        
        assert!(validate_float_range(0.5, 0.0, 1.0, "temp").is_ok());
        assert!(validate_float_range(1.5, 0.0, 1.0, "temp").is_err());
    }
}
