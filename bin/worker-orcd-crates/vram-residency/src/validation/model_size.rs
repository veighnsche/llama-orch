//! Model size validation
//!
//! Validates model sizes to prevent DoS attacks.

use crate::error::{Result, VramError};

/// Validate model size
///
/// # Arguments
///
/// * `size` - Model size in bytes
/// * `max_size` - Maximum allowed size
///
/// # Returns
///
/// `Ok(())` if valid, error otherwise
pub fn validate_model_size(size: usize, max_size: usize) -> Result<()> {
    if size == 0 {
        return Err(VramError::InvalidInput("model size cannot be zero".to_string()));
    }
    
    if size > max_size {
        return Err(VramError::InvalidInput(format!(
            "model size {} exceeds maximum {}",
            size, max_size
        )));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_model_size() {
        assert!(validate_model_size(1, 1000).is_ok());
        assert!(validate_model_size(500, 1000).is_ok());
        assert!(validate_model_size(1000, 1000).is_ok());
    }

    #[test]
    fn test_zero_size_rejected() {
        let result = validate_model_size(0, 1000);
        assert!(result.is_err());
        assert!(matches!(result, Err(VramError::InvalidInput(_))));
    }

    #[test]
    fn test_size_exceeds_max_rejected() {
        assert!(validate_model_size(1001, 1000).is_err());
        assert!(validate_model_size(2000, 1000).is_err());
        assert!(validate_model_size(usize::MAX, 1000).is_err());
    }

    #[test]
    fn test_boundary_size_equals_max_accepted() {
        assert!(validate_model_size(1000, 1000).is_ok());
        assert!(validate_model_size(1024, 1024).is_ok());
    }

    #[test]
    fn test_boundary_size_one_over_max_rejected() {
        assert!(validate_model_size(1001, 1000).is_err());
    }

    #[test]
    fn test_large_sizes() {
        let one_gb = 1024 * 1024 * 1024;
        assert!(validate_model_size(one_gb, one_gb).is_ok());
        assert!(validate_model_size(one_gb + 1, one_gb).is_err());
    }
}
