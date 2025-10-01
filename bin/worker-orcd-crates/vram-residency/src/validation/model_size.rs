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
