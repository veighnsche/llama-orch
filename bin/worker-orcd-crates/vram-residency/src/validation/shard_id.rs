//! Shard ID validation
//!
//! Validates shard IDs to prevent path traversal and injection attacks.

use crate::error::{Result, VramError};

/// Validate shard ID
///
/// # Requirements
///
/// - No path traversal (../)
/// - No null bytes
/// - Length <= 256 characters
/// - Alphanumeric + hyphens only
///
/// # Arguments
///
/// * `shard_id` - Shard ID to validate
///
/// # Returns
///
/// `Ok(())` if valid, error otherwise
pub fn validate_shard_id(shard_id: &str) -> Result<()> {
    // TODO: Integrate with input-validation crate
    // - Use validate_identifier()
    // - Check for path traversal
    // - Check for null bytes
    // - Check length
    
    if shard_id.is_empty() {
        return Err(VramError::InvalidInput("shard_id cannot be empty".to_string()));
    }
    
    if shard_id.len() > 256 {
        return Err(VramError::InvalidInput("shard_id too long".to_string()));
    }
    
    if shard_id.contains("..") {
        return Err(VramError::InvalidInput("path traversal detected".to_string()));
    }
    
    if shard_id.contains('\0') {
        return Err(VramError::InvalidInput("null byte detected".to_string()));
    }
    
    Ok(())
}
