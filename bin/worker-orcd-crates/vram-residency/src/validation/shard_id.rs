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
/// - Alphanumeric + hyphens + underscores only
///
/// # Arguments
///
/// * `shard_id` - Shard ID to validate
///
/// # Returns
///
/// `Ok(())` if valid, error otherwise
///
/// # Security
///
/// Prevents path traversal, injection attacks, and buffer overflows.
pub fn validate_shard_id(shard_id: &str) -> Result<()> {
    // Check empty
    if shard_id.is_empty() {
        return Err(VramError::InvalidInput("shard_id cannot be empty".to_string()));
    }
    
    // Check length (prevent buffer overflow)
    if shard_id.len() > 256 {
        return Err(VramError::InvalidInput(format!(
            "shard_id too long: {} bytes (max: 256)",
            shard_id.len()
        )));
    }
    
    // Check for path traversal
    if shard_id.contains("..") || shard_id.contains('/') || shard_id.contains('\\') {
        return Err(VramError::InvalidInput(
            "shard_id contains path traversal characters".to_string()
        ));
    }
    
    // Check for null bytes (C string injection)
    if shard_id.contains('\0') {
        return Err(VramError::InvalidInput(
            "shard_id contains null byte".to_string()
        ));
    }
    
    // Check for control characters
    if shard_id.chars().any(|c| c.is_control()) {
        return Err(VramError::InvalidInput(
            "shard_id contains control characters".to_string()
        ));
    }
    
    // Check for valid characters (alphanumeric + hyphen + underscore + colon)
    if !shard_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == ':') {
        return Err(VramError::InvalidInput(
            "shard_id contains invalid characters (only alphanumeric, -, _, : allowed)".to_string()
        ));
    }
    
    tracing::debug!(
        shard_id = %shard_id,
        len = %shard_id.len(),
        "Shard ID validated"
    );
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_shard_id_alphanumeric() {
        assert!(validate_shard_id("shard-123").is_ok());
        assert!(validate_shard_id("abc_def_456").is_ok());
        assert!(validate_shard_id("model:v1:shard-0").is_ok());
    }

    #[test]
    fn test_valid_shard_id_with_allowed_chars() {
        assert!(validate_shard_id("shard-abc-123_xyz:v1").is_ok());
        assert!(validate_shard_id("0123456789").is_ok());
        assert!(validate_shard_id("ABCDEFGHIJKLMNOPQRSTUVWXYZ").is_ok());
    }

    #[test]
    fn test_empty_shard_id_rejected() {
        let result = validate_shard_id("");
        assert!(result.is_err());
        assert!(matches!(result, Err(VramError::InvalidInput(_))));
    }

    #[test]
    fn test_too_long_shard_id_rejected() {
        let long_id = "a".repeat(257);
        let result = validate_shard_id(&long_id);
        assert!(result.is_err());
        assert!(matches!(result, Err(VramError::InvalidInput(_))));
    }

    #[test]
    fn test_exactly_256_chars_accepted() {
        let id = "a".repeat(256);
        assert!(validate_shard_id(&id).is_ok());
    }

    #[test]
    fn test_path_traversal_with_dotdot_rejected() {
        assert!(validate_shard_id("../etc/passwd").is_err());
        assert!(validate_shard_id("shard-..").is_err());
        assert!(validate_shard_id("..").is_err());
    }

    #[test]
    fn test_path_traversal_with_slash_rejected() {
        assert!(validate_shard_id("shard/123").is_err());
        assert!(validate_shard_id("/etc/passwd").is_err());
        assert!(validate_shard_id("a/b/c").is_err());
    }

    #[test]
    fn test_path_traversal_with_backslash_rejected() {
        assert!(validate_shard_id("shard\\123").is_err());
        assert!(validate_shard_id("C:\\Windows").is_err());
    }

    #[test]
    fn test_null_byte_injection_rejected() {
        assert!(validate_shard_id("shard\0null").is_err());
        assert!(validate_shard_id("\0").is_err());
    }

    #[test]
    fn test_control_characters_rejected() {
        assert!(validate_shard_id("shard\n123").is_err());
        assert!(validate_shard_id("shard\r\n").is_err());
        assert!(validate_shard_id("shard\t123").is_err());
        assert!(validate_shard_id("\x01\x02\x03").is_err());
    }

    #[test]
    fn test_invalid_special_characters_rejected() {
        assert!(validate_shard_id("shard@123").is_err());
        assert!(validate_shard_id("shard#123").is_err());
        assert!(validate_shard_id("shard$123").is_err());
        assert!(validate_shard_id("shard%123").is_err());
        assert!(validate_shard_id("shard&123").is_err());
        assert!(validate_shard_id("shard*123").is_err());
    }

    #[test]
    fn test_boundary_length_257_rejected() {
        let id = "a".repeat(257);
        assert!(validate_shard_id(&id).is_err());
    }
}
