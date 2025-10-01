//! File-based secret loading
//!
//! Loads secrets from filesystem with permission validation and path canonicalization.
//!
//! # Security
//!
//! - Validates file permissions (rejects world/group readable)
//! - Canonicalizes paths (resolves .. and symlinks)
//! - Validates file format (hex for keys, trimmed strings for secrets)

use std::path::Path;
use crate::{Secret, SecretKey, Result, SecretError};
use crate::validation::{validate_file_permissions, canonicalize_path};

/// Load a secret (string) from file
///
/// # Security
///
/// - Validates file permissions (must be 0600 on Unix)
/// - Canonicalizes path (prevents traversal)
/// - Trims whitespace
/// - Rejects empty files
///
/// # Errors
///
/// Returns `SecretError` if:
/// - File not found or cannot be read
/// - File permissions too open (world/group readable on Unix)
/// - File is empty after trimming
/// - Path traversal detected
///
/// # Example
///
/// ```rust,no_run
/// use secrets_management::Secret;
///
/// let token = Secret::load_from_file("/etc/llorch/secrets/api-token")?;
/// # Ok::<(), secrets_management::SecretError>(())
/// ```
pub fn load_secret_from_file(path: impl AsRef<Path>) -> Result<Secret> {
    let path = path.as_ref();
    
    // Canonicalize path (resolve .. and symlinks)
    let canonical = canonicalize_path(path)?;
    
    // Validate file permissions (Unix only)
    validate_file_permissions(&canonical)?;
    
    // Check file size before reading (prevent DoS)
    const MAX_SECRET_SIZE: u64 = 1024 * 1024; // 1MB
    let metadata = std::fs::metadata(&canonical)?;
    if metadata.len() > MAX_SECRET_SIZE {
        return Err(SecretError::InvalidFormat(
            format!("file too large: {} bytes (max: {})", metadata.len(), MAX_SECRET_SIZE)
        ));
    }
    
    // Read file contents
    let contents = std::fs::read_to_string(&canonical)?;
    let trimmed = contents.trim();
    
    if trimmed.is_empty() {
        return Err(SecretError::InvalidFormat("empty file".to_string()));
    }
    
    tracing::info!(path = %canonical.display(), "Secret loaded from file");
    Ok(Secret::new(trimmed.to_string()))
}

/// Load a secret key (32 bytes) from hex file
///
/// # Security
///
/// - Validates file permissions (must be 0600 on Unix)
/// - Canonicalizes path (prevents traversal)
/// - Decodes hex (expects 64 hex chars = 32 bytes)
/// - Validates key length
///
/// # Errors
///
/// Returns `SecretError` if:
/// - File not found or cannot be read
/// - File permissions too open (world/group readable on Unix)
/// - File is empty or not exactly 64 hex characters
/// - Hex decoding fails
/// - Decoded bytes are not exactly 32 bytes
///
/// # Example
///
/// ```rust,no_run
/// use secrets_management::SecretKey;
///
/// let seal_key = SecretKey::load_from_file("/etc/llorch/secrets/seal-key")?;
/// # Ok::<(), secrets_management::SecretError>(())
/// ```
pub fn load_key_from_file(path: impl AsRef<Path>) -> Result<SecretKey> {
    let path = path.as_ref();
    
    // Canonicalize path (resolve .. and symlinks)
    let canonical = canonicalize_path(path)?;
    
    // Validate file permissions (Unix only)
    validate_file_permissions(&canonical)?;
    
    // Check file size before reading (prevent DoS)
    const MAX_KEY_FILE_SIZE: u64 = 1024; // 1KB (64 hex chars + whitespace)
    let metadata = std::fs::metadata(&canonical)?;
    if metadata.len() > MAX_KEY_FILE_SIZE {
        return Err(SecretError::InvalidFormat(
            format!("file too large: {} bytes (max: {})", metadata.len(), MAX_KEY_FILE_SIZE)
        ));
    }
    
    // Read file contents
    let contents = std::fs::read_to_string(&canonical)?;
    let trimmed = contents.trim();
    
    if trimmed.is_empty() {
        return Err(SecretError::InvalidFormat("empty file".to_string()));
    }
    
    // Decode hex (expect 64 hex chars = 32 bytes)
    if trimmed.len() != 64 {
        return Err(SecretError::InvalidFormat(
            format!("expected 64 hex chars, got {}", trimmed.len())
        ));
    }
    
    let bytes = hex::decode(trimmed)
        .map_err(|e| SecretError::InvalidFormat(
            format!("invalid hex encoding: {}", e)
        ))?;
    
    if bytes.len() != 32 {
        return Err(SecretError::InvalidFormat(
            format!("expected 32 bytes, got {}", bytes.len())
        ));
    }
    
    let mut key = [0u8; 32];
    key.copy_from_slice(&bytes);
    
    tracing::info!(path = %canonical.display(), "Secret key loaded from file");
    Ok(SecretKey::new(key))
}

impl Secret {
    /// Load a secret from file (convenience method)
    ///
    /// See [`load_secret_from_file`] for details.
    ///
    /// # Errors
    ///
    /// See [`load_secret_from_file`] for error conditions.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        load_secret_from_file(path)
    }
}

impl SecretKey {
    /// Load a secret key from file (convenience method)
    ///
    /// See [`load_key_from_file`] for details.
    ///
    /// # Errors
    ///
    /// See [`load_key_from_file`] for error conditions.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        load_key_from_file(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    
    #[test]
    fn test_load_secret_from_file_success() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"test-secret-value").unwrap();
        
        // Set correct permissions
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let secret = load_secret_from_file(file.path()).unwrap();
        assert_eq!(secret.expose(), "test-secret-value");
    }
    
    #[test]
    fn test_load_secret_trims_whitespace() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"  test-secret  \n").unwrap();
        
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let secret = load_secret_from_file(file.path()).unwrap();
        assert_eq!(secret.expose(), "test-secret");
    }
    
    #[test]
    fn test_load_secret_rejects_empty_file() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"   \n  ").unwrap();
        
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let result = load_secret_from_file(file.path());
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::InvalidFormat(_))));
    }
    
    #[test]
    fn test_load_secret_rejects_world_readable() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"secret").unwrap();
        
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o644); // World readable
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let result = load_secret_from_file(file.path());
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::PermissionsTooOpen { .. })));
    }
    
    #[test]
    fn test_load_secret_rejects_group_readable() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"secret").unwrap();
        
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o640); // Group readable
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let result = load_secret_from_file(file.path());
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::PermissionsTooOpen { .. })));
    }
    
    #[test]
    fn test_load_key_from_file_success() {
        let mut file = NamedTempFile::new().unwrap();
        // 64 hex chars = 32 bytes
        file.write_all(b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef").unwrap();
        
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let key = load_key_from_file(file.path()).unwrap();
        assert_eq!(key.as_bytes().len(), 32);
    }
    
    #[test]
    fn test_load_key_trims_whitespace() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"  0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  \n").unwrap();
        
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let key = load_key_from_file(file.path()).unwrap();
        assert_eq!(key.as_bytes().len(), 32);
    }
    
    #[test]
    fn test_load_key_rejects_wrong_length() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"0123456789abcdef").unwrap(); // Only 16 hex chars
        
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let result = load_key_from_file(file.path());
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::InvalidFormat(_))));
    }
    
    #[test]
    fn test_load_key_rejects_invalid_hex() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ").unwrap();
        
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let result = load_key_from_file(file.path());
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::InvalidFormat(_))));
    }
    
    #[test]
    fn test_load_key_rejects_world_readable() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef").unwrap();
        
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let result = load_key_from_file(file.path());
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::PermissionsTooOpen { .. })));
    }
    
    #[test]
    fn test_load_nonexistent_file() {
        let result = load_secret_from_file("/nonexistent/path/to/secret");
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::FileNotFound(_))));
    }
    
    #[test]
    fn test_load_secret_rejects_large_file() {
        let mut file = NamedTempFile::new().unwrap();
        // Write 2MB of data (exceeds 1MB limit)
        let large_data = vec![b'a'; 2 * 1024 * 1024];
        file.write_all(&large_data).unwrap();
        
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let result = load_secret_from_file(file.path());
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::InvalidFormat(_))));
        if let Err(SecretError::InvalidFormat(msg)) = result {
            assert!(msg.contains("file too large"));
        }
    }
    
    #[test]
    fn test_load_key_rejects_large_file() {
        let mut file = NamedTempFile::new().unwrap();
        // Write 2KB of data (exceeds 1KB limit)
        let large_data = vec![b'0'; 2048];
        file.write_all(&large_data).unwrap();
        
        let mut perms = std::fs::metadata(file.path()).unwrap().permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(file.path(), perms).unwrap();
        
        let result = load_key_from_file(file.path());
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::InvalidFormat(_))));
        if let Err(SecretError::InvalidFormat(msg)) = result {
            assert!(msg.contains("file too large"));
        }
    }
}
