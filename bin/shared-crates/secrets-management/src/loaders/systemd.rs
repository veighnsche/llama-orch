//! Systemd credential loading
//!
//! Loads secrets from systemd LoadCredential mechanism.
//!
//! # Security
//!
//! - Validates $CREDENTIALS_DIRECTORY environment variable
//! - Validates credential name (no path separators)
//! - Uses file loader with permission validation

use std::path::PathBuf;
use crate::{Secret, SecretKey, Result, SecretError};
use crate::loaders::file::{load_secret_from_file, load_key_from_file};

/// Load a secret from systemd credential
///
/// # Security
///
/// - Validates $CREDENTIALS_DIRECTORY is set
/// - Validates credential name (no path separators)
/// - Uses file loader with permission validation
///
/// # Errors
///
/// Returns `SecretError` if:
/// - `$CREDENTIALS_DIRECTORY` environment variable not set
/// - Credential name contains path separators
/// - Credential file not found
/// - File permissions too open
///
/// # Example
///
/// ```rust,no_run
/// use secrets_management::Secret;
///
/// // Systemd service configuration:
/// // [Service]
/// // LoadCredential=api_token:/etc/llorch/secrets/api-token
///
/// let token = Secret::from_systemd_credential("api_token")?;
/// # Ok::<(), secrets_management::SecretError>(())
/// ```
pub fn load_from_systemd_credential(name: &str) -> Result<Secret> {
    // Validate credential name (no path separators, not empty)
    if name.is_empty() {
        return Err(SecretError::PathValidationFailed(
            "credential name cannot be empty".to_string()
        ));
    }
    
    if name.contains('/') || name.contains('\\') {
        return Err(SecretError::PathValidationFailed(
            "credential name cannot contain path separators".to_string()
        ));
    }
    
    // Additional validation: only allow alphanumeric, underscore, hyphen
    if name.contains(|c: char| !c.is_alphanumeric() && c != '_' && c != '-') {
        return Err(SecretError::PathValidationFailed(
            "credential name must contain only alphanumeric, underscore, or hyphen characters".to_string()
        ));
    }
    
    // Get credentials directory from environment
    let creds_dir = std::env::var("CREDENTIALS_DIRECTORY")
        .map_err(|_| SecretError::SystemdCredentialNotFound(
            "CREDENTIALS_DIRECTORY not set".to_string()
        ))?;
    
    // Validate that CREDENTIALS_DIRECTORY is an absolute path
    let creds_path = PathBuf::from(&creds_dir);
    if !creds_path.is_absolute() {
        return Err(SecretError::PathValidationFailed(
            format!("CREDENTIALS_DIRECTORY must be absolute path, got: {}", creds_dir)
        ));
    }
    
    // Construct path
    let path = creds_path.join(name);
    
    if !path.exists() {
        return Err(SecretError::SystemdCredentialNotFound(
            format!("credential not found: {}", name)
        ));
    }
    
    // Use file loader (includes permission validation)
    load_secret_from_file(path)
}

/// Load a secret key from systemd credential
///
/// See [`load_from_systemd_credential`] for details.
pub fn load_key_from_systemd_credential(name: &str) -> Result<SecretKey> {
    // Validate credential name (no path separators, not empty)
    if name.is_empty() {
        return Err(SecretError::PathValidationFailed(
            "credential name cannot be empty".to_string()
        ));
    }
    
    if name.contains('/') || name.contains('\\') {
        return Err(SecretError::PathValidationFailed(
            "credential name cannot contain path separators".to_string()
        ));
    }
    
    // Additional validation: only allow alphanumeric, underscore, hyphen
    if name.contains(|c: char| !c.is_alphanumeric() && c != '_' && c != '-') {
        return Err(SecretError::PathValidationFailed(
            "credential name must contain only alphanumeric, underscore, or hyphen characters".to_string()
        ));
    }
    
    // Get credentials directory from environment
    let creds_dir = std::env::var("CREDENTIALS_DIRECTORY")
        .map_err(|_| SecretError::SystemdCredentialNotFound(
            "CREDENTIALS_DIRECTORY not set".to_string()
        ))?;
    
    // Validate that CREDENTIALS_DIRECTORY is an absolute path
    let creds_path = PathBuf::from(&creds_dir);
    if !creds_path.is_absolute() {
        return Err(SecretError::PathValidationFailed(
            format!("CREDENTIALS_DIRECTORY must be absolute path, got: {}", creds_dir)
        ));
    }
    
    // Construct path
    let path = creds_path.join(name);
    
    if !path.exists() {
        return Err(SecretError::SystemdCredentialNotFound(
            format!("credential not found: {}", name)
        ));
    }
    
    // Use file loader (includes permission validation)
    load_key_from_file(path)
}

impl Secret {
    /// Load from systemd credential (convenience method)
    ///
    /// See [`load_from_systemd_credential`] for details.
    ///
    /// # Errors
    ///
    /// See [`load_from_systemd_credential`] for error conditions.
    pub fn from_systemd_credential(name: &str) -> Result<Self> {
        load_from_systemd_credential(name)
    }
}

impl SecretKey {
    /// Load from systemd credential (convenience method)
    ///
    /// See [`load_key_from_systemd_credential`] for details.
    ///
    /// # Errors
    ///
    /// See [`load_key_from_systemd_credential`] for error conditions.
    pub fn from_systemd_credential(name: &str) -> Result<Self> {
        load_key_from_systemd_credential(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    
    #[test]
    fn test_load_from_systemd_credential_success() {
        let temp_dir = TempDir::new().unwrap();
        std::env::set_var("CREDENTIALS_DIRECTORY", temp_dir.path().to_str().unwrap());
        
        // Create credential file with correct permissions from the start
        let cred_path = temp_dir.path().join("api_token");
        let mut file = fs::File::create(&cred_path).unwrap();
        let mut perms = file.metadata().unwrap().permissions();
        perms.set_mode(0o600);
        fs::set_permissions(&cred_path, perms).unwrap();
        
        use std::io::Write;
        file.write_all(b"test-credential-value").unwrap();
        drop(file);
        
        let secret = load_from_systemd_credential("api_token").unwrap();
        assert_eq!(secret.expose(), "test-credential-value");
        
        std::env::remove_var("CREDENTIALS_DIRECTORY");
    }
    
    #[test]
    fn test_load_from_systemd_credential_not_set() {
        std::env::remove_var("CREDENTIALS_DIRECTORY");
        
        let result = load_from_systemd_credential("api_token");
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::SystemdCredentialNotFound(_))));
    }
    
    #[test]
    fn test_load_from_systemd_credential_rejects_path_separators() {
        let temp_dir = TempDir::new().unwrap();
        std::env::set_var("CREDENTIALS_DIRECTORY", temp_dir.path().to_str().unwrap());
        
        let result = load_from_systemd_credential("../etc/passwd");
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::PathValidationFailed(_))));
        
        std::env::remove_var("CREDENTIALS_DIRECTORY");
    }
    
    #[test]
    fn test_load_from_systemd_credential_file_not_found() {
        let temp_dir = TempDir::new().unwrap();
        std::env::set_var("CREDENTIALS_DIRECTORY", temp_dir.path().to_str().unwrap());
        
        let result = load_from_systemd_credential("nonexistent");
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::SystemdCredentialNotFound(_))));
        
        std::env::remove_var("CREDENTIALS_DIRECTORY");
    }
    
    #[test]
    fn test_load_from_systemd_credential_validates_permissions() {
        let temp_dir = TempDir::new().unwrap();
        std::env::set_var("CREDENTIALS_DIRECTORY", temp_dir.path().to_str().unwrap());
        
        let cred_path = temp_dir.path().join("api_token");
        fs::write(&cred_path, "test-value").unwrap();
        
        // Set world-readable permissions
        let mut perms = fs::metadata(&cred_path).unwrap().permissions();
        perms.set_mode(0o644);
        fs::set_permissions(&cred_path, perms).unwrap();
        
        let result = load_from_systemd_credential("api_token");
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::PermissionsTooOpen { .. })));
        
        std::env::remove_var("CREDENTIALS_DIRECTORY");
    }
    
    #[test]
    fn test_load_key_from_systemd_credential_success() {
        let temp_dir = TempDir::new().unwrap();
        std::env::set_var("CREDENTIALS_DIRECTORY", temp_dir.path().to_str().unwrap());
        
        let cred_path = temp_dir.path().join("seal_key");
        let mut file = fs::File::create(&cred_path).unwrap();
        let mut perms = file.metadata().unwrap().permissions();
        perms.set_mode(0o600);
        fs::set_permissions(&cred_path, perms).unwrap();
        
        use std::io::Write;
        file.write_all(b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef").unwrap();
        drop(file);
        
        let key = load_key_from_systemd_credential("seal_key").unwrap();
        assert_eq!(key.as_bytes().len(), 32);
        
        std::env::remove_var("CREDENTIALS_DIRECTORY");
    }
    
    #[test]
    fn test_load_from_systemd_credential_rejects_empty_name() {
        let temp_dir = TempDir::new().unwrap();
        std::env::set_var("CREDENTIALS_DIRECTORY", temp_dir.path().to_str().unwrap());
        
        let result = load_from_systemd_credential("");
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::PathValidationFailed(_))));
        
        std::env::remove_var("CREDENTIALS_DIRECTORY");
    }
    
    #[test]
    fn test_load_from_systemd_credential_rejects_invalid_chars() {
        let temp_dir = TempDir::new().unwrap();
        std::env::set_var("CREDENTIALS_DIRECTORY", temp_dir.path().to_str().unwrap());
        
        let result = load_from_systemd_credential("api@token");
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::PathValidationFailed(_))));
        
        std::env::remove_var("CREDENTIALS_DIRECTORY");
    }
    
    #[test]
    fn test_load_from_systemd_credential_rejects_relative_path() {
        std::env::set_var("CREDENTIALS_DIRECTORY", "relative/path");
        
        let result = load_from_systemd_credential("api_token");
        assert!(result.is_err());
        assert!(matches!(result, Err(SecretError::PathValidationFailed(_))));
        if let Err(SecretError::PathValidationFailed(msg)) = result {
            assert!(msg.contains("absolute path"));
        }
        
        std::env::remove_var("CREDENTIALS_DIRECTORY");
    }
}
