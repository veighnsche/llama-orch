//! Error types for secrets-management
//!
//! Defines all error conditions that can occur during secret loading, validation,
//! and key derivation operations.
//!
//! # Security Note
//!
//! Error messages MUST NOT contain secret values. Only metadata (paths, lengths,
//! permission modes) should be included in error messages.

use thiserror::Error;

/// Errors that can occur during secret operations
#[derive(Debug, Clone, Error)]
pub enum SecretError {
    /// File not found at specified path
    #[error("secret file not found: {0}")]
    FileNotFound(String),
    
    /// File permissions are too open (world or group readable)
    #[error("file permissions too open: {path} (mode: {mode:o}, expected 0600)")]
    PermissionsTooOpen {
        path: String,
        mode: u32,
    },
    
    /// Invalid secret format (empty, wrong length, bad encoding)
    #[error("invalid secret format: {0}")]
    InvalidFormat(String),
    
    /// Systemd credential not found
    #[error("systemd credential not found: {0}")]
    SystemdCredentialNotFound(String),
    
    /// Path validation failed (traversal, outside allowed directory)
    #[error("path validation failed: {0}")]
    PathValidationFailed(String),
    
    /// I/O error occurred
    #[error("I/O error: {0}")]
    Io(String),
    
    /// Key derivation error (HKDF failure)
    #[error("key derivation error: {0}")]
    KeyDerivation(String),
}

impl From<std::io::Error> for SecretError {
    fn from(e: std::io::Error) -> Self {
        SecretError::Io(e.to_string())
    }
}

/// Result type alias for secret operations
pub type Result<T> = std::result::Result<T, SecretError>;
