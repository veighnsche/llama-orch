//! Error types for audit logging
//!
//! All errors are structured and never expose sensitive information.

use thiserror::Error;

/// Audit logging errors
#[derive(Error, Debug)]
pub enum AuditError {
    /// Buffer is full, event dropped
    #[error("Audit buffer full, event dropped")]
    BufferFull,

    /// I/O error during file operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Hash chain validation failed
    #[error("Invalid hash chain: {0}")]
    InvalidChain(String),

    /// Hash chain broken at specific event
    #[error("Broken hash chain at event {0}")]
    BrokenChain(String),

    /// File checksum mismatch
    #[error("Checksum mismatch for {file}")]
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },

    /// Missing signature (platform mode)
    #[error("Missing signature")]
    MissingSignature,

    /// Invalid signature (platform mode)
    #[error("Invalid signature for event {audit_id}")]
    InvalidSignature { audit_id: String },

    /// Invalid path (security)
    #[error("Invalid path: {0}")]
    InvalidPath(String),

    /// Field too long (security)
    #[error("Field too long: {0}")]
    FieldTooLong(&'static str),

    /// Integer overflow (security)
    #[error("Integer overflow: {0}")]
    IntegerOverflow(&'static str),

    /// Unauthorized access
    #[error("Unauthorized")]
    Unauthorized,

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    /// Query too large
    #[error("Query too large")]
    QueryTooLarge,

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Counter overflow
    #[error("Audit counter overflow")]
    CounterOverflow,
    
    /// Disk space low
    #[error("Disk space low: {available} bytes available, {required} bytes required")]
    DiskSpaceLow { available: u64, required: u64 },
    
    /// File rotation failed
    #[error("File rotation failed: {0}")]
    RotationFailed(String),
    
    /// Invalid file permissions
    #[error("Invalid file permissions: expected 0600")]
    InvalidPermissions,
}

/// Result type for audit operations
pub type Result<T> = std::result::Result<T, AuditError>;
