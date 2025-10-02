//! Error types for model loading and validation
//!
//! All errors are structured with actionable diagnostics.
//! Error messages MUST NOT expose sensitive data (file contents, paths).

use thiserror::Error;

/// Model loading and validation errors
#[derive(Debug, Error)]
pub enum LoadError {
    /// I/O error (file not found, permission denied)
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Hash mismatch (integrity violation)
    #[error("hash mismatch: expected {expected}, got {actual}")]
    HashMismatch {
        expected: String,
        actual: String,
    },
    
    /// Model file too large
    #[error("model too large: {actual} bytes (max {max} bytes)")]
    TooLarge {
        actual: usize,
        max: usize,
    },
    
    /// Invalid GGUF format
    #[error("invalid GGUF format: {0}")]
    InvalidFormat(String),
    
    /// Signature verification failed
    #[error("signature verification failed")]
    SignatureVerificationFailed,
    
    /// Path validation failed (traversal attempt, outside allowed directory)
    #[error("path validation failed: {0}")]
    PathValidationFailed(String),
    
    /// Tensor count exceeds maximum allowed
    #[error("tensor count {count} exceeds maximum {max}")]
    TensorCountExceeded {
        count: usize,
        max: usize,
    },
    
    /// String length exceeds maximum allowed
    #[error("string length {length} exceeds maximum {max}")]
    StringTooLong {
        length: usize,
        max: usize,
    },
    
    /// Invalid data type enum value
    #[error("invalid data type: {0}")]
    InvalidDataType(u8),
    
    /// Buffer overflow detected
    #[error("buffer overflow: tried to read {length} bytes at offset {offset}, but only {available} bytes available")]
    BufferOverflow {
        offset: usize,
        length: usize,
        available: usize,
    },
}

/// Result type alias for model loader operations
pub type Result<T> = std::result::Result<T, LoadError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let err = LoadError::HashMismatch {
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        
        assert!(err.to_string().contains("hash mismatch"));
        assert!(err.to_string().contains("abc123"));
    }
    
    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let load_err: LoadError = io_err.into();
        
        assert!(matches!(load_err, LoadError::Io(_)));
    }
}
