//! Error types for jwt-guardian

use thiserror::Error;

/// Result type alias for jwt-guardian operations
pub type Result<T> = std::result::Result<T, JwtError>;

/// JWT validation and lifecycle errors
#[derive(Debug, Error)]
pub enum JwtError {
    /// Signature verification failed
    #[error("Invalid JWT signature")]
    InvalidSignature,

    /// Token past expiration time
    #[error("Token expired")]
    TokenExpired,

    /// Token in revocation list
    #[error("Token revoked")]
    TokenRevoked,

    /// Issuer claim mismatch
    #[error("Invalid issuer: expected {expected}, got {actual}")]
    InvalidIssuer { expected: String, actual: String },

    /// Audience claim mismatch
    #[error("Invalid audience: expected {expected}, got {actual}")]
    InvalidAudience { expected: String, actual: String },

    /// Required claim missing
    #[error("Missing required claim: {0}")]
    MissingClaim(String),

    /// Algorithm not in whitelist
    #[error("Algorithm mismatch: {0}")]
    AlgorithmMismatch(String),

    /// Malformed token
    #[error("Invalid token format: {0}")]
    InvalidFormat(String),

    /// Time drift too large
    #[error("Clock skew exceeded tolerance")]
    ClockSkewExceeded,

    /// Redis connection or operation error
    #[cfg(feature = "revocation")]
    #[error("Redis error: {0}")]
    RedisError(String),

    /// Generic validation error
    #[error("Validation error: {0}")]
    ValidationError(String),
}

impl From<jsonwebtoken::errors::Error> for JwtError {
    fn from(err: jsonwebtoken::errors::Error) -> Self {
        use jsonwebtoken::errors::ErrorKind;

        match err.kind() {
            ErrorKind::InvalidSignature => JwtError::InvalidSignature,
            ErrorKind::ExpiredSignature => JwtError::TokenExpired,
            ErrorKind::InvalidIssuer => JwtError::ValidationError("Invalid issuer".to_string()),
            ErrorKind::InvalidAudience => JwtError::ValidationError("Invalid audience".to_string()),
            ErrorKind::InvalidToken => JwtError::InvalidFormat("Malformed token".to_string()),
            _ => JwtError::ValidationError(err.to_string()),
        }
    }
}

#[cfg(feature = "revocation")]
impl From<redis::RedisError> for JwtError {
    fn from(err: redis::RedisError) -> Self {
        JwtError::RedisError(err.to_string())
    }
}
