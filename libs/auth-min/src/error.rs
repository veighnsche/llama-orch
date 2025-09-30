//! Error types for auth-min operations

use std::fmt;

/// Authentication errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    /// No token configured in environment
    NoTokenConfigured,

    /// Missing Authorization header
    MissingAuthHeader,

    /// Invalid Authorization header format
    InvalidAuthHeader(String),

    /// Missing Bearer prefix
    MissingBearerPrefix,

    /// Empty token after Bearer prefix
    EmptyToken,

    /// Invalid token (failed comparison)
    InvalidToken,

    /// Bind policy violation
    BindPolicyViolation(String),
}

impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTokenConfigured => {
                write!(f, "No authentication token configured (set LLORCH_API_TOKEN)")
            }
            Self::MissingAuthHeader => {
                write!(f, "Missing Authorization header")
            }
            Self::InvalidAuthHeader(msg) => {
                write!(f, "Invalid Authorization header: {}", msg)
            }
            Self::MissingBearerPrefix => {
                write!(f, "Authorization header missing 'Bearer ' prefix")
            }
            Self::EmptyToken => {
                write!(f, "Empty token after Bearer prefix")
            }
            Self::InvalidToken => {
                write!(f, "Invalid authentication token")
            }
            Self::BindPolicyViolation(msg) => {
                write!(f, "Bind policy violation: {}", msg)
            }
        }
    }
}

impl std::error::Error for AuthError {}

/// Result type for auth-min operations
pub type Result<T> = std::result::Result<T, AuthError>;
