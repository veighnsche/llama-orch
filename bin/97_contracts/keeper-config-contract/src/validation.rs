//! Configuration validation errors
//!
//! TEAM-315: Validation error types for keeper configuration

use thiserror::Error;

/// Configuration validation error
#[derive(Debug, Clone, Error)]
pub enum ValidationError {
    /// Invalid port number
    #[error("Invalid port {port}: {reason}")]
    InvalidPort {
        /// The invalid port number
        port: u16,
        /// Reason why it's invalid
        reason: String,
    },
}
