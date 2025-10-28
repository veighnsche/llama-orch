//! Shared error types
//!
//! TEAM-284: Common error definitions for contract operations

use thiserror::Error;

/// Contract-related errors
#[derive(Debug, Error)]
pub enum ContractError {
    /// Component not found in registry
    #[error("Component not found: {0}")]
    ComponentNotFound(String),

    /// Heartbeat is stale (too old)
    #[error("Heartbeat is stale: {age_secs}s old (timeout: {timeout_secs}s)")]
    StaleHeartbeat {
        /// Age of heartbeat in seconds
        age_secs: i64,
        /// Timeout threshold in seconds
        timeout_secs: u64,
    },

    /// Invalid component state
    #[error("Invalid component state: {0}")]
    InvalidState(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
}

impl From<serde_json::Error> for ContractError {
    fn from(err: serde_json::Error) -> Self {
        ContractError::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = ContractError::ComponentNotFound("worker-123".to_string());
        assert_eq!(err.to_string(), "Component not found: worker-123");

        let err = ContractError::StaleHeartbeat { age_secs: 120, timeout_secs: 90 };
        assert!(err.to_string().contains("120s old"));
        assert!(err.to_string().contains("timeout: 90s"));
    }
}
