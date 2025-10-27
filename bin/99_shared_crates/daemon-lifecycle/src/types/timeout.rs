//! Timeout configuration types
//!
//! TEAM-329: Extracted from src/utils/timeout.rs

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for timeout enforcement
///
/// TEAM-276: Wrapper around TimeoutEnforcer for daemon operations
/// TEAM-329: Moved from src/utils/timeout.rs to types/timeout.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Operation name for narration
    pub operation_name: String,

    /// Timeout duration (serialized as seconds)
    #[serde(
        serialize_with = "serialize_duration",
        deserialize_with = "deserialize_duration"
    )]
    pub timeout: Duration,

    /// Optional job_id for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

impl TimeoutConfig {
    /// Create a new timeout config
    pub fn new(operation_name: impl Into<String>, timeout: Duration) -> Self {
        Self { operation_name: operation_name.into(), timeout, job_id: None }
    }

    /// Set the job_id for narration routing
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    /// Set timeout in seconds (convenience method)
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout = Duration::from_secs(secs);
        self
    }
}

// Serde helpers for Duration (serialize as seconds)
fn serialize_duration<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_u64(duration.as_secs())
}

fn deserialize_duration<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let secs = u64::deserialize(deserializer)?;
    Ok(Duration::from_secs(secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeout_config_builder() {
        let config = TimeoutConfig::new("health_check", Duration::from_secs(30))
            .with_job_id("job-123")
            .with_timeout_secs(60);

        assert_eq!(config.operation_name, "health_check");
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.job_id, Some("job-123".to_string()));
    }

    #[test]
    fn test_timeout_config_serialization() {
        let config = TimeoutConfig::new("startup", Duration::from_secs(45))
            .with_job_id("job-456");

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: TimeoutConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.operation_name, deserialized.operation_name);
        assert_eq!(config.timeout, deserialized.timeout);
        assert_eq!(config.job_id, deserialized.job_id);
    }
}
