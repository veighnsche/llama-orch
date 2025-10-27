//! Health check types
//!
//! TEAM-315: Extracted from daemon-lifecycle
//! TEAM-329: Moved from daemon-contract to daemon-lifecycle/types (inline)
//! TEAM-329: Renamed status.rs → health.rs (matches src/health.rs)
//! TEAM-329: Added HealthPollConfig from src/health.rs

use serde::{Deserialize, Serialize};

/// Request to check daemon status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusRequest {
    /// ID of the daemon instance (e.g., alias, worker ID)
    pub id: String,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

impl StatusRequest {
    /// Create a new status request
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            job_id: None,
        }
    }

    /// Set job_id for narration
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }
}

/// Response from daemon status check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    /// ID of the daemon instance
    pub id: String,

    /// Whether the daemon is running
    pub is_running: bool,

    /// Health status (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_status: Option<String>,

    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_request_serialization() {
        let request = StatusRequest {
            id: "test-daemon".to_string(),
            job_id: Some("job-123".to_string()),
        };
        let json = serde_json::to_string(&request).unwrap();
        let deserialized: StatusRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(request.id, deserialized.id);
        assert_eq!(request.job_id, deserialized.job_id);
    }

    #[test]
    fn test_status_response_serialization() {
        let response = StatusResponse {
            id: "test-daemon".to_string(),
            is_running: true,
            health_status: Some("healthy".to_string()),
            metadata: None,
        };
        let json = serde_json::to_string(&response).unwrap();
        let deserialized: StatusResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response.id, deserialized.id);
        assert_eq!(response.is_running, deserialized.is_running);
    }
}

/// Configuration for health polling with exponential backoff
///
/// TEAM-276: Added for daemon startup synchronization
/// TEAM-329: Moved from src/health.rs to types/health.rs
pub struct HealthPollConfig {
    /// Base URL of daemon (e.g., "http://localhost:8500")
    pub base_url: String,

    /// Optional health endpoint path (default: "/health")
    pub health_endpoint: Option<String>,

    /// Maximum number of polling attempts (default: 10)
    pub max_attempts: usize,

    /// Initial delay in milliseconds (default: 200ms)
    pub initial_delay_ms: u64,

    /// Backoff multiplier for exponential backoff (default: 1.5)
    pub backoff_multiplier: f64,

    /// Optional job_id for narration routing
    pub job_id: Option<String>,

    /// Optional daemon name for narration (default: "daemon")
    pub daemon_name: Option<String>,
}

impl Default for HealthPollConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            health_endpoint: None,
            max_attempts: 10,
            initial_delay_ms: 200,
            backoff_multiplier: 1.5,
            job_id: None,
            daemon_name: None,
        }
    }
}

impl HealthPollConfig {
    /// Create a new config with just the base URL
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { base_url: base_url.into(), ..Default::default() }
    }

    /// Set the health endpoint
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.health_endpoint = Some(endpoint.into());
        self
    }

    /// Set the maximum attempts
    pub fn with_max_attempts(mut self, attempts: usize) -> Self {
        self.max_attempts = attempts;
        self
    }

    /// Set the job_id for narration
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    /// Set the daemon name for narration
    pub fn with_daemon_name(mut self, name: impl Into<String>) -> Self {
        self.daemon_name = Some(name.into());
        self
    }
}
