//! Daemon status types
//!
//! TEAM-315: Extracted from daemon-lifecycle

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
