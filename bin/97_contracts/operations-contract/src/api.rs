//! API specification for hive operations
//!
//! TEAM-284: Defines the HTTP API contract between queen and hive
//!
//! This ensures both sides use the same endpoints and formats.

use serde::{Deserialize, Serialize};

/// Hive API specification
///
/// All hives MUST implement these endpoints
pub struct HiveApiSpec;

impl HiveApiSpec {
    /// Create a new job
    ///
    /// POST /v1/jobs
    /// Body: Operation (JSON)
    /// Returns: JobResponse
    pub const CREATE_JOB: &'static str = "/v1/jobs";

    /// Stream job results
    ///
    /// GET /v1/jobs/{job_id}/stream
    /// Returns: Server-Sent Events (SSE) stream
    pub const STREAM_JOB: &'static str = "/v1/jobs/{job_id}/stream";

    /// Health check
    ///
    /// GET /health
    /// Returns: HealthResponse
    pub const HEALTH: &'static str = "/health";

    /// Get capabilities
    ///
    /// GET /v1/capabilities
    /// Returns: CapabilitiesResponse
    pub const CAPABILITIES: &'static str = "/v1/capabilities";
}

/// Response from creating a job
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobResponse {
    /// Job ID
    pub job_id: String,
    /// SSE stream URL
    pub sse_url: String,
}

impl JobResponse {
    /// Create a job response
    pub fn new(job_id: String) -> Self {
        Self { sse_url: format!("/v1/jobs/{}/stream", job_id), job_id }
    }
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthResponse {
    /// Status (e.g., "ok")
    pub status: String,
    /// Optional message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl HealthResponse {
    /// Create a healthy response
    pub fn ok() -> Self {
        Self { status: "ok".to_string(), message: None }
    }

    /// Create an unhealthy response
    pub fn error(message: String) -> Self {
        Self { status: "error".to_string(), message: Some(message) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_endpoints() {
        assert_eq!(HiveApiSpec::CREATE_JOB, "/v1/jobs");
        assert_eq!(HiveApiSpec::STREAM_JOB, "/v1/jobs/{job_id}/stream");
        assert_eq!(HiveApiSpec::HEALTH, "/health");
    }

    #[test]
    fn test_job_response_creates_sse_url() {
        let response = JobResponse::new("job-123".to_string());

        assert_eq!(response.job_id, "job-123");
        assert_eq!(response.sse_url, "/v1/jobs/job-123/stream");
    }

    #[test]
    fn test_health_response_ok() {
        let response = HealthResponse::ok();

        assert_eq!(response.status, "ok");
        assert_eq!(response.message, None);

        let json = serde_json::to_string(&response).unwrap();
        assert!(!json.contains("\"message\""));
    }

    #[test]
    fn test_health_response_error() {
        let response = HealthResponse::error("Something went wrong".to_string());

        assert_eq!(response.status, "error");
        assert_eq!(response.message, Some("Something went wrong".to_string()));
    }
}
