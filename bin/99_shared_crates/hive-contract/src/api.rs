//! Hive HTTP API specification
//!
//! TEAM-284: Defines the HTTP endpoints all hives must implement

use serde::{Deserialize, Serialize};

/// Hive API specification
///
/// All hive implementations must provide these HTTP endpoints.
///
/// # Endpoints
///
/// - `GET /health` - Health check
/// - `GET /capabilities` - Device capabilities
/// - `POST /workers` - Spawn worker
/// - `GET /workers` - List workers
/// - `DELETE /workers/{id}` - Stop worker
pub struct HiveApiSpec;

impl HiveApiSpec {
    /// Health check endpoint
    pub const HEALTH: &'static str = "/health";
    
    /// Capabilities endpoint
    pub const CAPABILITIES: &'static str = "/capabilities";
    
    /// Workers endpoint
    pub const WORKERS: &'static str = "/workers";
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Status (e.g., "ok")
    pub status: String,
    
    /// Optional message
    pub message: Option<String>,
}

impl HealthResponse {
    /// Create a healthy response
    pub fn ok() -> Self {
        Self {
            status: "ok".to_string(),
            message: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_spec_endpoints() {
        assert_eq!(HiveApiSpec::HEALTH, "/health");
        assert_eq!(HiveApiSpec::CAPABILITIES, "/capabilities");
        assert_eq!(HiveApiSpec::WORKERS, "/workers");
    }

    #[test]
    fn health_response_ok() {
        let response = HealthResponse::ok();
        assert_eq!(response.status, "ok");
        assert_eq!(response.message, None);
    }
}
