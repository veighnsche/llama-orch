//! GET /v1/health endpoint - Health check
//!
//! Per test-001-mvp.md Phase 2: Pool Preflight
//! Returns version and status information
//!
//! Created by: TEAM-026

use axum::Json;
use serde::Serialize;
use tracing::debug;

/// Health check response
///
/// Per test-001-mvp.md lines 46-54
#[derive(Serialize)]
pub struct HealthResponse {
    /// Health status: "alive"
    status: String,
    /// Version: "0.1.0"
    version: String,
    /// API version: "v1"
    api_version: String,
}

/// Handle GET /v1/health
///
/// Returns 200 OK with version and status information
pub async fn handle_health() -> Json<HealthResponse> {
    debug!("Health check requested");

    Json(HealthResponse {
        status: "alive".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        api_version: "v1".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_structure() {
        let response = HealthResponse {
            status: "alive".to_string(),
            version: "0.1.0".to_string(),
            api_version: "v1".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\""));
        assert!(json.contains("\"alive\""));
        assert!(json.contains("\"version\""));
        assert!(json.contains("\"api_version\""));
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "alive".to_string(),
            version: "0.1.0".to_string(),
            api_version: "v1".to_string(),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "alive");
        assert_eq!(json["version"], "0.1.0");
        assert_eq!(json["api_version"], "v1");
    }
}
