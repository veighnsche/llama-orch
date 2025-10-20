//! Health check functionality for queen-rbee
//!
//! Created by: TEAM-043
//! Refactored by: TEAM-052
//! Migrated by: TEAM-151 (2025-10-20)
//!
//! This crate provides the health check endpoint for queen-rbee.
//! Used by rbee-keeper to check if queen is running before sending commands.
//!
//! # Happy Flow Integration
//! From `a_human_wrote_this.md` line 9:
//! > "bee keeper first tests if queen is running? by calling the health."
//!
//! # Endpoint
//! - `GET /health` - Returns 200 OK with version and status

use axum::{response::IntoResponse, Json};
use serde::Serialize;

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// Handle GET /health
///
/// Returns 200 OK with version and status information.
/// This endpoint is public (no authentication required) so rbee-keeper
/// can check if queen is alive before attempting to authenticate.
///
/// # Happy Flow
/// 1. rbee-keeper calls `GET http://localhost:8500/health`
/// 2. If 200 OK → queen is running
/// 3. If connection refused → queen needs to be started
pub async fn handle_health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_endpoint_returns_ok() {
        let response = handle_health().await;
        let json_response = response.into_response();
        
        // Should return 200 OK
        assert_eq!(json_response.status(), 200);
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse {
            status: "ok".to_string(),
            version: "0.1.0".to_string(),
        };
        
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"version\":\"0.1.0\""));
    }
}
