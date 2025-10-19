//! GET /health endpoint - Health check
//!
//! Created by: TEAM-043
//! Refactored by: TEAM-052

use axum::{response::IntoResponse, Json};

use crate::http::types::HealthResponse;

/// Handle GET /health
///
/// Returns 200 OK with version and status information
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
    async fn test_health_endpoint() {
        let response = handle_health().await;
        // Response should be Json type
        let _json = response.into_response();
    }
}
