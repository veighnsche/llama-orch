//! GET /health endpoint - Health check
//!
//! Returns a simple health status for the worker.
//!
//! # Spec References
//! - M0-W-1320: Health endpoint returns 200 OK with {"status": "healthy"}

use crate::http::routes::AppState;
use axum::{
    extract::{Extension, State},
    Json,
};
use serde::Serialize;
use tracing::debug;

/// Health check response
///
/// Per spec M0-W-1320, this endpoint returns a simple status indicator.
#[derive(Serialize)]
pub struct HealthResponse {
    /// Health status: "healthy" or "unhealthy"
    status: String,
}

/// Handle GET /health
///
/// Returns 200 OK with `{"status": "healthy"}` if the worker is operational.
///
/// # Future Enhancements (M1+)
/// - VRAM residency check
/// - Uptime tracking
/// - Model load status
pub async fn handle_health(
    Extension(correlation_id): Extension<String>,
    State(_state): State<AppState>,
) -> Json<HealthResponse> {
    debug!(
        correlation_id = %correlation_id,
        "Health check requested"
    );

    // Per spec: simple health check
    // VRAM checks and detailed status deferred to M1+
    Json(HealthResponse { status: "healthy".to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_structure() {
        let response = HealthResponse { status: "healthy".to_string() };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\""));
        assert!(json.contains("\"healthy\""));
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse { status: "healthy".to_string() };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "healthy");
    }
}

// ---
// Built by Foundation-Alpha 🏗️
