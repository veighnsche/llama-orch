//! Health check endpoints
//!
//! Per test-001-mvp.md Phase 2: Pool Preflight
//! Returns version and status information
//!
//! # Endpoints
//! - `GET /v1/health` - Basic health check
//! - `GET /health/live` - Kubernetes liveness probe (TEAM-104)
//! - `GET /health/ready` - Kubernetes readiness probe (TEAM-104)
//!
//! Created by: TEAM-026
//! Modified by: TEAM-104 (added Kubernetes-compatible endpoints)

use axum::{extract::State, http::StatusCode, Json};
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

/// TEAM-104: Liveness probe response
#[derive(Serialize)]
pub struct LivenessResponse {
    status: String,
}

/// TEAM-104: Readiness probe response
#[derive(Serialize)]
pub struct ReadinessResponse {
    status: String,
    workers_total: usize,
    workers_ready: usize,
}

/// TEAM-104: Handle GET /health/live
///
/// Kubernetes liveness probe - checks if the process is alive
/// Returns 200 OK if the service is running
pub async fn handle_liveness() -> Json<LivenessResponse> {
    debug!("Liveness probe requested");
    Json(LivenessResponse {
        status: "alive".to_string(),
    })
}

/// TEAM-104: Handle GET /health/ready
///
/// Kubernetes readiness probe - checks if the service is ready to accept traffic
/// Returns 200 OK if at least one worker is ready, 503 otherwise
pub async fn handle_readiness(
    State(state): State<crate::http::routes::AppState>,
) -> Result<Json<ReadinessResponse>, StatusCode> {
    debug!("Readiness probe requested");

    let workers = state.registry.list().await;
    let workers_ready = workers
        .iter()
        .filter(|w| {
            matches!(
                w.state,
                crate::registry::WorkerState::Idle | crate::registry::WorkerState::Busy
            )
        })
        .count();

    if workers_ready > 0 {
        debug!(
            workers_total = workers.len(),
            workers_ready = workers_ready,
            "Service ready"
        );
        Ok(Json(ReadinessResponse {
            status: "ready".to_string(),
            workers_total: workers.len(),
            workers_ready,
        }))
    } else {
        // No ready workers - service not ready
        debug!(
            workers_total = workers.len(),
            "Service not ready - no workers available"
        );
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}
