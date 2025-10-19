//! Health check endpoints
//!
//! Per test-001-mvp.md Phase 2: Pool Preflight
//! Returns version and status information
//!
//! # Endpoints
//! - `GET /v1/health` - Basic health check
//! - `GET /health/live` - Kubernetes liveness probe (TEAM-104) - ⚠️ REMOVED BY TEAM-113
//! - `GET /health/ready` - Kubernetes readiness probe (TEAM-104) - ⚠️ REMOVED BY TEAM-113
//!
//! # ⚠️ WARNING: NO KUBERNETES PATTERNS
//!
//! rbee IS the orchestrator, not an app running IN Kubernetes.
//! We don't need Kubernetes-style liveness/readiness/startup probes.
//! The simple /v1/health endpoint is sufficient.
//!
//! See: .docs/components/TEAM_113_EXIT_INTERVIEW.md for why Kubernetes drift kills products.
//!
//! Created by: TEAM-026
//! Modified by: TEAM-104 (added Kubernetes-compatible endpoints)
//! Modified by: TEAM-113 (removed Kubernetes drift, kept comments for history)

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

// ============================================================================
// ⚠️ KUBERNETES DRIFT REMOVED BY TEAM-113
// ============================================================================
//
// TEAM-104 added /health/live and /health/ready endpoints labeled as
// "Kubernetes liveness/readiness probes". This was KUBERNETES DRIFT.
//
// WHY THIS WAS REMOVED:
// - rbee IS the orchestrator, not an app running IN Kubernetes
// - We don't need Kubernetes-style probes (liveness, readiness, startup)
// - The simple /v1/health endpoint is sufficient
// - Kubernetes patterns lead to complexity creep and product death
//
// IF YOU'RE THINKING OF ADDING KUBERNETES PATTERNS:
// 1. Read: .docs/components/TEAM_113_EXIT_INTERVIEW.md
// 2. Read: .docs/components/KUBERNETES_DRIFT_FOUND.md
// 3. Ask yourself: "Does rbee run IN Kubernetes?" (Answer: NO)
// 4. Ask yourself: "Is rbee the orchestrator?" (Answer: YES)
// 5. Don't add Kubernetes patterns
//
// rbee is the SIMPLE alternative to Kubernetes. Keep it that way.
//
// ============================================================================
