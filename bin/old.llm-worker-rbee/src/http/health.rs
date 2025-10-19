// TEAM-110: Audited 2025-10-18 - ✅ CLEAN - Standard health check endpoint with VRAM tracking
//
//! GET /health endpoint - Health check
//!
//! Returns a simple health status for the worker.
//!
//! Modified by: TEAM-017 (updated to use Mutex-wrapped backend)
//!
//! # Spec References
//! - M0-W-1320: Health endpoint returns 200 OK with {"status": "healthy"}

use crate::http::backend::InferenceBackend;
use crate::narration::{ACTION_HEALTH_CHECK, ACTOR_HTTP_SERVER};
use axum::{extract::State, Json};
use observability_narration_core::{narrate, NarrationFields};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::debug;

/// Health check response
///
/// Per spec M0-W-1320, this endpoint returns a simple status indicator.
#[derive(Serialize)]
pub struct HealthResponse {
    /// Health status: "healthy" or "unhealthy"
    status: String,
    /// VRAM usage in bytes
    vram_bytes: u64,
}

/// Handle GET /health
///
/// Returns 200 OK with `{"status": "healthy"}` if the worker is operational.
/// TEAM-017: Updated to use Mutex-wrapped backend
pub async fn handle_health<B: InferenceBackend>(
    State(backend): State<Arc<Mutex<B>>>,
) -> Json<HealthResponse> {
    debug!("Health check requested");

    let backend = backend.lock().await;
    let status = if backend.is_healthy() { "healthy" } else { "unhealthy" };
    let vram_bytes = backend.vram_usage();

    narrate(NarrationFields {
        actor: ACTOR_HTTP_SERVER,
        action: ACTION_HEALTH_CHECK,
        target: status.to_string(),
        human: format!("Health check: {} (VRAM: {} MB)", status, vram_bytes / 1_000_000),
        cute: Some(format!("Feeling {status}! 💪")),
        ..Default::default()
    });

    Json(HealthResponse { status: status.to_string(), vram_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_structure() {
        let response = HealthResponse { status: "healthy".to_string(), vram_bytes: 8_000_000_000 };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\""));
        assert!(json.contains("\"healthy\""));
        assert!(json.contains("\"vram_bytes\""));
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse { status: "healthy".to_string(), vram_bytes: 16_000_000_000 };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "healthy");
        assert_eq!(json["vram_bytes"], 16_000_000_000u64);
    }

    #[test]
    fn test_health_response_unhealthy() {
        let response = HealthResponse { status: "unhealthy".to_string(), vram_bytes: 0 };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "unhealthy");
        assert_eq!(json["vram_bytes"], 0);
    }

    #[test]
    fn test_health_response_various_vram_sizes() {
        let sizes = vec![
            0u64,
            1_000_000_000,  // 1GB
            8_000_000_000,  // 8GB
            16_000_000_000, // 16GB
            80_000_000_000, // 80GB
        ];

        for vram_bytes in sizes {
            let response = HealthResponse { status: "healthy".to_string(), vram_bytes };

            let json = serde_json::to_value(&response).unwrap();
            assert_eq!(json["vram_bytes"], vram_bytes);
        }
    }
}

// ---
// Built by Foundation-Alpha 🏗️
