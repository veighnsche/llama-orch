//! GET /health endpoint - Health check
//!
//! Returns a simple health status for the worker.
//!
//! # Spec References
//! - M0-W-1320: Health endpoint returns 200 OK with {"status": "healthy"}

use crate::backend::InferenceBackend;
use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;
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
pub async fn handle_health<B: InferenceBackend>(
    State(backend): State<Arc<B>>,
) -> Json<HealthResponse> {
    debug!("Health check requested");

    let status = if backend.is_healthy() { "healthy" } else { "unhealthy" };
    let vram_bytes = backend.vram_usage();

    Json(HealthResponse {
        status: status.to_string(),
        vram_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_response_structure() {
        let response = HealthResponse { 
            status: "healthy".to_string(),
            vram_bytes: 8_000_000_000,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"status\""));
        assert!(json.contains("\"healthy\""));
        assert!(json.contains("\"vram_bytes\""));
    }

    #[test]
    fn test_health_response_serialization() {
        let response = HealthResponse { 
            status: "healthy".to_string(),
            vram_bytes: 16_000_000_000,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "healthy");
        assert_eq!(json["vram_bytes"], 16_000_000_000u64);
    }

    #[test]
    fn test_health_response_unhealthy() {
        let response = HealthResponse {
            status: "unhealthy".to_string(),
            vram_bytes: 0,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "unhealthy");
        assert_eq!(json["vram_bytes"], 0);
    }

    #[test]
    fn test_health_response_various_vram_sizes() {
        let sizes = vec![
            0u64,
            1_000_000_000,      // 1GB
            8_000_000_000,      // 8GB
            16_000_000_000,     // 16GB
            80_000_000_000,     // 80GB
        ];

        for vram_bytes in sizes {
            let response = HealthResponse {
                status: "healthy".to_string(),
                vram_bytes,
            };

            let json = serde_json::to_value(&response).unwrap();
            assert_eq!(json["vram_bytes"], vram_bytes);
        }
    }
}

// ---
// Built by Foundation-Alpha 🏗️
