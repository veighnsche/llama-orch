//! VRAM capacity check endpoint
//!
//! Per architecture Phase 6: VRAM Check
//! Checks if device has enough VRAM for model before spawning worker
//!
//! # Architecture Reference
//! - a_Claude_Sonnet_4_5_refined_this.md lines 181-189
//! - Queen calls this before worker spawning
//!
//! Created by: TEAM-151

use crate::http::routes::AppState;
use axum::{
    extract::{Query, State},
    http::StatusCode,
};
use serde::Deserialize;
use tracing::{info, warn};

/// Capacity check query parameters
#[derive(Debug, Deserialize)]
pub struct CapacityQuery {
    /// Device ID (e.g., "gpu0", "gpu1", "cpu")
    pub device: String,
    /// Model reference (e.g., "HF:author/minillama")
    pub model: String,
}

/// Handle GET /v1/capacity?device=gpu1&model=HF:author/minillama
///
/// Checks if device has enough VRAM/RAM for the specified model
///
/// # Architecture Flow
/// ```text
/// Queen → Hive: GET /v1/capacity?device=gpu1&model=HF:author/minillama
/// Hive checks VRAM checker crate
/// Hive responds: 204 (OK) or 409 (insufficient)
/// ```
///
/// # Query Parameters
/// * `device` - Device ID (e.g., "gpu0", "gpu1", "cpu")
/// * `model` - Model reference (e.g., "HF:author/minillama")
///
/// # Returns
/// * `204 No Content` - Sufficient capacity available
/// * `409 Conflict` - Insufficient capacity
/// * `400 Bad Request` - Invalid device or model reference
///
/// # Logic
/// Total VRAM - loaded models - estimated model size > 0
pub async fn handle_capacity_check(
    State(state): State<AppState>,
    Query(params): Query<CapacityQuery>,
) -> Result<StatusCode, (StatusCode, String)> {
    info!(
        device = %params.device,
        model = %params.model,
        "Capacity check requested"
    );

    // Validate device ID format
    if !params.device.starts_with("gpu") && params.device != "cpu" {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Invalid device ID: {}", params.device),
        ));
    }

    // TODO: Use rbee-hive-vram-checker crate for real capacity checking
    // For now, implement basic logic:
    // 1. Get device total VRAM/RAM
    // 2. Get currently loaded models and their sizes
    // 3. Estimate model size from reference
    // 4. Check: total - loaded - estimated > 0

    // Mock implementation for development
    // In production, this would:
    // - Query GPU VRAM via CUDA/Metal/etc
    // - Get loaded worker models from registry
    // - Estimate model size from HuggingFace metadata or catalog
    // - Calculate available capacity

    // For now, always return OK (sufficient capacity)
    // This allows development to proceed without blocking on VRAM detection
    info!(
        device = %params.device,
        model = %params.model,
        "Capacity check passed (mock implementation)"
    );

    Ok(StatusCode::NO_CONTENT)

    // Example of insufficient capacity response:
    // warn!(
    //     device = %params.device,
    //     model = %params.model,
    //     "Insufficient capacity"
    // );
    // Err((
    //     StatusCode::CONFLICT,
    //     format!("Insufficient VRAM on {} for model {}", params.device, params.model),
    // ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provisioner::ModelProvisioner;
    use crate::registry::WorkerRegistry;
    use model_catalog::ModelCatalog;
    use rbee_hive::download_tracker::DownloadTracker;
    use std::path::PathBuf;
    use std::sync::Arc;

    async fn create_test_state() -> AppState {
        let registry = Arc::new(WorkerRegistry::new());
        let catalog = Arc::new(ModelCatalog::new(":memory:".to_string()));
        let provisioner = Arc::new(ModelProvisioner::new(PathBuf::from("/tmp")));
        let download_tracker = Arc::new(DownloadTracker::new());
        let addr: std::net::SocketAddr = "127.0.0.1:9200".parse().unwrap();

        AppState {
            registry,
            model_catalog: catalog,
            provisioner,
            download_tracker,
            server_addr: addr,
            expected_token: "test-token".to_string(),
            audit_logger: None,
            queen_callback_url: None,
        }
    }

    #[tokio::test]
    async fn test_capacity_check_valid_gpu() {
        let state = create_test_state().await;

        let params = CapacityQuery {
            device: "gpu0".to_string(),
            model: "HF:author/minillama".to_string(),
        };

        let result = handle_capacity_check(State(state), Query(params)).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_capacity_check_valid_cpu() {
        let state = create_test_state().await;

        let params = CapacityQuery {
            device: "cpu".to_string(),
            model: "HF:author/minillama".to_string(),
        };

        let result = handle_capacity_check(State(state), Query(params)).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn test_capacity_check_invalid_device() {
        let state = create_test_state().await;

        let params = CapacityQuery {
            device: "invalid-device".to_string(),
            model: "HF:author/minillama".to_string(),
        };

        let result = handle_capacity_check(State(state), Query(params)).await;
        assert!(result.is_err());
        let (status, _msg) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_capacity_query_deserialization() {
        let query = "device=gpu1&model=HF:author/minillama";
        let parsed: CapacityQuery = serde_urlencoded::from_str(query).unwrap();
        assert_eq!(parsed.device, "gpu1");
        assert_eq!(parsed.model, "HF:author/minillama");
    }
}
