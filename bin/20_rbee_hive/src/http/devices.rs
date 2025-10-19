//! Device detection endpoint
//!
//! Per architecture Phase 4: Device Detection
//! Returns CPU info, GPU list, model count, worker count
//!
//! # Architecture Reference
//! - a_Claude_Sonnet_4_5_refined_this.md lines 136-164
//! - Queen calls this on first heartbeat or stale capabilities
//!
//! Created by: TEAM-151

use crate::http::routes::AppState;
use axum::{extract::State, Json};
use serde::Serialize;
use tracing::info;

/// Device detection response
///
/// Per architecture Phase 4
#[derive(Debug, Serialize)]
pub struct DevicesResponse {
    /// CPU information
    pub cpu: CpuInfo,
    /// List of GPUs
    pub gpus: Vec<GpuInfo>,
    /// Number of models in catalog
    pub models: usize,
    /// Number of workers in registry
    pub workers: usize,
}

/// CPU information
#[derive(Debug, Serialize)]
pub struct CpuInfo {
    /// Number of CPU cores
    pub cores: u32,
    /// RAM in GB
    pub ram_gb: u32,
}

/// GPU information
#[derive(Debug, Serialize)]
pub struct GpuInfo {
    /// GPU ID (e.g., "gpu0", "gpu1")
    pub id: String,
    /// GPU name (e.g., "RTX 3090")
    pub name: String,
    /// VRAM in GB
    pub vram_gb: u32,
}

/// Handle GET /v1/devices
///
/// Returns device capabilities for this hive
///
/// # Architecture Flow
/// ```text
/// Queen → Hive: GET /v1/devices
/// Hive runs device-detection crate
/// Hive responds with: CPU + GPU list + model/worker counts
/// Queen updates hive-catalog + hive-registry
/// ```
///
/// # Returns
/// * `200 OK` with device information
///
/// # Example Response
/// ```json
/// {
///   "cpu": {"cores": 8, "ram_gb": 32},
///   "gpus": [
///     {"id": "gpu0", "name": "RTX 3060", "vram_gb": 12},
///     {"id": "gpu1", "name": "RTX 3090", "vram_gb": 24}
///   ],
///   "models": 0,
///   "workers": 0
/// }
/// ```
pub async fn handle_devices(State(state): State<AppState>) -> Json<DevicesResponse> {
    info!("Device detection requested");

    // TODO: Use rbee-hive-device-detection crate for real hardware detection
    // For now, return mock data for development
    
    // Get model count from catalog
    let models_count = state
        .model_catalog
        .list_models()
        .await
        .map(|models| models.len())
        .unwrap_or(0);

    // Get worker count from registry
    let workers = state.registry.list().await;
    let workers_count = workers.len();

    // Mock CPU info (TODO: Replace with real detection)
    let cpu = CpuInfo {
        cores: num_cpus::get() as u32,
        ram_gb: 32, // TODO: Get real RAM size
    };

    // Mock GPU info (TODO: Replace with real detection from device-detection crate)
    let gpus = vec![
        // GpuInfo {
        //     id: "gpu0".to_string(),
        //     name: "RTX 3060".to_string(),
        //     vram_gb: 12,
        // },
        // GpuInfo {
        //     id: "gpu1".to_string(),
        //     name: "RTX 3090".to_string(),
        //     vram_gb: 24,
        // },
    ];

    info!(
        cpu_cores = cpu.cores,
        ram_gb = cpu.ram_gb,
        gpus = gpus.len(),
        models = models_count,
        workers = workers_count,
        "Device detection complete"
    );

    Json(DevicesResponse {
        cpu,
        gpus,
        models: models_count,
        workers: workers_count,
    })
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

    #[tokio::test]
    async fn test_devices_response_structure() {
        let registry = Arc::new(WorkerRegistry::new());
        let catalog = Arc::new(ModelCatalog::new(":memory:".to_string()));
        let provisioner = Arc::new(ModelProvisioner::new(PathBuf::from("/tmp")));
        let download_tracker = Arc::new(DownloadTracker::new());
        let addr: std::net::SocketAddr = "127.0.0.1:9200".parse().unwrap();

        let state = AppState {
            registry,
            model_catalog: catalog,
            provisioner,
            download_tracker,
            server_addr: addr,
            expected_token: "test-token".to_string(),
            audit_logger: None,
            queen_callback_url: None,
        };

        let response = handle_devices(State(state)).await;

        // Verify response structure
        assert!(response.0.cpu.cores > 0);
        assert_eq!(response.0.models, 0);
        assert_eq!(response.0.workers, 0);
    }

    #[test]
    fn test_devices_response_serialization() {
        let response = DevicesResponse {
            cpu: CpuInfo {
                cores: 8,
                ram_gb: 32,
            },
            gpus: vec![GpuInfo {
                id: "gpu0".to_string(),
                name: "RTX 3090".to_string(),
                vram_gb: 24,
            }],
            models: 5,
            workers: 2,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["cpu"]["cores"], 8);
        assert_eq!(json["cpu"]["ram_gb"], 32);
        assert_eq!(json["gpus"][0]["id"], "gpu0");
        assert_eq!(json["models"], 5);
        assert_eq!(json["workers"], 2);
    }
}
