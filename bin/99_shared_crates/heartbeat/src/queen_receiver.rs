//! Queen heartbeat receiver - handles hive heartbeats with device detection
//!
//! Created by: TEAM-159
//! Consolidated from: queen-rbee/src/http/heartbeat.rs
//!
//! This module provides the logic for queen to receive heartbeats from hives
//! and trigger device detection on first heartbeat.

use crate::traits::{
    CatalogError, CpuDevice, DeviceBackend, DeviceCapabilities, DeviceDetector, DeviceResponse,
    GpuDevice, HiveCatalog, HiveStatus,
};
use crate::types::HiveHeartbeatPayload;
use observability_narration_core::Narration;
use serde::Serialize;
use std::sync::Arc;

const ACTOR_QUEEN_HEARTBEAT: &str = "👑 queen-heartbeat";
const ACTION_HEARTBEAT: &str = "heartbeat";
const ACTION_DEVICE_DETECTION: &str = "device_detection";
const ACTION_ERROR: &str = "error";

/// Heartbeat acknowledgement response
#[derive(Debug, Serialize)]
pub struct HeartbeatAcknowledgement {
    /// Whether heartbeat was acknowledged
    pub acknowledged: bool,
}

impl HeartbeatAcknowledgement {
    /// Create a success acknowledgement
    pub fn success() -> Self {
        Self { acknowledged: true }
    }
}

/// Heartbeat error
#[derive(Debug, thiserror::Error)]
pub enum HeartbeatError {
    /// Hive not found in catalog
    #[error("Hive not found: {0}")]
    HiveNotFound(String),

    /// Catalog error
    #[error("Catalog error: {0}")]
    Catalog(#[from] CatalogError),

    /// Device detection failed
    #[error("Device detection failed: {0}")]
    DeviceDetection(String),

    /// Other error
    #[error("{0}")]
    Other(String),
}

/// Handle hive heartbeat
///
/// Receives heartbeat from hive and triggers device detection on first heartbeat.
///
/// # Arguments
/// * `catalog` - Hive catalog implementation
/// * `payload` - Heartbeat payload from hive
/// * `device_detector` - Device detector implementation
///
/// # Returns
/// * `Ok(HeartbeatAcknowledgement)` - Heartbeat acknowledged
/// * `Err(HeartbeatError)` - Error processing heartbeat
///
/// # Flow
/// 1. Parse timestamp and update heartbeat in catalog
/// 2. Get hive record from catalog
/// 3. If status is Unknown (first heartbeat):
///    a. Emit narration: "First heartbeat from {hive_id}"
///    b. Request device detection from hive
///    c. Convert device response to DeviceCapabilities
///    d. Store devices in catalog
///    e. Update hive status to Online
///    f. Emit narration: "Hive {hive_id} is now online"
/// 4. Return acknowledgement
pub async fn handle_hive_heartbeat<C, D>(
    catalog: Arc<C>,
    payload: HiveHeartbeatPayload,
    device_detector: Arc<D>,
) -> Result<HeartbeatAcknowledgement, HeartbeatError>
where
    C: HiveCatalog,
    D: DeviceDetector,
{
    // Parse timestamp to milliseconds for catalog
    let timestamp_ms = chrono::DateTime::parse_from_rfc3339(&payload.timestamp)
        .map(|dt| dt.timestamp_millis())
        .unwrap_or_else(|_| chrono::Utc::now().timestamp_millis());

    // Update heartbeat in catalog
    catalog
        .update_heartbeat(&payload.hive_id, timestamp_ms)
        .await?;

    // Check if this is first heartbeat (ONLY narrate on first heartbeat)
    let hive = catalog
        .get_hive(&payload.hive_id)
        .await?
        .ok_or_else(|| HeartbeatError::HiveNotFound(payload.hive_id.clone()))?;

    // If first heartbeat (status is Unknown), trigger device detection
    if matches!(hive.status, HiveStatus::Unknown) {
        Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_HEARTBEAT, &payload.hive_id)
            .human(format!(
                "First heartbeat from {}. Checking capabilities...",
                payload.hive_id
            ))
            .emit();

        // Request device detection from hive
        let hive_url = format!("http://{}:{}", hive.host, hive.port);

        Narration::new(
            ACTOR_QUEEN_HEARTBEAT,
            ACTION_DEVICE_DETECTION,
            &payload.hive_id,
        )
        .human(format!(
            "Unknown capabilities of beehive {}. Asking the beehive to detect devices",
            payload.hive_id
        ))
        .emit();

        let devices = device_detector
            .detect_devices(&hive_url)
            .await
            .map_err(|e| {
                Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_ERROR, &payload.hive_id)
                    .human(format!("Failed to request device detection: {}", e))
                    .error_kind("device_detection_failed")
                    .emit();
                HeartbeatError::DeviceDetection(e.to_string())
            })?;

        // Convert DeviceResponse to DeviceCapabilities
        let device_caps = convert_device_response(devices.clone());

        // Store devices in hive catalog
        catalog
            .update_devices(&payload.hive_id, device_caps)
            .await
            .map_err(|e| {
                Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_ERROR, &payload.hive_id)
                    .human(format!("Failed to store device capabilities: {}", e))
                    .error_kind("device_storage_failed")
                    .emit();
                e
            })?;

        // Build device summary for narration
        let gpu_summary = if devices.gpus.is_empty() {
            "no GPUs".to_string()
        } else {
            devices
                .gpus
                .iter()
                .map(|gpu| format!("{} {} ({}GB)", gpu.id, gpu.name, gpu.vram_gb))
                .collect::<Vec<_>>()
                .join(", ")
        };

        Narration::new(
            ACTOR_QUEEN_HEARTBEAT,
            ACTION_DEVICE_DETECTION,
            &payload.hive_id,
        )
        .human(format!(
            "The beehive {} has cpu ({} cores, {}GB RAM), {}, model catalog has {} models, {} workers available",
            payload.hive_id,
            devices.cpu.cores,
            devices.cpu.ram_gb,
            gpu_summary,
            devices.models,
            devices.workers
        ))
        .emit();

        // Update hive status to Online
        catalog
            .update_hive_status(&payload.hive_id, HiveStatus::Online)
            .await
            .map_err(|e| {
                Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_ERROR, &payload.hive_id)
                    .human(format!("Failed to update hive status: {}", e))
                    .error_kind("status_update_failed")
                    .emit();
                e
            })?;

        Narration::new(
            ACTOR_QUEEN_HEARTBEAT,
            ACTION_DEVICE_DETECTION,
            &payload.hive_id,
        )
        .human(format!("Hive {} is now online", payload.hive_id))
        .emit();
    }

    Ok(HeartbeatAcknowledgement::success())
}

/// Convert DeviceResponse to DeviceCapabilities
fn convert_device_response(response: DeviceResponse) -> DeviceCapabilities {
    let mut caps = DeviceCapabilities {
        cpu: Some(CpuDevice {
            cores: response.cpu.cores,
            ram_gb: response.cpu.ram_gb,
        }),
        gpus: vec![],
    };

    // Add GPUs - detect backend from platform
    // If we have GPUs, determine if they're CUDA or Metal
    // For now, assume CUDA on Linux/Windows, Metal on macOS
    for gpu in &response.gpus {
        let backend = if cfg!(target_os = "macos") {
            DeviceBackend::Metal
        } else {
            DeviceBackend::Cuda
        };

        caps.gpus.push(GpuDevice {
            index: gpu.id.trim_start_matches("gpu").parse().unwrap_or(0),
            name: gpu.name.clone(),
            vram_gb: gpu.vram_gb,
            backend,
        });
    }

    caps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{CpuInfo, DetectionError, GpuInfo, HiveRecord};
    use crate::types::HiveHeartbeatPayload;
    use async_trait::async_trait;
    use std::sync::Mutex;

    // Mock catalog for testing
    struct MockCatalog {
        hives: Mutex<Vec<HiveRecord>>,
    }

    impl MockCatalog {
        fn new() -> Self {
            Self {
                hives: Mutex::new(vec![]),
            }
        }

        fn add_hive(&self, hive: HiveRecord) {
            self.hives.lock().unwrap().push(hive);
        }
    }

    #[async_trait]
    impl HiveCatalog for MockCatalog {
        async fn update_heartbeat(
            &self,
            hive_id: &str,
            timestamp_ms: i64,
        ) -> Result<(), CatalogError> {
            let mut hives = self.hives.lock().unwrap();
            if let Some(hive) = hives.iter_mut().find(|h| h.id == hive_id) {
                hive.last_heartbeat_ms = Some(timestamp_ms);
                Ok(())
            } else {
                Err(CatalogError::NotFound(hive_id.to_string()))
            }
        }

        async fn get_hive(&self, hive_id: &str) -> Result<Option<HiveRecord>, CatalogError> {
            let hives = self.hives.lock().unwrap();
            Ok(hives.iter().find(|h| h.id == hive_id).cloned())
        }

        async fn update_devices(
            &self,
            hive_id: &str,
            devices: DeviceCapabilities,
        ) -> Result<(), CatalogError> {
            let mut hives = self.hives.lock().unwrap();
            if let Some(hive) = hives.iter_mut().find(|h| h.id == hive_id) {
                hive.devices = Some(devices);
                Ok(())
            } else {
                Err(CatalogError::NotFound(hive_id.to_string()))
            }
        }

        async fn update_hive_status(
            &self,
            hive_id: &str,
            status: HiveStatus,
        ) -> Result<(), CatalogError> {
            let mut hives = self.hives.lock().unwrap();
            if let Some(hive) = hives.iter_mut().find(|h| h.id == hive_id) {
                hive.status = status;
                Ok(())
            } else {
                Err(CatalogError::NotFound(hive_id.to_string()))
            }
        }
    }

    // Mock device detector for testing
    struct MockDetector;

    #[async_trait]
    impl DeviceDetector for MockDetector {
        async fn detect_devices(&self, _hive_url: &str) -> Result<DeviceResponse, DetectionError> {
            Ok(DeviceResponse {
                cpu: CpuInfo {
                    cores: 8,
                    ram_gb: 32,
                },
                gpus: vec![GpuInfo {
                    id: "gpu0".to_string(),
                    name: "RTX 3060".to_string(),
                    vram_gb: 12,
                }],
                models: 0,
                workers: 0,
            })
        }
    }

    #[tokio::test]
    async fn test_handle_hive_heartbeat_first_time() {
        let catalog = Arc::new(MockCatalog::new());
        catalog.add_hive(HiveRecord {
            id: "test-hive".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8600,
            status: HiveStatus::Unknown,
            last_heartbeat_ms: None,
            devices: None,
        });

        let detector = Arc::new(MockDetector);

        let payload = HiveHeartbeatPayload {
            hive_id: "test-hive".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            workers: vec![],
        };

        let result = handle_hive_heartbeat(catalog.clone(), payload, detector).await;
        assert!(result.is_ok());

        // Verify hive status was updated
        let hive = catalog.get_hive("test-hive").await.unwrap().unwrap();
        assert_eq!(hive.status, HiveStatus::Online);
        assert!(hive.devices.is_some());
    }

    #[tokio::test]
    async fn test_handle_hive_heartbeat_subsequent() {
        let catalog = Arc::new(MockCatalog::new());
        catalog.add_hive(HiveRecord {
            id: "test-hive".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8600,
            status: HiveStatus::Online,
            last_heartbeat_ms: None,
            devices: None,
        });

        let detector = Arc::new(MockDetector);

        let payload = HiveHeartbeatPayload {
            hive_id: "test-hive".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            workers: vec![],
        };

        let result = handle_hive_heartbeat(catalog.clone(), payload, detector).await;
        assert!(result.is_ok());

        // Verify heartbeat was updated but no device detection
        let hive = catalog.get_hive("test-hive").await.unwrap().unwrap();
        assert!(hive.last_heartbeat_ms.is_some());
    }
}
