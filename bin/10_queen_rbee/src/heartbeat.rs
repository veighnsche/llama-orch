//! Queen-rbee heartbeat handling
//!
//! TEAM-164: Binary-specific heartbeat logic for queen-rbee
//!
//! **What lives here:**
//! - Queen receives heartbeats from hives
//! - Triggers device detection on first heartbeat
//! - Updates hive catalog with heartbeat data
//!
//! **What lives in shared crate:**
//! - Heartbeat types (HiveHeartbeatPayload, etc.)
//! - Heartbeat traits (DeviceDetector, etc.)
//! - Common heartbeat logic

use anyhow::Result;
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::HiveCatalog;
use rbee_heartbeat::traits::{DeviceDetector};
use rbee_heartbeat::HiveHeartbeatPayload;
use std::sync::Arc;

const ACTOR_QUEEN_HEARTBEAT: &str = "👑 queen-heartbeat";
const ACTION_RECEIVE: &str = "receive_hive_heartbeat";
const ACTION_FIRST_HEARTBEAT: &str = "first_heartbeat";
const ACTION_DEVICE_DETECTION: &str = "device_detection";

/// Response from heartbeat processing
#[derive(Debug, Clone)]
pub struct HeartbeatAcknowledgement {
    pub status: String,
    pub message: String,
}

/// Handle hive heartbeat
///
/// **Flow:**
/// 1. Receive heartbeat from hive
/// 2. Check if this is first heartbeat (last_heartbeat_ms is None)
/// 3. If first heartbeat → trigger device detection
/// 4. Update catalog with heartbeat timestamp
/// 5. Return acknowledgement
///
/// This is the callback mechanism - hive sends heartbeat when ready!
pub async fn handle_hive_heartbeat<D: DeviceDetector>(
    catalog: Arc<HiveCatalog>,
    payload: HiveHeartbeatPayload,
    device_detector: Arc<D>,
) -> Result<HeartbeatAcknowledgement> {
    // Commented out due to noise.
    // Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_RECEIVE, &payload.hive_id)
    //     .human(format!("Received heartbeat from hive {}", payload.hive_id))
    //     .emit();

    // Get hive from catalog
    let hive = catalog.get_hive(&payload.hive_id).await?
        .ok_or_else(|| anyhow::anyhow!("Hive {} not found", payload.hive_id))?;

    // Check if this is first heartbeat
    let is_first_heartbeat = hive.last_heartbeat_ms.is_none();

    if is_first_heartbeat {
        Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_FIRST_HEARTBEAT, &payload.hive_id)
            .human(format!("First heartbeat from hive {} - triggering device detection", payload.hive_id))
            .emit();

        // Trigger device detection
        let hive_url = format!("http://{}:{}", hive.host, hive.port);
        
        Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_DEVICE_DETECTION, &hive_url)
            .human(format!("Detecting devices at {}", hive_url))
            .emit();

        match device_detector.detect_devices(&hive_url).await {
            Ok(devices) => {
                // Convert DeviceResponse to DeviceCapabilities
                let device_caps = queen_rbee_hive_catalog::DeviceCapabilities {
                    cpu: Some(queen_rbee_hive_catalog::CpuDevice {
                        cores: devices.cpu.cores,
                        ram_gb: devices.cpu.ram_gb,
                    }),
                    gpus: devices.gpus.into_iter().enumerate().map(|(idx, gpu)| queen_rbee_hive_catalog::GpuDevice {
                        index: idx as u32,
                        name: gpu.name,
                        vram_gb: gpu.vram_gb,
                        backend: queen_rbee_hive_catalog::DeviceBackend::Cuda, // TODO: Get from device response
                    }).collect(),
                };
                
                // Update catalog with devices
                catalog.update_devices(&payload.hive_id, device_caps).await?;
                
                Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_DEVICE_DETECTION, "success")
                    .human(format!("Device detection complete for hive {}", payload.hive_id))
                    .emit();
            }
            Err(e) => {
                Narration::new(ACTOR_QUEEN_HEARTBEAT, ACTION_DEVICE_DETECTION, "failed")
                    .human(format!("Device detection failed for hive {}: {}", payload.hive_id, e))
                    .emit();
                // Don't fail the heartbeat if device detection fails
            }
        }

        // Update hive status to Online
        catalog.update_hive_status(&payload.hive_id, queen_rbee_hive_catalog::HiveStatus::Online).await?;
    }

    // Update heartbeat timestamp
    let timestamp_ms = chrono::DateTime::parse_from_rfc3339(&payload.timestamp)
        .map(|dt| dt.timestamp_millis())
        .unwrap_or_else(|_| chrono::Utc::now().timestamp_millis());
    
    catalog.update_heartbeat(&payload.hive_id, timestamp_ms).await?;

    Ok(HeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Heartbeat received from {}", payload.hive_id),
    })
}
