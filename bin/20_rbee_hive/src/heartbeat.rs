//! Hive telemetry handling
//!
//! TEAM-361: Sends worker telemetry to Queen every 1s
//! TEAM-363: Cleaned up per RULE ZERO
//! TEAM-365: Added exponential backoff discovery for bidirectional handshake
//! TEAM-366: Added edge case guards for handshake reliability
//! TEAM-367: Added capabilities support for Queen restart detection

use anyhow::Result;
use hive_contract::heartbeat::HiveDevice; // TEAM-372: Import from heartbeat module
use hive_contract::HiveInfo;
use observability_narration_core::n; // TEAM-365: Narration for discovery
use std::sync::atomic::{AtomicBool, Ordering}; // TEAM-366: Circuit breaker
use std::sync::Arc; // TEAM-366: Shared state

// TEAM-361: Worker telemetry collection

/// TEAM-367: Detect hive capabilities (devices)
/// 
/// Returns Vec<HiveDevice> with all GPUs + CPU-0
#[allow(dead_code)]
fn detect_capabilities() -> Vec<HiveDevice> {
    let mut devices = Vec::new();
    
    // Detect GPUs
    let gpu_info = rbee_hive_device_detection::detect_gpus();
    for gpu in &gpu_info.devices {
        devices.push(HiveDevice {
            id: format!("GPU-{}", gpu.index),
            name: gpu.name.clone(),
            device_type: "gpu".to_string(),
            vram_gb: Some(gpu.vram_total_gb() as u32),
            compute_capability: Some(format!("{}.{}", gpu.compute_capability.0, gpu.compute_capability.1)),
        });
    }
    
    // Add CPU device (always available)
    let cpu_cores = rbee_hive_device_detection::get_cpu_cores();
    let system_ram_gb = rbee_hive_device_detection::get_system_ram_gb();
    devices.push(HiveDevice {
        id: "CPU-0".to_string(),
        name: format!("CPU ({} cores)", cpu_cores),
        device_type: "cpu".to_string(),
        vram_gb: Some(system_ram_gb),
        compute_capability: None,
    });
    
    devices
}

// TEAM-374: DELETED send_heartbeat_to_queen() - replaced by SSE stream
// Old POST-based continuous telemetry is deprecated.
// Hive now broadcasts via SSE GET /v1/heartbeats/stream

/// Send ready callback to queen (one-time discovery)
///
/// TEAM-374: Replaces send_heartbeat_to_queen() for discovery.
/// This is a ONE-TIME callback that tells Queen "I'm ready, subscribe to my SSE stream".
/// After this, Queen subscribes to GET /v1/heartbeats/stream for continuous telemetry.
async fn send_ready_callback_to_queen(
    hive_info: &HiveInfo,
    queen_url: &str,
) -> Result<()> {
    tracing::debug!("Sending ready callback to queen at {}", queen_url);

    #[derive(serde::Serialize)]
    struct HiveReadyCallback {
        hive_id: String,
        hive_url: String,
    }

    let callback = HiveReadyCallback {
        hive_id: hive_info.id.clone(),
        hive_url: format!("http://{}:{}", hive_info.hostname, hive_info.port),
    };

    // TEAM-374: 5-second timeout
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    
    let response = client
        .post(format!("{}/v1/hive/ready", queen_url))
        .json(&callback)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_else(|_| "unknown error".to_string());
        anyhow::bail!("Ready callback failed with status {}: {}", status, body);
    }

    tracing::debug!("Ready callback sent successfully");
    Ok(())
}

/// Start heartbeat task with discovery phase
///
/// TEAM-365: Implements exponential backoff discovery per HEARTBEAT_ARCHITECTURE.md
/// TEAM-361: Sends worker telemetry to Queen every 1s after discovery
/// TEAM-366: Added validation and edge case guards
/// TEAM-367: Made queen_url optional - hive can run standalone
pub fn start_heartbeat_task(
    hive_info: HiveInfo,
    queen_url: Option<String>,
    running_flag: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    // TEAM-367: FIX EDGE CASE #2 - queen_url is optional (standalone hive)
    let queen_url = match queen_url {
        None => {
            n!("heartbeat_disabled", "ℹ️  No queen_url configured, heartbeat disabled (standalone mode)");
            return tokio::spawn(async {});
        }
        Some(url) if url.is_empty() => {
            n!("heartbeat_disabled", "ℹ️  Empty queen_url, heartbeat disabled (standalone mode)");
            return tokio::spawn(async {});
        }
        Some(url) => {
            // Validate URL format
            if let Err(e) = url::Url::parse(&url) {
                n!("heartbeat_invalid_url", "❌ Invalid queen_url '{}': {}. Heartbeat disabled.", url, e);
                return tokio::spawn(async {});
            }
            url
        }
    };
    
    tokio::spawn(async move {
        // TEAM-366: EDGE CASE #4 - Ensure flag is cleared if task crashes
        let _guard = HeartbeatGuard::new(running_flag);
        
        // TEAM-365: Start with discovery phase (exponential backoff)
        start_discovery_with_backoff(hive_info, queen_url).await;
    })
}

/// TEAM-366: EDGE CASE #4 - Guard to clear heartbeat_running flag on task crash
struct HeartbeatGuard {
    flag: Arc<AtomicBool>,
}

impl HeartbeatGuard {
    fn new(flag: Arc<AtomicBool>) -> Self {
        Self { flag }
    }
}

impl Drop for HeartbeatGuard {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::SeqCst);
        n!("heartbeat_stopped", "⏹️  Heartbeat task stopped (flag cleared)");
    }
}

/// TEAM-365: Discovery phase with exponential backoff
///
/// Implements Scenario 2 from HEARTBEAT_ARCHITECTURE.md:
/// - 5 attempts with exponential backoff: 0s, 2s, 4s, 8s, 16s
/// - On first 200 OK: transition to normal telemetry
/// - After 5 failures: stop and wait for Queen discovery via /capabilities
async fn start_discovery_with_backoff(hive_info: HiveInfo, queen_url: String) {
    let delays = [0, 2, 4, 8, 16];  // TEAM-365: Exponential backoff in seconds
    
    n!("discovery_start", "🔍 Starting discovery with exponential backoff");
    
    for (attempt, delay) in delays.iter().enumerate() {
        if *delay > 0 {
            tokio::time::sleep(tokio::time::Duration::from_secs(*delay)).await;
        }
        
        n!("discovery_attempt", "🔍 Discovery attempt {} (delay: {}s)", attempt + 1, delay);
        
        // TEAM-374: Send ready callback (one-time, no capabilities needed)
        // Queen will subscribe to our SSE stream after receiving this
        match send_ready_callback_to_queen(&hive_info, &queen_url).await {
            Ok(_) => {
                n!("discovery_success", "✅ Discovery successful! Queen will subscribe to our SSE stream");
                // TEAM-374: No telemetry task needed - SSE broadcaster handles it
                return;
            }
            Err(e) => {
                n!("discovery_failed", "❌ Discovery attempt {} failed: {}", attempt + 1, e);
            }
        }
    }
    
    // TEAM-365: All 5 attempts failed
    n!("discovery_stopped", "⏸️  All discovery attempts failed. Waiting for Queen to discover us via /capabilities");
}

// TEAM-374: DELETED start_normal_telemetry_task() - replaced by SSE broadcaster
// Old POST-based continuous telemetry (1s interval) is deprecated.
// Hive now broadcasts telemetry via SSE stream (heartbeat_stream.rs)
// TEAM-361: Worker telemetry collection implemented in rbee-hive-monitor crate
