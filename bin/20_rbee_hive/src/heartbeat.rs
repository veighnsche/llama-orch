//! Hive telemetry handling
//!
//! TEAM-361: Sends worker telemetry to Queen every 1s
//! TEAM-363: Cleaned up per RULE ZERO
//! TEAM-365: Added exponential backoff discovery for bidirectional handshake
//! TEAM-366: Added edge case guards for handshake reliability
//! TEAM-367: Added capabilities support for Queen restart detection

use anyhow::Result;
use hive_contract::{HiveDevice, HiveHeartbeat, HiveInfo};
use observability_narration_core::n; // TEAM-365: Narration for discovery
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering}; // TEAM-366: Circuit breaker
use std::sync::Arc; // TEAM-366: Shared state

// TEAM-361: Worker telemetry collection
use rbee_hive_monitor;

/// TEAM-367: Detect hive capabilities (devices)
/// 
/// Returns Vec<HiveDevice> with all GPUs + CPU-0
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

/// Send telemetry to queen
///
/// TEAM-361: Sends HiveHeartbeat with worker telemetry every 1s
/// TEAM-367: Optionally includes capabilities during discovery/rediscovery
pub async fn send_heartbeat_to_queen(
    hive_info: &HiveInfo, 
    queen_url: &str,
    capabilities: Option<Vec<HiveDevice>>,
) -> Result<()> {
    tracing::debug!("Sending hive telemetry to queen at {}", queen_url);

    // TEAM-361: Collect worker telemetry from cgroup + GPU
    let workers = rbee_hive_monitor::collect_all_workers().await.unwrap_or_else(|e| {
        tracing::warn!("Failed to collect worker telemetry: {}", e);
        Vec::new()
    });

    tracing::trace!("Collected telemetry for {} workers", workers.len());

    // TEAM-367: Build heartbeat with optional capabilities
    let heartbeat = if let Some(caps) = capabilities {
        tracing::debug!("Including {} devices in heartbeat (discovery mode)", caps.len());
        HiveHeartbeat::with_capabilities(hive_info.clone(), workers, caps)
    } else {
        HiveHeartbeat::with_workers(hive_info.clone(), workers)
    };

    // TEAM-364: Add 5-second timeout to prevent hangs (Critical Issue #6)
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let response =
        client.post(format!("{}/v1/hive-heartbeat", queen_url)).json(&heartbeat).send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_else(|_| "unknown error".to_string());
        anyhow::bail!("Telemetry failed with status {}: {}", status, body);
    }

    tracing::trace!("Hive telemetry sent successfully");
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
        
        // TEAM-367: Send discovery heartbeat WITH capabilities
        let capabilities = detect_capabilities();
        n!("discovery_capabilities", "🔍 Detected {} device(s) to send", capabilities.len());
        
        match send_heartbeat_to_queen(&hive_info, &queen_url, Some(capabilities)).await {
            Ok(_) => {
                n!("discovery_success", "✅ Discovery successful! Starting normal telemetry");
                // TEAM-365: Start normal telemetry task
                start_normal_telemetry_task(hive_info, queen_url).await;
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

/// TEAM-365: Normal telemetry task (runs after discovery)
/// TEAM-366: Added circuit breaker for edge case #5
/// TEAM-367: Added Queen restart detection - triggers rediscovery on 400/404
///
/// Sends worker telemetry to Queen every 1s for real-time scheduling
async fn start_normal_telemetry_task(hive_info: HiveInfo, queen_url: String) {
    tokio::spawn(async move {
        // TEAM-361: Send telemetry every 1s for real-time scheduling
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
        
        // TEAM-366: EDGE CASE #5 - Circuit breaker to prevent log flooding
        let consecutive_failures = Arc::new(AtomicUsize::new(0));
        let max_failures = 10; // Stop logging after 10 consecutive failures

        loop {
            interval.tick().await;

            // TEAM-361: Collect and send worker telemetry (no capabilities)
            match send_heartbeat_to_queen(&hive_info, &queen_url, None).await {
                Ok(_) => {
                    // TEAM-366: Reset circuit breaker on success
                    let prev = consecutive_failures.swap(0, Ordering::SeqCst);
                    if prev >= max_failures {
                        n!("heartbeat_recovered", "✅ Heartbeat recovered after {} failures", prev);
                    }
                }
                Err(e) => {
                    let failures = consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
                    
                    // TEAM-367: EDGE CASE #4 - Detect Queen restart (400/404/connection refused)
                    // Queen restart means it lost all hive state, needs rediscovery
                    let error_str = e.to_string();
                    let is_queen_restart = error_str.contains("status 400") 
                        || error_str.contains("status 404")
                        || error_str.contains("connection refused")
                        || error_str.contains("Connection refused");
                    
                    if failures == 1 && is_queen_restart {
                        n!("queen_restart_detected", "⚠️  Queen restart detected! Starting rediscovery with capabilities...");
                        
                        // TEAM-367: Restart discovery - this task will exit, new one starts
                        start_discovery_with_backoff(hive_info.clone(), queen_url.clone()).await;
                        return; // Exit this task, discovery will spawn new telemetry task
                    }
                    
                    // TEAM-366: Only log first failure and every 60th failure after threshold
                    if failures == 1 {
                        tracing::warn!("Failed to send hive telemetry: {}", e);
                    } else if failures == max_failures {
                        tracing::error!(
                            "Heartbeat failing consistently ({} consecutive failures). \
                            Suppressing further logs. Queen may be down.",
                            failures
                        );
                    } else if failures > max_failures && failures % 60 == 0 {
                        tracing::warn!("Still failing: {} consecutive heartbeat failures", failures);
                    }
                }
            }
        }
    });
}

// TEAM-361: Worker telemetry collection implemented in rbee-hive-monitor crate
