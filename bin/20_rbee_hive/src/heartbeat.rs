//! Hive telemetry handling
//!
//! TEAM-361: Sends worker telemetry to Queen every 1s
//! TEAM-363: Cleaned up per RULE ZERO
//! TEAM-365: Added exponential backoff discovery for bidirectional handshake

use anyhow::Result;
use hive_contract::{HiveHeartbeat, HiveInfo};
use observability_narration_core::n; // TEAM-365: Narration for discovery

// TEAM-361: Worker telemetry collection
use rbee_hive_monitor;

/// Send telemetry to queen
///
/// TEAM-361: Sends HiveHeartbeat with worker telemetry every 1s
pub async fn send_heartbeat_to_queen(hive_info: &HiveInfo, queen_url: &str) -> Result<()> {
    tracing::debug!("Sending hive telemetry to queen at {}", queen_url);

    // TEAM-361: Collect worker telemetry from cgroup + GPU
    let workers = rbee_hive_monitor::collect_all_workers().await.unwrap_or_else(|e| {
        tracing::warn!("Failed to collect worker telemetry: {}", e);
        Vec::new()
    });

    tracing::trace!("Collected telemetry for {} workers", workers.len());

    // TEAM-361: Build heartbeat with worker telemetry
    let heartbeat = HiveHeartbeat::with_workers(hive_info.clone(), workers);

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
pub fn start_heartbeat_task(hive_info: HiveInfo, queen_url: String) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // TEAM-365: Start with discovery phase (exponential backoff)
        start_discovery_with_backoff(hive_info, queen_url).await;
    })
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
        
        // TEAM-365: Send discovery heartbeat (same format as normal)
        match send_heartbeat_to_queen(&hive_info, &queen_url).await {
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
///
/// Sends worker telemetry to Queen every 1s for real-time scheduling
async fn start_normal_telemetry_task(hive_info: HiveInfo, queen_url: String) {
    tokio::spawn(async move {
        // TEAM-361: Send telemetry every 1s for real-time scheduling
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));

        loop {
            interval.tick().await;

            // TEAM-361: Collect and send worker telemetry
            if let Err(e) = send_heartbeat_to_queen(&hive_info, &queen_url).await {
                tracing::warn!("Failed to send hive telemetry: {}", e);
            }
        }
    });
}

// TEAM-361: Worker telemetry collection implemented in rbee-hive-monitor crate
