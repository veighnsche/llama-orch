//! Hive telemetry handling
//!
//! TEAM-361: Sends worker telemetry to Queen every 1s
//! TEAM-363: Cleaned up per RULE ZERO

use anyhow::Result;
use hive_contract::{HiveHeartbeat, HiveInfo};

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

/// Start periodic telemetry task
///
/// TEAM-361: Sends worker telemetry to Queen every 1s
pub fn start_heartbeat_task(hive_info: HiveInfo, queen_url: String) -> tokio::task::JoinHandle<()> {
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
    })
}

// TEAM-361: Worker telemetry collection implemented in rbee-hive-monitor crate
