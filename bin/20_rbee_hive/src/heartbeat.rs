//! Hive heartbeat handling
//!
//! TEAM-284: Hive sends heartbeats to queen
//!
//! **What lives here:**
//! - Hive sends heartbeats to queen
//! - Periodic heartbeat task
//! - Hive health status reporting

use anyhow::Result;
use hive_contract::{HiveHeartbeat, HiveInfo};

/// Send heartbeat to queen
///
/// TEAM-284: Hive sends heartbeats directly to queen
///
/// **Flow:**
/// 1. Build HiveHeartbeat with full HiveInfo
/// 2. Send POST /v1/hive-heartbeat to queen
/// 3. Return acknowledgement
///
/// This is called periodically (e.g., every 30s) to signal hive is alive
pub async fn send_heartbeat_to_queen(hive_info: &HiveInfo, queen_url: &str) -> Result<()> {
    tracing::debug!("Sending hive heartbeat to queen at {}", queen_url);

    // TEAM-285: Implemented HTTP POST to queen
    let heartbeat = HiveHeartbeat::new(hive_info.clone());

    let client = reqwest::Client::new();
    let response =
        client.post(format!("{}/v1/hive-heartbeat", queen_url)).json(&heartbeat).send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_else(|_| "unknown error".to_string());
        anyhow::bail!("Heartbeat failed with status {}: {}", status, body);
    }

    tracing::trace!("Hive heartbeat sent successfully");
    Ok(())
}

/// Start periodic heartbeat task
///
/// TEAM-284: Hive sends heartbeats to queen
///
/// **Flow:**
/// 1. Spawn tokio task
/// 2. Every 30 seconds, send heartbeat to queen with full HiveInfo
/// 3. Continue until task is cancelled
///
/// This runs in the background for the lifetime of the hive
pub fn start_heartbeat_task(hive_info: HiveInfo, queen_url: String) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

        loop {
            interval.tick().await;

            // Send heartbeat to queen with full HiveInfo
            if let Err(e) = send_heartbeat_to_queen(&hive_info, &queen_url).await {
                tracing::warn!("Failed to send hive heartbeat: {}", e);
            }
        }
    })
}
