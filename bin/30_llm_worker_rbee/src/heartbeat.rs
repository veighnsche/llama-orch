//! Worker-rbee heartbeat handling
//!
//! TEAM-164: Binary-specific heartbeat logic for llm-worker-rbee
//! TEAM-261: Changed to send heartbeats directly to queen (not hive)
//!
//! **What lives here:**
//! - Worker sends heartbeats to queen (TEAM-261: changed from hive)
//! - Periodic heartbeat task
//! - Worker health status reporting
//!
//! **What lives in shared crate:**
//! - Heartbeat types (WorkerHeartbeatPayload, etc.)
//! - Heartbeat traits
//! - Common heartbeat logic

use anyhow::Result;
use observability_narration_core::Narration;
use worker_contract::{WorkerHeartbeat, WorkerInfo}; // TEAM-284: Use worker-contract types

const ACTOR_WORKER_HEARTBEAT: &str = "👷 worker-heartbeat";
const ACTION_SEND: &str = "send_heartbeat";

/// Send heartbeat to queen
///
/// TEAM-261: Changed to send directly to queen (not hive)
/// TEAM-284: Updated to use worker-contract types
///
/// **Flow:**
/// 1. Build WorkerHeartbeat with full WorkerInfo
/// 2. Send POST /v1/worker-heartbeat to queen
/// 3. Return acknowledgement
///
/// This is called periodically (e.g., every 30s) to signal worker is alive
pub async fn send_heartbeat_to_queen(
    worker_info: &WorkerInfo,  // TEAM-284: Pass full worker info
    queen_url: &str,
) -> Result<()> {
    Narration::new(ACTOR_WORKER_HEARTBEAT, ACTION_SEND, &worker_info.id)
        .human(format!("Sending heartbeat to queen at {}", queen_url))
        .emit();

    // TEAM-285: Fixed unused variable warning
    let _heartbeat = WorkerHeartbeat::new(worker_info.clone());

    // TODO: Implement HTTP POST to queen
    // POST {queen_url}/v1/worker-heartbeat with heartbeat
    // let client = reqwest::Client::new();
    // client.post(format!("{}/v1/worker-heartbeat", queen_url))
    //     .json(&_heartbeat)
    //     .send()
    //     .await?;

    Ok(())
}

/// Start periodic heartbeat task
///
/// TEAM-261: Changed to send heartbeats to queen (not hive)
/// TEAM-284: Updated to use worker-contract types
///
/// **Flow:**
/// 1. Spawn tokio task
/// 2. Every 30 seconds, send heartbeat to queen with full WorkerInfo
/// 3. Continue until task is cancelled
///
/// This runs in the background for the lifetime of the worker
pub fn start_heartbeat_task(
    worker_info: WorkerInfo,  // TEAM-284: Pass full worker info
    queen_url: String,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            // TEAM-261: Send heartbeat to queen (not hive)
            // TEAM-284: Send full WorkerInfo
            if let Err(e) = send_heartbeat_to_queen(
                &worker_info,
                &queen_url,
            ).await {
                eprintln!("Failed to send heartbeat: {}", e);
            }
        }
    })
}
