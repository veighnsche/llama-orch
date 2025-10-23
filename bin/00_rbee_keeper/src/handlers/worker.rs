//! Worker command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-274: Updated worker actions for new architecture

use anyhow::Result;
use rbee_operations::Operation;

// TEAM-278: DELETED WorkerBinaryAction import - no longer exists
use crate::cli::{WorkerAction, WorkerProcessAction};
use crate::job_client::submit_and_stream_job;

pub async fn handle_worker(hive_id: String, action: WorkerAction, queen_url: &str) -> Result<()> {
    let operation = match &action {
        WorkerAction::Spawn { model, device } => {
            // Parse device string (e.g., "cuda:0" -> worker="cuda", device=0)
            let (worker, device_id) = if device.contains(':') {
                let parts: Vec<&str> = device.split(':').collect();
                let worker_type = parts[0].to_string();
                let device_num = parts.get(1).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                (worker_type, device_num)
            } else {
                // If no colon, assume device 0
                (device.clone(), 0)
            };

            Operation::WorkerSpawn {
                hive_id,
                model: model.clone(),
                worker,
                device: device_id,
            }
        }
        // TEAM-278: DELETED WorkerAction::Binary match arm
        // Worker binary management is now handled by PackageSync
        WorkerAction::Process(process_action) => match process_action {
            WorkerProcessAction::List => Operation::WorkerProcessList { hive_id },
            WorkerProcessAction::Get { pid } => Operation::WorkerProcessGet { hive_id, pid: *pid },
            WorkerProcessAction::Delete { pid } => {
                Operation::WorkerProcessDelete { hive_id, pid: *pid }
            }
        },
    };
    submit_and_stream_job(queen_url, operation).await
}
