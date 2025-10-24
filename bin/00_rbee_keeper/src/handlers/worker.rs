//! Worker command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-274: Updated worker actions for new architecture

use anyhow::Result;
use operations_contract::{
    Operation, WorkerProcessDeleteRequest, WorkerProcessGetRequest, WorkerProcessListRequest,
    WorkerSpawnRequest,
}; // TEAM-284: Renamed from rbee_operations

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

            // TEAM-284: Use typed WorkerSpawnRequest
            Operation::WorkerSpawn(WorkerSpawnRequest {
                hive_id: hive_id.clone(),
                model: model.clone(),
                worker: worker.to_string(),
                device: device_id,
            })
        }
        // TEAM-278: DELETED WorkerAction::Binary match arm
        // Worker binary management is now handled by PackageSync
        // TEAM-284: Use typed requests
        WorkerAction::Process(proc_action) => match proc_action {
            WorkerProcessAction::List => {
                Operation::WorkerProcessList(WorkerProcessListRequest { hive_id })
            }
            WorkerProcessAction::Get { pid } => {
                Operation::WorkerProcessGet(WorkerProcessGetRequest {
                    hive_id,
                    pid: *pid,
                })
            }
            WorkerProcessAction::Delete { pid } => {
                Operation::WorkerProcessDelete(WorkerProcessDeleteRequest {
                    hive_id,
                    pid: *pid,
                })
            }
        },
    };
    submit_and_stream_job(queen_url, operation).await
}
