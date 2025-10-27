//! Worker command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-274: Updated worker actions for new architecture
//! TEAM-324: Moved WorkerAction and WorkerProcessAction enums here to eliminate duplication

use anyhow::Result;
use clap::Subcommand;
use operations_contract::{
    Operation, WorkerProcessDeleteRequest, WorkerProcessGetRequest, WorkerProcessListRequest,
    WorkerSpawnRequest,
}; // TEAM-284: Renamed from rbee_operations

use crate::job_client::submit_and_stream_job;

#[derive(Subcommand)]
pub enum WorkerAction {
    /// Spawn a worker process on hive
    Spawn {
        /// Model identifier
        #[arg(long)]
        model: String,
        /// Device specification: cpu, cuda:0, cuda:1, metal:0, etc.
        #[arg(long)]
        device: String,
    },

    // TEAM-278: DELETED WorkerBinaryAction subcommand
    // Worker binary management (download, build, list, get, delete) is now handled by PackageSync
    /// Worker process management (local ps on hive)
    #[command(subcommand)]
    Process(WorkerProcessAction),
}

// TEAM-278: DELETED WorkerBinaryAction enum entirely
// Replaced by package commands (sync, install, uninstall)

#[derive(Subcommand)]
pub enum WorkerProcessAction {
    /// List worker processes (local ps)
    List,
    /// Get worker process details by PID
    Get { pid: u32 },
    /// Delete (kill) worker process by PID
    Delete { pid: u32 },
}

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
