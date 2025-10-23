//! Worker CLI actions
//!
//! TEAM-276: Extracted from main.rs

use clap::Subcommand;

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

    /// Worker binary management (catalog on hive)
    #[command(subcommand)]
    Binary(WorkerBinaryAction),

    /// Worker process management (local ps on hive)
    #[command(subcommand)]
    Process(WorkerProcessAction),
}

#[derive(Subcommand)]
pub enum WorkerBinaryAction {
    /// List worker binaries on hive
    List,
    /// Get worker binary details
    Get { worker_type: String },
    /// Delete worker binary
    Delete { worker_type: String },
}

#[derive(Subcommand)]
pub enum WorkerProcessAction {
    /// List worker processes (local ps)
    List,
    /// Get worker process details by PID
    Get { pid: u32 },
    /// Delete (kill) worker process by PID
    Delete { pid: u32 },
}
