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
