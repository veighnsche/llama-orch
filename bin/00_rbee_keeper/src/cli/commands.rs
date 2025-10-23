//! Top-level CLI commands
//!
//! TEAM-276: Extracted from main.rs

use clap::{Parser, Subcommand};

use super::{HiveAction, ModelAction, QueenAction, WorkerAction};

#[derive(Parser)]
#[command(name = "rbee")]
#[command(about = "rbee infrastructure management CLI", version)]
#[command(long_about = "CLI tool for managing queen-rbee, hives, workers, and inference")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
/// TEAM-185: Added comprehensive inference parameters (top_p, top_k, device, worker_id, stream)
/// TEAM-190: Added Status command for live hive/worker overview
/// TEAM-282: Added package manager commands (Sync, PackageStatus, Validate, Migrate)
pub enum Commands {
    /// Show live status of all hives and workers
    /// TEAM-190: Queries hive-registry for runtime state (not catalog)
    Status,

    /// Manage queen-rbee daemon
    Queen {
        #[command(subcommand)]
        action: QueenAction,
    },

    /// Hive management
    Hive {
        #[command(subcommand)]
        action: HiveAction,
    },

    /// Worker management
    Worker {
        /// Hive alias to operate on (defaults to localhost)
        #[arg(long = "hive", default_value = "localhost")]
        hive_id: String,
        #[command(subcommand)]
        action: WorkerAction,
    },

    /// Model management
    Model {
        /// Hive alias to operate on (defaults to localhost)
        #[arg(long = "hive", default_value = "localhost")]
        hive_id: String,
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Run inference
    Infer {
        /// Hive alias to run inference on
        #[arg(long = "hive", default_value = "localhost")]
        hive_id: String,
        /// Model identifier
        #[arg(long)]
        model: String,
        /// Input prompt
        prompt: String,
        /// Maximum tokens to generate
        #[arg(long, default_value = "20")]
        max_tokens: u32,
        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        /// Nucleus sampling (top_p)
        #[arg(long)]
        top_p: Option<f32>,
        /// Top-k sampling
        #[arg(long)]
        top_k: Option<u32>,
        /// Device type: cpu, cuda, or metal (filters compatible workers)
        #[arg(long)]
        device: Option<String>,
        /// Specific worker ID to use
        #[arg(long)]
        worker_id: Option<String>,
        /// Stream tokens as generated
        #[arg(long, default_value = "true")]
        stream: bool,
    },

    // ============================================================================
    // PACKAGE MANAGER COMMANDS (TEAM-282)
    // ============================================================================
    /// Sync all hives to match declarative config
    /// TEAM-282: Declarative lifecycle management
    Sync {
        /// Show what would be done without making changes
        #[arg(long)]
        dry_run: bool,

        /// Remove components not in config
        #[arg(long)]
        remove_extra: bool,

        /// Force reinstall even if already installed
        #[arg(long)]
        force: bool,

        /// Optional: sync only this hive
        #[arg(long)]
        hive: Option<String>,
    },

    /// Check package status and detect drift from config
    /// TEAM-282: Declarative lifecycle management
    PackageStatus {
        /// Show detailed status information
        #[arg(long)]
        verbose: bool,
    },

    /// Validate declarative config file
    /// TEAM-282: Declarative lifecycle management
    Validate {
        /// Optional: path to config file (default: ~/.config/rbee/hives.conf)
        #[arg(long)]
        config: Option<String>,
    },

    /// Generate declarative config from current state
    /// TEAM-282: Declarative lifecycle management
    Migrate {
        /// Path where config should be written
        #[arg(long)]
        output: String,
    },
}
