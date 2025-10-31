//! Top-level CLI commands
//!
//! TEAM-276: Extracted from main.rs

use clap::{Parser, Subcommand};

// TEAM-380: Updated to use HiveLifecycleAction and HiveJobsAction
use super::{HiveJobsAction, HiveLifecycleAction, ModelAction, QueenAction, WorkerAction};

#[derive(Parser)]
#[command(name = "rbee")]
#[command(about = "rbee infrastructure management CLI", version)]
#[command(long_about = "CLI tool for managing queen-rbee, hives, workers, and inference")]
// TEAM-295: Made command optional - if no subcommand, launch Tauri GUI
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
/// TEAM-185: Added comprehensive inference parameters (top_p, top_k, device, worker_id, stream)
/// TEAM-190: Added Status command for live hive/worker overview
/// TEAM-282: Added package manager commands (Sync, PackageStatus, Validate, Migrate)
/// TEAM-284: DELETED package manager commands (SSH/remote operations removed)
pub enum Commands {
    /// Show live status of all hives and workers
    /// TEAM-190: Queries hive-registry for runtime state (not catalog)
    Status,

    /// Run self-check with narration test
    /// TEAM-309: Tests narration system with all 3 modes
    SelfCheck,

    /// Run queen-check (deep narration test through queen job server)
    /// TEAM-312: Tests narration through entire SSE streaming pipeline
    QueenCheck,

    /// Manage queen-rbee daemon
    Queen {
        #[command(subcommand)]
        action: QueenAction,
    },

    /// Hive lifecycle management (start, stop, install, etc.)
    Hive {
        #[command(subcommand)]
        action: HiveLifecycleAction,
    },

    /// Hive job operations (worker/model management)
    HiveJobs {
        /// Hive alias to operate on (defaults to localhost)
        #[arg(long = "hive", default_value = "localhost")]
        hive_id: String,
        #[command(subcommand)]
        action: HiveJobsAction,
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
    // TEAM-284: DELETED PACKAGE MANAGER COMMANDS
    // ============================================================================
    // Sync, PackageStatus, Validate, Migrate removed (SSH/remote operations)
}
