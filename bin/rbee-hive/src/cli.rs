//! CLI argument parsing
//!
//! Created by: TEAM-022
//! Modified by: TEAM-027

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rbee-hive")]
#[command(about = "Pool manager control CLI", version, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Model management commands
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
    /// Worker management commands
    Worker {
        #[command(subcommand)]
        action: WorkerAction,
    },
    /// Show pool status
    Status,
    /// Start HTTP daemon (pool manager)
    /// TEAM-027: Per test-001-mvp.md Phase 2
    Daemon {
        #[arg(long, default_value = "0.0.0.0:8080")]
        addr: String,
    },
}

#[derive(Subcommand)]
pub enum ModelsAction {
    /// Download a model
    Download { model: String },
    /// List available models
    List,
    /// Show model catalog
    Catalog,
    /// Register a new model in the catalog
    Register {
        id: String,
        #[arg(long)]
        name: String,
        #[arg(long)]
        repo: String,
        #[arg(long)]
        architecture: String,
    },
    /// Remove a model from the catalog
    Unregister { id: String },
}

#[derive(Subcommand)]
pub enum WorkerAction {
    /// Spawn a new worker
    Spawn {
        backend: String,
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0")]
        gpu: u32,
    },
    /// List running workers
    List,
    /// Stop a worker
    Stop { worker_id: String },
}

impl Cli {
    pub fn parse_args() -> Self {
        Self::parse()
    }
}

pub async fn handle_command(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Commands::Models { action } => crate::commands::models::handle(action),
        Commands::Worker { action } => crate::commands::worker::handle(action),
        Commands::Status => crate::commands::status::handle(),
        Commands::Daemon { addr } => crate::commands::daemon::handle(addr).await,
    }
}
