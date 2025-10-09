//! CLI argument parsing
//!
//! Created by: TEAM-022

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "llorch")]
#[command(about = "Orchestrator control CLI", version, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Pool management commands
    Pool {
        #[command(subcommand)]
        action: PoolAction,
    },
    /// Test inference on a worker (TEAM-024)
    Infer {
        /// Worker host:port (e.g., localhost:8080)
        #[arg(long)]
        worker: String,
        /// Prompt text
        #[arg(long)]
        prompt: String,
        /// Maximum tokens to generate
        #[arg(long, default_value = "50")]
        max_tokens: u32,
        /// Temperature (0.0-2.0)
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },
}

#[derive(Subcommand)]
pub enum PoolAction {
    /// Model management on remote pool
    Models {
        #[command(subcommand)]
        action: ModelsAction,
        #[arg(long)]
        host: String,
    },
    /// Worker management on remote pool
    Worker {
        #[command(subcommand)]
        action: WorkerAction,
        #[arg(long)]
        host: String,
    },
    /// Git operations on remote pool
    Git {
        #[command(subcommand)]
        action: GitAction,
        #[arg(long)]
        host: String,
    },
    /// Show pool status
    Status {
        #[arg(long)]
        host: String,
    },
}

#[derive(Subcommand)]
pub enum ModelsAction {
    /// Download a model on remote pool
    Download { model: String },
    /// List models on remote pool
    List,
    /// Show catalog on remote pool
    Catalog,
    /// Register a model on remote pool
    Register {
        id: String,
        #[arg(long)]
        name: String,
        #[arg(long)]
        repo: String,
        #[arg(long)]
        architecture: String,
    },
}

#[derive(Subcommand)]
pub enum WorkerAction {
    /// Spawn worker on remote pool
    Spawn {
        backend: String,
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0")]
        gpu: u32,
    },
    /// List workers on remote pool
    List,
    /// Stop worker on remote pool
    Stop { worker_id: String },
}

#[derive(Subcommand)]
pub enum GitAction {
    /// Pull latest changes on remote pool
    Pull,
    /// Show git status on remote pool
    Status,
    /// Build pool-ctl on remote pool
    Build,
}

impl Cli {
    pub fn parse_args() -> Self {
        Self::parse()
    }
}

pub fn handle_command(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Commands::Pool { action } => crate::commands::pool::handle(action),
        Commands::Infer { worker, prompt, max_tokens, temperature } => {
            crate::commands::infer::handle(worker, prompt, max_tokens, temperature)
        }
    }
}
