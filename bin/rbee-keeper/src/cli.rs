//! CLI argument parsing
//!
//! Created by: TEAM-022
//! Modified by: TEAM-027

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rbee")]
#[command(about = "Orchestrator control CLI", version, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Install rbee binaries to standard paths (TEAM-036)
    Install {
        /// Install to system paths (requires sudo)
        #[arg(long)]
        system: bool,
    },
    /// Pool management commands
    Pool {
        #[command(subcommand)]
        action: PoolAction,
    },
    /// Test inference on a worker (TEAM-024)
    /// TEAM-027: Updated for MVP cross-node inference per test-001-mvp.md
    Infer {
        /// Node name (e.g., "mac", "workstation")
        #[arg(long)]
        node: String,
        /// Model reference (e.g., "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
        #[arg(long)]
        model: String,
        /// Prompt text
        #[arg(long)]
        prompt: String,
        /// Maximum tokens to generate
        #[arg(long, default_value = "20")]
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
    /// Build rbee-hive on remote pool
    Build,
}

impl Cli {
    pub fn parse_args() -> Self {
        Self::parse()
    }
}

pub async fn handle_command(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Commands::Install { system } => crate::commands::install::handle(system),
        Commands::Pool { action } => crate::commands::pool::handle(action),
        Commands::Infer { node, model, prompt, max_tokens, temperature } => {
            crate::commands::infer::handle(node, model, prompt, max_tokens, temperature).await
        }
    }
}
