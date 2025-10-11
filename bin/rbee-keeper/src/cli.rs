//! CLI argument parsing
//!
//! Created by: TEAM-022
//! Modified by: TEAM-027
//! Modified by: TEAM-036 (added Install command)

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
    /// Setup and manage rbee-hive nodes (TEAM-043)
    Setup {
        #[command(subcommand)]
        action: SetupAction,
    },
    /// Hive management commands (TEAM-085: renamed from "pool")
    Hive {
        #[command(subcommand)]
        action: HiveAction,
    },
    /// Test inference on a worker (TEAM-024)
    /// TEAM-027: Updated for MVP cross-node inference per test-001-mvp.md
    /// TEAM-055: Added backend and device parameters per test-001 spec
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
        /// Backend (e.g., "cuda", "cpu", "metal")
        #[arg(long)]
        backend: Option<String>,
        /// Device ID (e.g., 0, 1)
        #[arg(long)]
        device: Option<u32>,
    },
    /// Worker management commands (TEAM-046)
    Workers {
        #[command(subcommand)]
        action: WorkersAction,
    },
    /// View logs from remote nodes (TEAM-046)
    Logs {
        /// Node name
        #[arg(long)]
        node: String,
        /// Follow log output
        #[arg(long)]
        follow: bool,
    },
}

#[derive(Subcommand)]
pub enum SetupAction {
    /// Add a remote rbee-hive node to the registry
    AddNode {
        /// Node name (e.g., "mac", "workstation")
        #[arg(long)]
        name: String,
        /// SSH hostname
        #[arg(long)]
        ssh_host: String,
        /// SSH port
        #[arg(long, default_value = "22")]
        ssh_port: u16,
        /// SSH username
        #[arg(long)]
        ssh_user: String,
        /// Path to SSH private key
        #[arg(long)]
        ssh_key: Option<String>,
        /// Git repository URL
        #[arg(long)]
        git_repo: String,
        /// Git branch
        #[arg(long, default_value = "main")]
        git_branch: String,
        /// Installation path on remote node
        #[arg(long)]
        install_path: String,
    },
    /// List registered rbee-hive nodes
    ListNodes,
    /// Remove a node from the registry
    RemoveNode {
        /// Node name to remove
        #[arg(long)]
        name: String,
    },
    /// Install rbee-hive on a remote node
    Install {
        /// Node name to install on
        #[arg(long)]
        node: String,
    },
}

#[derive(Subcommand)]
pub enum HiveAction {
    /// Model management on remote hive
    Models {
        #[command(subcommand)]
        action: ModelsAction,
        #[arg(long)]
        host: String,
    },
    /// Worker management on remote hive
    Worker {
        #[command(subcommand)]
        action: WorkerAction,
        #[arg(long)]
        host: String,
    },
    /// Git operations on remote hive
    Git {
        #[command(subcommand)]
        action: GitAction,
        #[arg(long)]
        host: String,
    },
    /// Check hive status
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

// TEAM-046: Worker management commands
#[derive(Subcommand)]
pub enum WorkersAction {
    /// List all registered workers
    List,
    /// Check worker health on a specific node
    Health {
        /// Node name
        #[arg(long)]
        node: String,
    },
    /// Manually shutdown a worker
    Shutdown {
        /// Worker ID
        #[arg(long)]
        id: String,
    },
}

impl Cli {
    pub fn parse_args() -> Self {
        Self::parse()
    }
}

pub async fn handle_command(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Commands::Install { system } => crate::commands::install::handle(system),
        Commands::Setup { action } => crate::commands::setup::handle(action).await,
        Commands::Hive { action } => crate::commands::hive::handle(action),
        Commands::Infer { node, model, prompt, max_tokens, temperature, backend, device } => {
            crate::commands::infer::handle(node, model, prompt, max_tokens, temperature, backend, device).await
        }
        Commands::Workers { action } => crate::commands::workers::handle(action).await,
        Commands::Logs { node, follow } => crate::commands::logs::handle(node, follow).await,
    }
}
