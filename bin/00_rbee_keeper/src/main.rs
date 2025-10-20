//! rbee-keeper - CLI for managing rbee infrastructure
//!
//! TEAM-151: Migrated from old.rbee-keeper to numbered architecture
//! This binary contains only CLI parsing; all command logic is in rbee-keeper-commands crate
//!
//! Entry point for the happy flow:
//! ```bash
//! rbee-keeper infer "hello" --model HF:author/minillama
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rbee-keeper")]
#[command(about = "rbee infrastructure management CLI", version)]
#[command(long_about = "CLI tool for managing queen-rbee, hives, workers, and inference")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run inference request (happy flow entry point)
    Infer {
        /// Model reference (e.g., "HF:author/minillama")
        #[arg(long)]
        model: String,
        
        /// Prompt text
        prompt: String,
        
        /// Maximum tokens to generate
        #[arg(long, default_value = "20")]
        max_tokens: u32,
        
        /// Temperature (0.0-2.0)
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        
        /// Node name (optional, for multi-node setups)
        #[arg(long)]
        node: Option<String>,
        
        /// Backend (e.g., "cuda", "cpu", "metal")
        #[arg(long)]
        backend: Option<String>,
        
        /// Device ID (e.g., 0, 1)
        #[arg(long)]
        device: Option<u32>,
    },

    /// Setup and manage rbee-hive nodes
    Setup {
        #[command(subcommand)]
        action: SetupAction,
    },

    /// Hive management commands
    Hive {
        #[command(subcommand)]
        action: HiveAction,
    },

    /// Worker management commands
    Workers {
        #[command(subcommand)]
        action: WorkersAction,
    },

    /// View logs from remote nodes
    Logs {
        /// Node name
        #[arg(long)]
        node: String,
        
        /// Follow log output
        #[arg(long)]
        follow: bool,
    },

    /// Install rbee binaries to standard paths
    Install {
        /// Install to system paths (requires sudo)
        #[arg(long)]
        system: bool,
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
    /// Download a model on remote hive
    Download { 
        model: String 
    },
    
    /// List models on remote hive
    List,
    
    /// Show catalog on remote hive
    Catalog,
    
    /// Register a model on remote hive
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
    /// Spawn worker on remote hive
    Spawn {
        backend: String,
        
        #[arg(long)]
        model: String,
        
        #[arg(long, default_value = "0")]
        gpu: u32,
    },
    
    /// List workers on remote hive
    List,
    
    /// Stop worker on remote hive
    Stop { 
        worker_id: String 
    },
}

#[derive(Subcommand)]
pub enum GitAction {
    /// Pull latest changes on remote hive
    Pull,
    
    /// Show git status on remote hive
    Status,
    
    /// Build rbee-hive on remote hive
    Build,
}

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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    handle_command(cli).await
}

async fn handle_command(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Infer { model, prompt, max_tokens, temperature, node, backend, device } => {
            // TODO: Call rbee-keeper-commands::infer::handle()
            println!("TODO: Implement infer command");
            println!("  Model: {}", model);
            println!("  Prompt: {}", prompt);
            println!("  Max tokens: {}", max_tokens);
            println!("  Temperature: {}", temperature);
            if let Some(n) = node {
                println!("  Node: {}", n);
            }
            if let Some(b) = backend {
                println!("  Backend: {}", b);
            }
            if let Some(d) = device {
                println!("  Device: {}", d);
            }
            Ok(())
        }
        
        Commands::Setup { action } => {
            // TODO: Call rbee-keeper-commands::setup::handle()
            println!("TODO: Implement setup command");
            match action {
                SetupAction::AddNode { name, .. } => {
                    println!("  Action: Add node '{}'", name);
                }
                SetupAction::ListNodes => {
                    println!("  Action: List nodes");
                }
                SetupAction::RemoveNode { name } => {
                    println!("  Action: Remove node '{}'", name);
                }
                SetupAction::Install { node } => {
                    println!("  Action: Install on node '{}'", node);
                }
            }
            Ok(())
        }
        
        Commands::Hive { action: _ } => {
            // TODO: Call rbee-keeper-commands::hive::handle()
            println!("TODO: Implement hive command");
            Ok(())
        }
        
        Commands::Workers { action: _ } => {
            // TODO: Call rbee-keeper-commands::workers::handle()
            println!("TODO: Implement workers command");
            Ok(())
        }
        
        Commands::Logs { node, follow } => {
            // TODO: Call rbee-keeper-commands::logs::handle()
            println!("TODO: Implement logs command");
            println!("  Node: {}", node);
            println!("  Follow: {}", follow);
            Ok(())
        }
        
        Commands::Install { system } => {
            // TODO: Call rbee-keeper-commands::install::handle()
            println!("TODO: Implement install command");
            println!("  System: {}", system);
            Ok(())
        }
    }
}
