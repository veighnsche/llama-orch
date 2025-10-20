//! rbee-keeper - Thin HTTP client for queen-rbee
//!
//! TEAM-151: Migrated CLI from old.rbee-keeper to numbered architecture
//! TEAM-158: Cleaned up over-engineering - rbee-keeper is just a thin HTTP client!
//!
//! # CRITICAL ARCHITECTURE PRINCIPLE
//!
//! **rbee-keeper is a THIN HTTP CLIENT that talks to queen-rbee.**
//!
//! ```
//! User → rbee-keeper (CLI) → queen-rbee (HTTP API) → Everything else
//! ```
//!
//! ## What rbee-keeper does:
//! 1. Parse CLI arguments
//! 2. Ensure queen-rbee is running (auto-start if needed)
//! 3. Make HTTP request to queen-rbee
//! 4. Display response to user
//! 5. Cleanup (shutdown queen if we started it)
//!
//! ## What rbee-keeper does NOT do:
//! - ❌ Complex business logic (that's queen-rbee's job)
//! - ❌ SSH to remote nodes (that's queen-rbee's job)
//! - ❌ Orchestration decisions (that's queen-rbee's job)
//! - ❌ Run as a daemon (CLI tool only)
//!
//! Entry point for the happy flow:
//! ```bash
//! rbee-keeper infer "hello" --model HF:author/minillama
//! ```

mod actions;
mod config;
mod job_client;
mod narration_stream;

use actions::*;
use config::Config;
use job_client::submit_and_stream_job;
use narration_stream::stream_narration_to_stdout;
use anyhow::Result;
use clap::{Parser, Subcommand};
use observability_narration_core::Narration;

#[derive(Parser)]
#[command(name = "rbee")]
#[command(about = "rbee infrastructure management CLI", version)]
#[command(long_about = "CLI tool for managing queen-rbee, hives, workers, and inference")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Queen management
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
        /// Hive ID to operate on (defaults to localhost)
        #[arg(long, default_value = "localhost")]
        hive_id: String,
        #[command(subcommand)]
        action: WorkerAction,
    },

    /// Model management
    Model {
        /// Hive ID to operate on (defaults to localhost)
        #[arg(long, default_value = "localhost")]
        hive_id: String,
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Run inference
    Infer {
        #[arg(long)]
        model: String,
        prompt: String,
        #[arg(long, default_value = "20")]
        max_tokens: u32,
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },
}

#[derive(Subcommand)]
pub enum QueenAction {
    Start,
    Stop,
}

#[derive(Subcommand)]
pub enum HiveAction {
    Start {
        id: String,
    },
    Stop {
        id: String,
    },
    List,
    Get {
        id: String,
    },
    Create {
        #[arg(long)]
        host: String,
        #[arg(long)]
        port: u16,
    },
    Update {
        id: String,
    },
    Delete {
        id: String,
    },
}

#[derive(Subcommand)]
pub enum WorkerAction {
    Spawn {
        #[arg(long)]
        model: String,
        #[arg(long)]
        backend: String,
        #[arg(long)]
        device: u32,
    },
    List,
    Get {
        id: String,
    },
    Delete {
        id: String,
    },
}

#[derive(Subcommand)]
pub enum ModelAction {
    Download { model: String },
    List,
    Get { id: String },
    Delete { id: String },
}



#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    handle_command(cli).await
}

async fn handle_command(cli: Cli) -> Result<()> {
    let config = Config::load()?;
    let queen_url = config.queen_url();
    let client = reqwest::Client::new();

    match cli.command {
        Commands::Queen { action } => match action {
            QueenAction::Start => {
                let queen_handle =
                    rbee_keeper_queen_lifecycle::ensure_queen_running(&queen_url).await?;
                Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_START, queen_handle.base_url())
                    .human(format!("✅ Queen started on {}", queen_handle.base_url()))
                    .emit();
                std::mem::forget(queen_handle);
                Ok(())
            }
            QueenAction::Stop => {
                let client = reqwest::Client::builder()
                    .timeout(tokio::time::Duration::from_secs(30))
                    .build()?;
                match client.post(format!("{}/v1/shutdown", queen_url)).send().await {
                    Ok(_) => {
                        Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_STOP, "stopped")
                            .human("✅ Queen stopped")
                            .emit();
                        Ok(())
                    }
                    Err(_) => {
                        Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_STOP, "not_running")
                            .human("⚠️  Queen not running")
                            .emit();
                        Ok(())
                    }
                }
            }
        },

        Commands::Hive { action } => {
            let job_payload = match action {
                HiveAction::Start { ref id } => serde_json::json!({
                    "operation": "hive_start",
                    "hive_id": id
                }),
                HiveAction::Stop { ref id } => serde_json::json!({
                    "operation": "hive_stop",
                    "hive_id": id
                }),
                HiveAction::List => serde_json::json!({
                    "operation": "hive_list"
                }),
                HiveAction::Get { ref id } => serde_json::json!({
                    "operation": "hive_get",
                    "id": id
                }),
                HiveAction::Create { ref host, port } => serde_json::json!({
                    "operation": "hive_create",
                    "host": host,
                    "port": port
                }),
                HiveAction::Update { ref id } => serde_json::json!({
                    "operation": "hive_update",
                    "id": id
                }),
                HiveAction::Delete { ref id } => serde_json::json!({
                    "operation": "hive_delete",
                    "id": id
                }),
            };
            submit_and_stream_job(&client, &queen_url, job_payload).await
        },

        Commands::Worker { hive_id, action } => {
            let job_payload = match action {
                WorkerAction::Spawn { ref model, ref backend, device } => serde_json::json!({
                    "operation": "worker_spawn",
                    "hive_id": hive_id,
                    "model": model,
                    "backend": backend,
                    "device": device
                }),
                WorkerAction::List => serde_json::json!({
                    "operation": "worker_list",
                    "hive_id": hive_id
                }),
                WorkerAction::Get { ref id } => serde_json::json!({
                    "operation": "worker_get",
                    "hive_id": hive_id,
                    "id": id
                }),
                WorkerAction::Delete { ref id } => serde_json::json!({
                    "operation": "worker_delete",
                    "hive_id": hive_id,
                    "id": id
                }),
            };
            submit_and_stream_job(&client, &queen_url, job_payload).await
        },

        Commands::Model { hive_id, action } => {
            let job_payload = match action {
                ModelAction::Download { ref model } => serde_json::json!({
                    "operation": "model_download",
                    "hive_id": hive_id,
                    "model": model
                }),
                ModelAction::List => serde_json::json!({
                    "operation": "model_list",
                    "hive_id": hive_id
                }),
                ModelAction::Get { ref id } => serde_json::json!({
                    "operation": "model_get",
                    "hive_id": hive_id,
                    "id": id
                }),
                ModelAction::Delete { ref id } => serde_json::json!({
                    "operation": "model_delete",
                    "hive_id": hive_id,
                    "id": id
                }),
            };
            submit_and_stream_job(&client, &queen_url, job_payload).await
        },

        Commands::Infer { ref model, ref prompt, max_tokens, temperature } => {
            let job_payload = serde_json::json!({
                "operation": "infer",
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            });
            submit_and_stream_job(&client, &queen_url, job_payload).await
        }
    }
}
