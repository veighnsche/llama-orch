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

mod config;
mod job_client;
mod operations;

use config::Config;
use job_client::submit_and_stream_job;
use operations::*;
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
        /// Hive ID to run inference on
        #[arg(long, default_value = "localhost")]
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
        /// Model identifier
        #[arg(long)]
        model: String,
        /// Worker type: cpu, cuda, or metal
        #[arg(long)]
        worker: String,
        /// Device ID (GPU index for cuda/metal, ignored for cpu)
        #[arg(long, default_value = "0")]
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
                    "operation": OP_HIVE_START,
                    "hive_id": id
                }),
                HiveAction::Stop { ref id } => serde_json::json!({
                    "operation": OP_HIVE_STOP,
                    "hive_id": id
                }),
                HiveAction::List => serde_json::json!({
                    "operation": OP_HIVE_LIST
                }),
                HiveAction::Get { ref id } => serde_json::json!({
                    "operation": OP_HIVE_GET,
                    "id": id
                }),
                HiveAction::Create { ref host, port } => serde_json::json!({
                    "operation": OP_HIVE_CREATE,
                    "host": host,
                    "port": port
                }),
                HiveAction::Update { ref id } => serde_json::json!({
                    "operation": OP_HIVE_UPDATE,
                    "id": id
                }),
                HiveAction::Delete { ref id } => serde_json::json!({
                    "operation": OP_HIVE_DELETE,
                    "id": id
                }),
            };
            submit_and_stream_job(&client, &queen_url, job_payload).await
        },

        Commands::Worker { hive_id, action } => {
            let job_payload = match action {
                WorkerAction::Spawn { ref model, ref worker, device } => serde_json::json!({
                    "operation": OP_WORKER_SPAWN,
                    "hive_id": hive_id,
                    "model": model,
                    "worker": worker,
                    "device": device
                }),
                WorkerAction::List => serde_json::json!({
                    "operation": OP_WORKER_LIST,
                    "hive_id": hive_id
                }),
                WorkerAction::Get { ref id } => serde_json::json!({
                    "operation": OP_WORKER_GET,
                    "hive_id": hive_id,
                    "id": id
                }),
                WorkerAction::Delete { ref id } => serde_json::json!({
                    "operation": OP_WORKER_DELETE,
                    "hive_id": hive_id,
                    "id": id
                }),
            };
            submit_and_stream_job(&client, &queen_url, job_payload).await
        },

        Commands::Model { hive_id, action } => {
            let job_payload = match action {
                ModelAction::Download { ref model } => serde_json::json!({
                    "operation": OP_MODEL_DOWNLOAD,
                    "hive_id": hive_id,
                    "model": model
                }),
                ModelAction::List => serde_json::json!({
                    "operation": OP_MODEL_LIST,
                    "hive_id": hive_id
                }),
                ModelAction::Get { ref id } => serde_json::json!({
                    "operation": OP_MODEL_GET,
                    "hive_id": hive_id,
                    "id": id
                }),
                ModelAction::Delete { ref id } => serde_json::json!({
                    "operation": OP_MODEL_DELETE,
                    "hive_id": hive_id,
                    "id": id
                }),
            };
            submit_and_stream_job(&client, &queen_url, job_payload).await
        },

        Commands::Infer { 
            ref hive_id,
            ref model, 
            ref prompt, 
            max_tokens, 
            temperature,
            top_p,
            top_k,
            ref device,
            ref worker_id,
            stream,
        } => {
            let mut job_payload = serde_json::json!({
                "operation": OP_INFER,
                "hive_id": hive_id,
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
            });
            
            // Add optional parameters if provided
            if let Some(tp) = top_p {
                job_payload["top_p"] = serde_json::json!(tp);
            }
            if let Some(tk) = top_k {
                job_payload["top_k"] = serde_json::json!(tk);
            }
            if let Some(ref dev) = device {
                job_payload["device"] = serde_json::json!(dev);
            }
            if let Some(ref wid) = worker_id {
                job_payload["worker_id"] = serde_json::json!(wid);
            }
            
            submit_and_stream_job(&client, &queen_url, job_payload).await
        }
    }
}
