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
//!
//! TEAM-185: Consolidated queen-lifecycle crate into this binary
//! TEAM-185: Renamed actions module to operations
//! TEAM-185: Updated all operation strings to use constants

mod config;
mod job_client;
mod operations;
mod queen_lifecycle;

use config::Config;
use job_client::submit_and_stream_job;
use operations::*;
use queen_lifecycle::ensure_queen_running;
use anyhow::Result;
use clap::{Parser, Subcommand};
use observability_narration_core::Narration;
use rbee_operations::Operation;

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
pub enum Commands {
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
    /// Start queen-rbee daemon
    Start,
    /// Stop queen-rbee daemon
    Stop,
    /// Check queen-rbee daemon status
    Status,
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
        /// TEAM-185: Renamed from 'backend' to 'worker' for clarity
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
                let queen_handle = ensure_queen_running(&queen_url).await?;
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
            QueenAction::Status => {
                // TEAM-186: Check queen-rbee health endpoint
                let client = reqwest::Client::builder()
                    .timeout(tokio::time::Duration::from_secs(5))
                    .build()?;
                
                match client.get(format!("{}/health", queen_url)).send().await {
                    Ok(response) if response.status().is_success() => {
                        Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_STATUS, "running")
                            .human(format!("✅ Queen is running on {}", queen_url))
                            .emit();
                        
                        // Try to get more details from the response
                        if let Ok(body) = response.text().await {
                            println!("Status: {}", body);
                        }
                        Ok(())
                    }
                    Ok(response) => {
                        Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_STATUS, "unhealthy")
                            .human(format!("⚠️  Queen responded with status: {}", response.status()))
                            .emit();
                        Ok(())
                    }
                    Err(_) => {
                        Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_STATUS, "not_running")
                            .human(format!("❌ Queen is not running on {}", queen_url))
                            .emit();
                        Ok(())
                    }
                }
            }
        },

        Commands::Hive { action } => {
            // TEAM-186: Use typed Operation enum instead of JSON strings
            let operation = match action {
                HiveAction::Start { ref id } => Operation::HiveStart { hive_id: id.clone() },
                HiveAction::Stop { ref id } => Operation::HiveStop { hive_id: id.clone() },
                HiveAction::List => Operation::HiveList,
                HiveAction::Get { ref id } => Operation::HiveGet { id: id.clone() },
                HiveAction::Create { ref host, port } => Operation::HiveCreate { host: host.clone(), port },
                HiveAction::Update { ref id } => Operation::HiveUpdate { id: id.clone() },
                HiveAction::Delete { ref id } => Operation::HiveDelete { id: id.clone() },
            };
            let job_payload = serde_json::to_value(&operation)
                .expect("Failed to serialize operation");
            submit_and_stream_job(&client, &queen_url, job_payload).await
        },

        Commands::Worker { hive_id, action } => {
            // TEAM-186: Use typed Operation enum instead of JSON strings
            let operation = match action {
                WorkerAction::Spawn { ref model, ref worker, device } => Operation::WorkerSpawn {
                    hive_id: hive_id.clone(),
                    model: model.clone(),
                    worker: worker.clone(),
                    device,
                },
                WorkerAction::List => Operation::WorkerList { hive_id: hive_id.clone() },
                WorkerAction::Get { ref id } => Operation::WorkerGet { hive_id: hive_id.clone(), id: id.clone() },
                WorkerAction::Delete { ref id } => Operation::WorkerDelete { hive_id: hive_id.clone(), id: id.clone() },
            };
            let job_payload = serde_json::to_value(&operation)
                .expect("Failed to serialize operation");
            submit_and_stream_job(&client, &queen_url, job_payload).await
        },

        Commands::Model { hive_id, action } => {
            // TEAM-186: Use typed Operation enum instead of JSON strings
            let operation = match action {
                ModelAction::Download { ref model } => Operation::ModelDownload {
                    hive_id: hive_id.clone(),
                    model: model.clone(),
                },
                ModelAction::List => Operation::ModelList { hive_id: hive_id.clone() },
                ModelAction::Get { ref id } => Operation::ModelGet { hive_id: hive_id.clone(), id: id.clone() },
                ModelAction::Delete { ref id } => Operation::ModelDelete { hive_id: hive_id.clone(), id: id.clone() },
            };
            let job_payload = serde_json::to_value(&operation)
                .expect("Failed to serialize operation");
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
            // TEAM-186: Use typed Operation enum instead of JSON strings
            let operation = Operation::Infer {
                hive_id: hive_id.clone(),
                model: model.clone(),
                prompt: prompt.clone(),
                max_tokens,
                temperature,
                top_p,
                top_k,
                device: device.clone(),
                worker_id: worker_id.clone(),
                stream,
            };
            let job_payload = serde_json::to_value(&operation)
                .expect("Failed to serialize operation");
            submit_and_stream_job(&client, &queen_url, job_payload).await
        }
    }
}
