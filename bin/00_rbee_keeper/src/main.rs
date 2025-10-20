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

mod health_check;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use observability_narration_core::Narration;

// TEAM-164: Actor constants for narration
const ACTOR_RBEE_KEEPER: &str = "🧑‍🌾 rbee-keeper";
const ACTION_QUEEN_START: &str = "queen_start";
const ACTION_QUEEN_STOP: &str = "queen_stop";
const ACTION_HIVE_START: &str = "hive_start";
const ACTION_HIVE_STOP: &str = "hive_stop";
const ACTION_INFER: &str = "infer";
const ACTION_ADD_HIVE: &str = "add_hive";
const ACTION_HEALTH_CHECK: &str = "health_check";
const ACTION_STREAM: &str = "stream_sse";

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
        #[command(subcommand)]
        action: WorkerAction,
    },

    /// Model management
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Job management
    Job {
        #[command(subcommand)]
        action: JobAction,
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
    Start,
    Stop,
    List,
    Get { id: String },
    Create {
        #[arg(long)]
        host: String,
        #[arg(long)]
        port: u16,
    },
    Update { id: String },
    Delete { id: String },
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
    Get { id: String },
    Delete { id: String },
}

#[derive(Subcommand)]
pub enum ModelAction {
    Download { model: String },
    List,
    Get { id: String },
    Delete { id: String },
}

#[derive(Subcommand)]
pub enum JobAction {
    Stream { id: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    handle_command(cli).await
}

async fn handle_command(cli: Cli) -> Result<()> {
    let queen_url = "http://localhost:8500";
    let client = reqwest::Client::new();

    match cli.command {
        Commands::Queen { action } => match action {
            QueenAction::Start => {
                let queen_handle = rbee_keeper_queen_lifecycle::ensure_queen_running(queen_url).await?;
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
            let queen_handle = rbee_keeper_queen_lifecycle::ensure_queen_running(queen_url).await?;
            match action {
                HiveAction::Start => {
                    let res = client.post(format!("{}/v1/hive/start", queen_url)).send().await?;
                    let json: serde_json::Value = res.json().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, ACTION_HIVE_START, &json.to_string())
                        .human(format!("✅ {}", json))
                        .emit();
                }
                HiveAction::Stop => {
                    client.post(format!("{}/v1/hive/stop", queen_url)).send().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, ACTION_HIVE_STOP, "stopped")
                        .human("✅ Hive stopped")
                        .emit();
                }
                HiveAction::List => {
                    let res = client.get(format!("{}/v1/hives", queen_url)).send().await?;
                    let json: serde_json::Value = res.json().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "hive_list", "success")
                        .human(serde_json::to_string_pretty(&json)?)
                        .emit();
                }
                HiveAction::Get { ref id } => {
                    let res = client.get(format!("{}/v1/hives/{}", queen_url, id)).send().await?;
                    let json: serde_json::Value = res.json().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "hive_get", id)
                        .human(serde_json::to_string_pretty(&json)?)
                        .emit();
                }
                HiveAction::Create { ref host, port } => {
                    let body = serde_json::json!({ "host": host, "port": port });
                    let res = client.post(format!("{}/v1/hives", queen_url)).json(&body).send().await?;
                    let json: serde_json::Value = res.json().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "hive_create", host)
                        .human(format!("✅ {}", json))
                        .emit();
                }
                HiveAction::Update { ref id } => {
                    client.put(format!("{}/v1/hives/{}", queen_url, id)).send().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "hive_update", id)
                        .human(format!("✅ Hive {} updated", id))
                        .emit();
                }
                HiveAction::Delete { ref id } => {
                    client.delete(format!("{}/v1/hives/{}", queen_url, id)).send().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "hive_delete", id)
                        .human(format!("✅ Hive {} deleted", id))
                        .emit();
                }
            }
            std::mem::forget(queen_handle);
            Ok(())
        },

        Commands::Worker { action } => {
            let queen_handle = rbee_keeper_queen_lifecycle::ensure_queen_running(queen_url).await?;
            match action {
                WorkerAction::Spawn { ref model, ref backend, device } => {
                    let body = serde_json::json!({ "model": model, "backend": backend, "device": device });
                    let res = client.post(format!("{}/v1/workers/spawn", queen_url)).json(&body).send().await?;
                    let json: serde_json::Value = res.json().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "worker_spawn", model)
                        .human(format!("✅ {}", json))
                        .emit();
                }
                WorkerAction::List => {
                    let res = client.get(format!("{}/v1/workers", queen_url)).send().await?;
                    let json: serde_json::Value = res.json().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "worker_list", "success")
                        .human(serde_json::to_string_pretty(&json)?)
                        .emit();
                }
                WorkerAction::Get { ref id } => {
                    let res = client.get(format!("{}/v1/workers/{}", queen_url, id)).send().await?;
                    let json: serde_json::Value = res.json().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "worker_get", id)
                        .human(serde_json::to_string_pretty(&json)?)
                        .emit();
                }
                WorkerAction::Delete { ref id } => {
                    client.delete(format!("{}/v1/workers/{}", queen_url, id)).send().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "worker_delete", id)
                        .human(format!("✅ Worker {} deleted", id))
                        .emit();
                }
            }
            std::mem::forget(queen_handle);
            Ok(())
        },

        Commands::Model { action } => {
            let queen_handle = rbee_keeper_queen_lifecycle::ensure_queen_running(queen_url).await?;
            match action {
                ModelAction::Download { ref model } => {
                    let body = serde_json::json!({ "model": model });
                    let res = client.post(format!("{}/v1/models/download", queen_url)).json(&body).send().await?;
                    let json: serde_json::Value = res.json().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "model_download", model)
                        .human(format!("✅ {}", json))
                        .emit();
                }
                ModelAction::List => {
                    let res = client.get(format!("{}/v1/models", queen_url)).send().await?;
                    let json: serde_json::Value = res.json().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "model_list", "success")
                        .human(serde_json::to_string_pretty(&json)?)
                        .emit();
                }
                ModelAction::Get { ref id } => {
                    let res = client.get(format!("{}/v1/models/{}", queen_url, id)).send().await?;
                    let json: serde_json::Value = res.json().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "model_get", id)
                        .human(serde_json::to_string_pretty(&json)?)
                        .emit();
                }
                ModelAction::Delete { ref id } => {
                    client.delete(format!("{}/v1/models/{}", queen_url, id)).send().await?;
                    Narration::new(ACTOR_RBEE_KEEPER, "model_delete", id)
                        .human(format!("✅ Model {} deleted", id))
                        .emit();
                }
            }
            std::mem::forget(queen_handle);
            Ok(())
        },

        Commands::Job { action } => {
            let queen_handle = rbee_keeper_queen_lifecycle::ensure_queen_running(queen_url).await?;
            match action {
                JobAction::Stream { id } => {
                    stream_sse_to_stdout(&format!("{}/v1/jobs/{}/stream", queen_url, id)).await?;
                }
            }
            std::mem::forget(queen_handle);
            Ok(())
        },

        Commands::Infer { ref model, ref prompt, max_tokens, temperature } => {
            let queen_handle = rbee_keeper_queen_lifecycle::ensure_queen_running(queen_url).await?;
            let body = serde_json::json!({
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            });
            let res = client.post(format!("{}/v1/jobs", queen_url)).json(&body).send().await?;
            let json: serde_json::Value = res.json().await?;
            let job_id = json["job_id"].as_str().ok_or_else(|| anyhow::anyhow!("No job_id"))?;
            let sse_url = json["sse_url"].as_str().ok_or_else(|| anyhow::anyhow!("No sse_url"))?;
            Narration::new(ACTOR_RBEE_KEEPER, ACTION_INFER, job_id)
                .human(format!("✅ Job {}", job_id))
                .emit();
            stream_sse_to_stdout(&format!("{}{}", queen_url, sse_url)).await?;
            queen_handle.shutdown().await?;
            Ok(())
        }
    }
}

// ============================================================================
// HTTP Client Helpers
// ============================================================================
// TEAM-158: Simple HTTP helpers - rbee-keeper is just a thin HTTP client!

/// Stream SSE events from queen-rbee to stdout
///
/// This is the "keeper → queen → SSE" part of the happy flow (lines 23-24)
async fn stream_sse_to_stdout(sse_url: &str) -> Result<()> {
    use futures::StreamExt;

    let client = reqwest::Client::new();
    let response = client.get(sse_url).send().await?;

    if !response.status().is_success() {
        let error = response.text().await?;
        anyhow::bail!("Failed to connect to SSE stream: {}", error);
    }

    Narration::new(ACTOR_RBEE_KEEPER, ACTION_STREAM, sse_url)
        .human("📡 Streaming events from queen-rbee...")
        .emit();

    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);

        // Print each SSE event to stdout
        for line in text.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..]; // Remove "data: " prefix
                Narration::new(ACTOR_RBEE_KEEPER, ACTION_STREAM, "token")
                    .human(data)
                    .emit();

                // Check for [DONE] marker
                if data.contains("[DONE]") {
                    Narration::new(ACTOR_RBEE_KEEPER, ACTION_STREAM, "complete")
                        .human("✅ Stream complete")
                        .emit();
                    return Ok(());
                }
            }
        }
    }

    Ok(())
}
