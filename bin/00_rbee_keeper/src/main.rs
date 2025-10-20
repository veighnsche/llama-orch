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

    /// Queen management commands (TEAM-160: E2E testing)
    Queen {
        #[command(subcommand)]
        action: QueenAction,
    },

    /// Add localhost to hive catalog (TEAM-160: E2E test command)
    AddHive {
        /// Hostname (default: localhost)
        #[arg(long, default_value = "localhost")]
        host: String,

        /// Port (default: 8600)
        #[arg(long, default_value = "8600")]
        port: u16,
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

    /// Test queen-rbee health check (for debugging)
    TestHealth {
        /// Queen URL (default: http://localhost:8500)
        #[arg(long, default_value = "http://localhost:8500")]
        queen_url: String,
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
pub enum QueenAction {
    /// Start queen-rbee daemon
    Start,
    
    /// Stop queen-rbee daemon
    Stop,
}

#[derive(Subcommand)]
pub enum HiveAction {
    /// Start rbee-hive on localhost
    Start,
    
    /// Stop rbee-hive on localhost
    Stop,

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
    Download { model: String },

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
    Stop { worker_id: String },
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
        Commands::Infer { model, prompt, max_tokens, temperature, node: _, backend: _, device: _ } => {
            // TEAM-158: rbee-keeper is a thin HTTP client!
            // Step 1: Ensure queen is running (auto-start if needed)
            let queen_handle =
                rbee_keeper_queen_lifecycle::ensure_queen_running("http://localhost:8500").await?;

            // Step 2: Submit job to queen via HTTP POST
            let client = reqwest::Client::new();
            let job_request = serde_json::json!({
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            });

            let response = client
                .post(format!("{}/jobs", queen_handle.base_url()))
                .json(&job_request)
                .send()
                .await?;

            if !response.status().is_success() {
                let error = response.text().await?;
                anyhow::bail!("Failed to submit job: {}", error);
            }

            let job_response: serde_json::Value = response.json().await?;
            let job_id = job_response["job_id"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("No job_id in response"))?;
            let sse_url = job_response["sse_url"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("No sse_url in response"))?;

            println!("✅ Job submitted: {}", job_id);

            // Step 3: Stream SSE events to stdout
            stream_sse_to_stdout(&format!("{}{}", queen_handle.base_url(), sse_url)).await?;

            // Step 4: Cleanup - shutdown queen ONLY if we started it
            queen_handle.shutdown().await?;

            Ok(())
        }

        Commands::Queen { action } => {
            match action {
                QueenAction::Start => {
                    Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_START, "localhost:8500")
                        .human("👑 Starting queen-rbee")
                        .emit();
                    
                    let queen_handle = rbee_keeper_queen_lifecycle::ensure_queen_running("http://localhost:8500").await?;
                    
                    Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_START, queen_handle.base_url())
                        .human(format!("✅ Queen started on {}", queen_handle.base_url()))
                        .emit();
                    
                    // Don't shutdown - keep queen running
                    std::mem::forget(queen_handle);
                    Ok(())
                }
                QueenAction::Stop => {
                    // TEAM-163: Added 30-second timeout for shutdown request
                    Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_STOP, "localhost:8500")
                        .human("👑 Stopping queen-rbee")
                        .emit();
                    let client = reqwest::Client::builder()
                        .timeout(tokio::time::Duration::from_secs(30))
                        .build()?;
                    let response = client.post("http://localhost:8500/shutdown").send().await;
                    
                    match response {
                        Ok(_) => {
                            Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_STOP, "signal_sent")
                                .human("✅ Queen shutdown signal sent")
                                .emit();
                            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                            Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_STOP, "stopped")
                                .human("✅ Queen stopped")
                                .emit();
                            Ok(())
                        }
                        Err(_) => {
                            Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_STOP, "not_running")
                                .human("⚠️  Queen is not running")
                                .emit();
                            Ok(())
                        }
                    }
                }
            }
        }

        Commands::Hive { action } => {
            match action {
                HiveAction::Start => {
                    // ============================================================
                    // ⚠️  WARNING: rbee-keeper is a THIN HTTP CLIENT ONLY!
                    // ============================================================
                    // rbee-keeper should NOT:
                    // - ❌ Know what host/port the hive uses
                    // - ❌ Build JSON requests with orchestration details
                    // - ❌ Make orchestration decisions
                    //
                    // rbee-keeper ONLY:
                    // - ✅ Ensure queen is running
                    // - ✅ Send simple HTTP request to queen
                    // - ✅ Display response to user
                    //
                    // QUEEN decides:
                    // - Where to spawn the hive (localhost vs remote)
                    // - What port to use
                    // - How to spawn it (SSH vs local)
                    // ============================================================
                    
                    Narration::new(ACTOR_RBEE_KEEPER, ACTION_HIVE_START, "request")
                        .human("🐝 Requesting hive start from queen")
                        .emit();
                    
                    // Step 1: Ensure queen is running
                    let queen_handle = rbee_keeper_queen_lifecycle::ensure_queen_running("http://localhost:8500").await?;
                    
                    // Step 2: Send hive start request to queen (NO orchestration details!)
                    let client = reqwest::Client::new();
                    let response = client
                        .post(format!("{}/hive/start", queen_handle.base_url()))
                        .send()
                        .await?;

                    if !response.status().is_success() {
                        let error = response.text().await?;
                        anyhow::bail!("Failed to start hive: {}", error);
                    }

                    let response_json: serde_json::Value = response.json().await?;
                    let hive_url = response_json["hive_url"]
                        .as_str()
                        .unwrap_or("unknown");

                    Narration::new(ACTOR_RBEE_KEEPER, ACTION_HIVE_START, hive_url)
                        .human(format!("✅ Hive started: {}", hive_url))
                        .emit();
                    
                    // Don't shutdown - keep everything running
                    std::mem::forget(queen_handle);
                    Ok(())
                }
                HiveAction::Stop => {
                    // TEAM-163: Added 30-second timeout for shutdown request
                    Narration::new(ACTOR_RBEE_KEEPER, ACTION_HIVE_STOP, "localhost:8600")
                        .human("🐝 Stopping rbee-hive on localhost")
                        .emit();
                    let client = reqwest::Client::builder()
                        .timeout(tokio::time::Duration::from_secs(30))
                        .build()?;
                    let response = client.post("http://localhost:8600/shutdown").send().await;
                    
                    match response {
                        Ok(_) => {
                            Narration::new(ACTOR_RBEE_KEEPER, ACTION_HIVE_STOP, "signal_sent")
                                .human("✅ Hive shutdown signal sent")
                                .emit();
                            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                            Narration::new(ACTOR_RBEE_KEEPER, ACTION_HIVE_STOP, "stopped")
                                .human("✅ Hive stopped")
                                .emit();
                            Ok(())
                        }
                        Err(_) => {
                            Narration::new(ACTOR_RBEE_KEEPER, ACTION_HIVE_STOP, "not_running")
                                .human("⚠️  Hive is not running")
                                .emit();
                            Ok(())
                        }
                    }
                }
                _ => {
                    println!("TODO: Implement other hive commands");
                    Ok(())
                }
            }
        }

        Commands::AddHive { host, port } => {
            // TEAM-160: Add hive to catalog via queen
            println!("📝 Adding hive {} to catalog...", host);

            // Step 1: Ensure queen is running
            let queen_handle =
                rbee_keeper_queen_lifecycle::ensure_queen_running("http://localhost:8500").await?;

            // Step 2: Send add-hive request to queen
            let client = reqwest::Client::new();
            let request = serde_json::json!({
                "host": host,
                "port": port
            });

            let response = client
                .post(format!("{}/add-hive", queen_handle.base_url()))
                .json(&request)
                .send()
                .await?;

            if !response.status().is_success() {
                let error = response.text().await?;
                anyhow::bail!("Failed to add hive: {}", error);
            }

            let response_json: serde_json::Value = response.json().await?;
            let hive_id = response_json["hive_id"]
                .as_str()
                .unwrap_or(&host);

            println!("✅ Hive added: {}", hive_id);

            // Step 3: Cleanup - shutdown queen ONLY if we started it
            queen_handle.shutdown().await?;

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

        Commands::TestHealth { queen_url } => {
            println!("🔍 Testing queen-rbee health at {}", queen_url);

            match health_check::is_queen_healthy(&queen_url).await {
                Ok(true) => {
                    println!("✅ queen-rbee is running and healthy");
                    Ok(())
                }
                Ok(false) => {
                    println!("❌ queen-rbee is not running (connection refused)");
                    println!("   Start queen with: queen-rbee --port 8500");
                    Ok(())
                }
                Err(e) => {
                    println!("⚠️  Health check error: {}", e);
                    Err(e)
                }
            }
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

    println!("📡 Streaming events from queen-rbee...\n");

    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);

        // Print each SSE event to stdout
        for line in text.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..]; // Remove "data: " prefix
                println!("{}", data);

                // Check for [DONE] marker
                if data.contains("[DONE]") {
                    println!("\n✅ Stream complete");
                    return Ok(());
                }
            }
        }
    }

    Ok(())
}
