//! rbee-keeper - Thin HTTP client for queen-rbee
//!
//! TEAM-151: Migrated CLI from old.rbee-keeper to numbered architecture
//! TEAM-158: Cleaned up over-engineering - rbee-keeper is just a thin HTTP client!
//! TEAM-216: Investigated - Complete behavior inventory created
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

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::Config;
use job_client::submit_and_stream_job;
use observability_narration_core::NarrationFactory;
use queen_lifecycle::ensure_queen_running;
use rbee_operations::Operation;

/// Narration factory for rbee-keeper main operations.
///
/// # Usage
/// ```rust,ignore
/// use crate::narration::NARRATE;
///
/// NARRATE.narrate("queen_start")
///     .context(queen_url)
///     .human("🚀 Starting queen on {}")
///     .emit();
/// ```
const NARRATE: NarrationFactory = NarrationFactory::new("keeper");

#[derive(Parser)]
#[command(name = "rbee")]
#[command(about = "rbee infrastructure management CLI", version)]
#[command(long_about = "CLI tool for managing queen-rbee, hives, workers, and inference")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

// ============================================================================
// CLI COMMANDS (Step 3 of 3-File Pattern)
// ============================================================================
//
// When adding a new operation:
// 1. Add Operation variant in rbee-operations/src/lib.rs (DONE)
// 2. Add match arm in job_router.rs (DONE)
// 3. Add CLI command HERE:
//    a. Add variant to Commands/HiveAction/WorkerAction/etc. enum
//    b. Add match arm in handle_command() to construct Operation
//
// The CLI is just a thin HTTP client - all business logic lives in queen-rbee.
// ============================================================================

#[derive(Subcommand)]
/// TEAM-185: Added comprehensive inference parameters (top_p, top_k, device, worker_id, stream)
/// TEAM-190: Added Status command for live hive/worker overview
pub enum Commands {
    /// Show live status of all hives and workers
    /// TEAM-190: Queries hive-registry for runtime state (not catalog)
    Status,

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
        /// Hive alias to operate on (defaults to localhost)
        #[arg(long = "hive", default_value = "localhost")]
        hive_id: String,
        #[command(subcommand)]
        action: WorkerAction,
    },

    /// Model management
    Model {
        /// Hive alias to operate on (defaults to localhost)
        #[arg(long = "hive", default_value = "localhost")]
        hive_id: String,
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Run inference
    Infer {
        /// Hive alias to run inference on
        #[arg(long = "hive", default_value = "localhost")]
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
    /// Rebuild queen with different configuration
    /// TEAM-262: Added for local-hive optimization
    Rebuild {
        /// Include local hive for localhost operations (50-100x faster)
        #[arg(long)]
        with_local_hive: bool,
    },
    /// Show queen build configuration
    /// TEAM-262: Query /v1/build-info endpoint
    Info,
    /// Install queen binary
    /// TEAM-262: Similar to hive install
    Install {
        /// Binary path (optional, auto-detect from target/)
        #[arg(short, long)]
        binary: Option<String>,
    },
    /// Uninstall queen binary
    /// TEAM-262: Similar to hive uninstall
    Uninstall,
}

#[derive(Subcommand)]
// TEAM-186: Updated to match new install/uninstall workflow
// TEAM-187: Added SshTest for pre-installation SSH validation
// ============================================================
// BUG FIX: TEAM-199 | Clap short option conflict -h vs --help
// ============================================================
// SUSPICION:
// - Error message: "Short option names must be unique for each argument,
//   but '-h' is in use by both 'alias' and 'help'"
// - Suspected all HiveAction variants using #[arg(short = 'h')]
//
// INVESTIGATION:
// - Checked main.rs line 159, 165, 171, 179, 185, 191, 198, 205
// - All HiveAction variants use -h for alias/host parameter
// - Clap auto-generates -h for --help flag
// - Conflict occurs when subcommand is parsed
//
// ROOT CAUSE:
// - Multiple HiveAction variants define #[arg(short = 'h')] for alias
// - Clap reserves -h for --help by default
// - Cannot use same short option for both help and custom argument
//
// FIX:
// - Changed all short = 'h' to short = 'a' (for alias)
// - Long option --host remains unchanged
// - Users can now use -a or --host for alias parameter
// - -h now correctly shows help as expected
//
// TESTING:
// - cargo build --bin rbee-keeper (compilation check)
// - ./rbee hive start (should work without panic)
// - ./rbee hive start -h (should show help)
// - ./rbee hive start -a localhost (should work)
// - ./rbee hive start --host localhost (should work)
// ============================================================
pub enum HiveAction {
    /// Test SSH connection using config from hives.conf
    SshTest {
        /// Hive alias from ~/.config/rbee/hives.conf
        #[arg(short = 'a', long = "host")]
        alias: String,
    },
    /// Install a hive (must be configured in hives.conf first)
    Install {
        /// Hive alias from ~/.config/rbee/hives.conf
        #[arg(short = 'a', long = "host")]
        alias: String,
    },
    /// Uninstall a hive
    Uninstall {
        /// Hive alias from ~/.config/rbee/hives.conf
        #[arg(short = 'a', long = "host")]
        alias: String,
    },
    /// List all hives
    List,
    /// Start a hive
    Start {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Stop a hive
    Stop {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Get hive details
    Get {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Check hive status
    /// TEAM-189: New command to check if hive is running via health endpoint
    Status {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Refresh device capabilities for a hive
    /// TEAM-196: Fetch and cache device capabilities from a running hive
    RefreshCapabilities {
        /// Hive alias from ~/.config/rbee/hives.conf
        #[arg(short = 'a', long = "host")]
        alias: String,
    },
    /// Import SSH config into hives.conf
    ImportSsh {
        /// Path to SSH config file
        #[arg(long, default_value = "~/.ssh/config")]
        ssh_config: String,
        /// Default HivePort for all imported hosts
        #[arg(long, default_value = "8081")]
        default_port: u16,
    },
}

#[derive(Subcommand)]
pub enum WorkerAction {
    Spawn {
        /// Model identifier
        #[arg(long)]
        model: String,
        /// Device specification: cpu, cuda:0, cuda:1, metal:0, etc.
        #[arg(long)]
        device: String,
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

    // ============================================================================
    // COMMAND ROUTING (Step 3b of 3-File Pattern)
    // ============================================================================
    //
    // For each CLI command, construct the corresponding Operation and submit it.
    // Pattern:
    //   1. Extract CLI arguments
    //   2. Construct Operation::Xxx { ... }
    //   3. Call submit_and_stream_job()
    //
    // See existing commands below for examples.
    // ============================================================================

    match cli.command {
        Commands::Status => {
            // TEAM-190: Show live status of all hives and workers from registry
            let operation = Operation::Status;
            submit_and_stream_job(&queen_url, operation).await
        }

        Commands::Queen { action } => match action {
            QueenAction::Start => {
                let queen_handle = ensure_queen_running(&queen_url).await?;
                NARRATE
                    .action("queen_start")
                    .context(queen_handle.base_url())
                    .human("✅ Queen started on {}")
                    .emit();
                std::mem::forget(queen_handle);
                Ok(())
            }
            QueenAction::Stop => {
                // First check if queen is running
                let health_check = client.get(format!("{}/health", queen_url)).send().await;

                let is_running =
                    matches!(health_check, Ok(response) if response.status().is_success());

                if !is_running {
                    NARRATE.action("queen_stop").human("⚠️  Queen not running").emit();
                    return Ok(());
                }

                // Queen is running, send shutdown request
                let shutdown_client = reqwest::Client::builder()
                    .timeout(tokio::time::Duration::from_secs(30))
                    .build()?;

                match shutdown_client.post(format!("{}/v1/shutdown", queen_url)).send().await {
                    Ok(_) => {
                        NARRATE.action("queen_stop").human("✅ Queen stopped").emit();
                        Ok(())
                    }
                    Err(e) => {
                        // Connection closed/reset is expected - queen shuts down before responding
                        if e.is_connect() || e.to_string().contains("connection closed") {
                            NARRATE.action("queen_stop").human("✅ Queen stopped").emit();
                            Ok(())
                        } else {
                            // Unexpected error
                            NARRATE
                                .action("queen_stop")
                                .context(e.to_string())
                                .human("⚠️  Failed to stop queen: {}")
                                .error_kind("shutdown_failed")
                                .emit();
                            Err(e.into())
                        }
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
                        NARRATE
                            .action("queen_status")
                            .context(&queen_url)
                            .human("✅ Queen is running on {}")
                            .emit();

                        // Try to get more details from the response
                        if let Ok(body) = response.text().await {
                            println!("Status: {}", body);
                        }
                        Ok(())
                    }
                    Ok(response) => {
                        NARRATE
                            .action("queen_status")
                            .context(response.status().to_string())
                            .human("⚠️  Queen responded with status: {}")
                            .emit();
                        Ok(())
                    }
                    Err(_) => {
                        NARRATE
                            .action("queen_status")
                            .context(&queen_url)
                            .human("❌ Queen is not running on {}")
                            .emit();
                        Ok(())
                    }
                }
            }
            QueenAction::Rebuild { with_local_hive } => {
                // TEAM-262: Added queen rebuild command for local-hive optimization
                // TEAM-263: Implemented actual build logic
                NARRATE.action("queen_rebuild").human("🔨 Rebuilding queen-rbee...").emit();

                // Determine build command
                let mut cmd = std::process::Command::new("cargo");
                cmd.arg("build").arg("--release").arg("--bin").arg("queen-rbee");

                if with_local_hive {
                    NARRATE
                        .action("queen_rebuild")
                        .human(
                            "✨ Building with integrated local hive (50-100x faster localhost)...",
                        )
                        .emit();
                    cmd.arg("--features").arg("local-hive");
                } else {
                    NARRATE
                        .action("queen_rebuild")
                        .human("📡 Building distributed queen (remote hives only)...")
                        .emit();
                }

                // Execute build
                NARRATE
                    .action("queen_rebuild")
                    .human("⏳ Running cargo build (this may take a few minutes)...")
                    .emit();

                let output = cmd.output()?;

                if output.status.success() {
                    NARRATE.action("queen_rebuild").human("✅ Build successful!").emit();

                    // Show binary location
                    let binary_path = "target/release/queen-rbee";
                    NARRATE
                        .action("queen_rebuild")
                        .context(binary_path)
                        .human("📦 Binary available at: {}")
                        .emit();

                    if with_local_hive {
                        NARRATE
                            .action("queen_rebuild")
                            .human("💡 Restart queen to use the new binary with local-hive feature")
                            .emit();
                    }
                    Ok(())
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    NARRATE
                        .action("queen_rebuild")
                        .context(stderr.to_string())
                        .human("❌ Build failed: {}")
                        .error_kind("build_failed")
                        .emit();
                    anyhow::bail!("Build failed");
                }
            }
            QueenAction::Info => {
                // TEAM-262: Query queen's /v1/build-info endpoint
                let client = reqwest::Client::builder()
                    .timeout(tokio::time::Duration::from_secs(5))
                    .build()?;

                match client.get(format!("{}/v1/build-info", queen_url)).send().await {
                    Ok(response) if response.status().is_success() => {
                        NARRATE.action("queen_info").human("📋 Queen build information:").emit();
                        if let Ok(body) = response.text().await {
                            println!("{}", body);
                        }
                        Ok(())
                    }
                    Err(_) => {
                        NARRATE
                            .action("queen_info")
                            .human("❌ Queen is not running or /v1/build-info not available")
                            .emit();
                        Ok(())
                    }
                    _ => {
                        NARRATE.action("queen_info").human("⚠️  Failed to get build info").emit();
                        Ok(())
                    }
                }
            }
            QueenAction::Install { binary } => {
                // TEAM-262: Install queen binary
                // TEAM-263: Implemented install logic
                NARRATE.action("queen_install").human("📦 Installing queen-rbee...").emit();

                // Resolve binary path
                let source_path = if let Some(path) = binary {
                    NARRATE
                        .action("queen_install")
                        .context(&path)
                        .human("Using provided binary: {}")
                        .emit();
                    std::path::PathBuf::from(path)
                } else {
                    // Auto-detect: try target/release first, then target/debug
                    let release_path = std::path::PathBuf::from("target/release/queen-rbee");
                    let debug_path = std::path::PathBuf::from("target/debug/queen-rbee");

                    if release_path.exists() {
                        NARRATE
                            .action("queen_install")
                            .context("target/release/queen-rbee")
                            .human("Found binary: {}")
                            .emit();
                        release_path
                    } else if debug_path.exists() {
                        NARRATE
                            .action("queen_install")
                            .context("target/debug/queen-rbee")
                            .human("Found binary: {}")
                            .emit();
                        debug_path
                    } else {
                        NARRATE
                            .action("queen_install")
                            .human("❌ No binary found. Run 'rbee-keeper queen rebuild' first")
                            .error_kind("binary_not_found")
                            .emit();
                        anyhow::bail!("Binary not found");
                    }
                };

                // Verify binary exists
                if !source_path.exists() {
                    NARRATE
                        .action("queen_install")
                        .context(source_path.display().to_string())
                        .human("❌ Binary not found: {}")
                        .error_kind("binary_not_found")
                        .emit();
                    anyhow::bail!("Binary not found");
                }

                // Determine install location (~/.local/bin/queen-rbee)
                let home = std::env::var("HOME")?;
                let install_dir = std::path::PathBuf::from(format!("{}/.local/bin", home));
                let install_path = install_dir.join("queen-rbee");

                // Create install directory if needed
                std::fs::create_dir_all(&install_dir)?;

                // Copy binary
                NARRATE
                    .action("queen_install")
                    .context(install_path.display().to_string())
                    .human("📋 Installing to: {}")
                    .emit();

                std::fs::copy(&source_path, &install_path)?;

                // Make executable (Unix only)
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mut perms = std::fs::metadata(&install_path)?.permissions();
                    perms.set_mode(0o755);
                    std::fs::set_permissions(&install_path, perms)?;
                }

                NARRATE.action("queen_install").human("✅ Queen installed successfully!").emit();
                NARRATE
                    .action("queen_install")
                    .context(install_path.display().to_string())
                    .human("📍 Binary location: {}")
                    .emit();
                NARRATE
                    .action("queen_install")
                    .human("💡 Make sure ~/.local/bin is in your PATH")
                    .emit();

                Ok(())
            }
            QueenAction::Uninstall => {
                // TEAM-262: Uninstall queen binary
                // TEAM-263: Implemented uninstall logic
                NARRATE.action("queen_uninstall").human("🗑️  Uninstalling queen-rbee...").emit();

                // Determine install location
                let home = std::env::var("HOME")?;
                let install_path =
                    std::path::PathBuf::from(format!("{}/.local/bin/queen-rbee", home));

                // Check if binary exists
                if !install_path.exists() {
                    NARRATE
                        .action("queen_uninstall")
                        .context(install_path.display().to_string())
                        .human("⚠️  Queen not installed at: {}")
                        .emit();
                    return Ok(());
                }

                // Check if queen is running
                let client = reqwest::Client::builder()
                    .timeout(tokio::time::Duration::from_secs(2))
                    .build()?;

                let is_running = matches!(
                    client.get(format!("{}/health", queen_url)).send().await,
                    Ok(response) if response.status().is_success()
                );

                if is_running {
                    NARRATE
                        .action("queen_uninstall")
                        .human("⚠️  Queen is currently running. Stop it first with: rbee-keeper queen stop")
                        .emit();
                    anyhow::bail!("Queen is running");
                }

                // Remove binary
                std::fs::remove_file(&install_path)?;

                NARRATE
                    .action("queen_uninstall")
                    .human("✅ Queen uninstalled successfully!")
                    .emit();
                NARRATE
                    .action("queen_uninstall")
                    .context(install_path.display().to_string())
                    .human("🗑️  Removed: {}")
                    .emit();

                Ok(())
            }
        },

        Commands::Hive { action } => {
            // TEAM-194: Use alias-based operations (config from hives.conf)
            // TEAM-196: Added RefreshCapabilities command
            // TEAM-263: Added smart prompt for localhost hive install

            // Special handling for HiveInstall on localhost
            if let HiveAction::Install { ref alias } = action {
                if alias == "localhost" {
                    // Check queen's build configuration
                    let check_client = reqwest::Client::builder()
                        .timeout(tokio::time::Duration::from_secs(3))
                        .build()?;

                    if let Ok(response) =
                        check_client.get(format!("{}/v1/build-info", queen_url)).send().await
                    {
                        if response.status().is_success() {
                            if let Ok(body) = response.text().await {
                                // Parse JSON to check for local-hive feature
                                if let Ok(build_info) =
                                    serde_json::from_str::<serde_json::Value>(&body)
                                {
                                    let features = build_info["features"].as_array();
                                    let has_local_hive = features
                                        .map(|f| f.iter().any(|v| v.as_str() == Some("local-hive")))
                                        .unwrap_or(false);

                                    if !has_local_hive {
                                        // PROMPT USER!
                                        eprintln!("\n⚠️  Performance Notice:");
                                        eprintln!();
                                        eprintln!("   You're installing a hive on localhost, but your queen-rbee");
                                        eprintln!("   was built without the 'local-hive' feature.");
                                        eprintln!();
                                        eprintln!("   📊 Performance comparison:");
                                        eprintln!(
                                            "      • Current setup:  ~5-10ms overhead (HTTP)"
                                        );
                                        eprintln!("      • Integrated:     ~0.1ms overhead (direct calls)");
                                        eprintln!("      • Speedup:        50-100x faster");
                                        eprintln!();
                                        eprintln!("   💡 Recommendation:");
                                        eprintln!("      Rebuild queen-rbee with integrated hive for localhost:");
                                        eprintln!();
                                        eprintln!(
                                            "      $ rbee-keeper queen rebuild --with-local-hive"
                                        );
                                        eprintln!("      $ rbee-keeper queen stop");
                                        eprintln!("      $ rbee-keeper queen start");
                                        eprintln!();
                                        eprintln!("   ℹ️  Or continue with distributed setup if you have specific needs.");
                                        eprintln!();

                                        // Ask user
                                        eprint!("   Continue with distributed setup? [y/N]: ");
                                        use std::io::Write;
                                        std::io::stdout().flush()?;

                                        let mut input = String::new();
                                        std::io::stdin().read_line(&mut input)?;

                                        if !matches!(
                                            input.trim().to_lowercase().as_str(),
                                            "y" | "yes"
                                        ) {
                                            eprintln!("\n✋ Installation cancelled.");
                                            eprintln!("   Run: rbee-keeper queen rebuild --with-local-hive");
                                            return Ok(());
                                        }
                                        eprintln!(); // Add spacing
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let operation = match action {
                HiveAction::SshTest { alias } => Operation::SshTest { alias },
                HiveAction::Install { alias } => Operation::HiveInstall { alias },
                HiveAction::Uninstall { alias } => Operation::HiveUninstall { alias },
                HiveAction::Start { alias } => Operation::HiveStart { alias },
                HiveAction::Stop { alias } => Operation::HiveStop { alias },
                HiveAction::List => Operation::HiveList,
                HiveAction::Get { alias } => Operation::HiveGet { alias },
                HiveAction::Status { alias } => Operation::HiveStatus { alias },
                HiveAction::RefreshCapabilities { alias } => {
                    Operation::HiveRefreshCapabilities { alias }
                }
                HiveAction::ImportSsh { ssh_config, default_port } => {
                    // Expand ~ to home directory
                    let ssh_config_path = if ssh_config.starts_with("~/") {
                        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
                        ssh_config.replacen("~", &home, 1)
                    } else {
                        ssh_config
                    };
                    Operation::HiveImportSsh { ssh_config_path, default_hive_port: default_port }
                }
            };
            submit_and_stream_job(&queen_url, operation).await
        }

        Commands::Worker { hive_id, action } => {
            // TEAM-186: Use typed Operation enum instead of JSON strings
            // TEAM-187: Match on &action to avoid cloning hive_id multiple times
            let operation = match &action {
                WorkerAction::Spawn { model, device } => {
                    // Parse device string (e.g., "cuda:0" -> worker="cuda", device=0)
                    let (worker, device_id) = if device.contains(':') {
                        let parts: Vec<&str> = device.split(':').collect();
                        let worker_type = parts[0].to_string();
                        let device_num =
                            parts.get(1).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                        (worker_type, device_num)
                    } else {
                        // If no colon, assume device 0
                        (device.clone(), 0)
                    };

                    Operation::WorkerSpawn {
                        hive_id,
                        model: model.clone(),
                        worker,
                        device: device_id,
                    }
                }
                WorkerAction::List => Operation::WorkerList { hive_id },
                WorkerAction::Get { id } => Operation::WorkerGet { hive_id, id: id.clone() },
                WorkerAction::Delete { id } => Operation::WorkerDelete { hive_id, id: id.clone() },
            };
            submit_and_stream_job(&queen_url, operation).await
        }

        Commands::Model { hive_id, action } => {
            // TEAM-186: Use typed Operation enum instead of JSON strings
            // TEAM-187: Match on &action to avoid cloning hive_id multiple times
            let operation = match &action {
                ModelAction::Download { model } => {
                    Operation::ModelDownload { hive_id, model: model.clone() }
                }
                ModelAction::List => Operation::ModelList { hive_id },
                ModelAction::Get { id } => Operation::ModelGet { hive_id, id: id.clone() },
                ModelAction::Delete { id } => Operation::ModelDelete { hive_id, id: id.clone() },
            };
            submit_and_stream_job(&queen_url, operation).await
        }

        Commands::Infer {
            hive_id,
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            device,
            worker_id,
            stream,
        } => {
            // TEAM-186: Use typed Operation enum instead of JSON strings
            // TEAM-187: Eliminated all clones by moving owned values directly
            let operation = Operation::Infer {
                hive_id,
                model,
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                device,
                worker_id,
                stream,
            };
            submit_and_stream_job(&queen_url, operation).await
        }
    }
}
