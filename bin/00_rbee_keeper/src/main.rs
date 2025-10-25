//! rbee-keeper - Thin HTTP client for queen-rbee
//!
//! TEAM-151: Migrated CLI from old.rbee-keeper to numbered architecture
//! TEAM-158: Cleaned up over-engineering - rbee-keeper is just a thin HTTP client!
//! TEAM-216: Investigated - Complete behavior inventory created
//! TEAM-276: Refactored into modular structure for maintainability
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
//! TEAM-185: Renamed actions module to operations_contract
//! TEAM-185: Updated all operation strings to use constants

mod cli;
mod config;
mod handlers;
mod job_client;

use anyhow::Result;
use clap::Parser;
use config::Config;

use cli::{Cli, Commands};
use handlers::{
    handle_hive, handle_infer, handle_model, handle_queen,
    handle_status, handle_worker,
};
// TEAM-284: DELETED handle_migrate, handle_package_status, handle_sync, handle_validate

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // TEAM-295: If no subcommand provided, launch Tauri GUI instead
    if cli.command.is_none() {
        launch_gui();
        return Ok(());
    }
    
    handle_command(cli).await
}

// TEAM-295: Launch Tauri GUI (synchronous, blocks until window closes)
fn launch_gui() {
    use rbee_keeper::tauri_commands::*;
    
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            // Status
            get_status,
            // Queen commands
            queen_start,
            queen_stop,
            queen_status,
            queen_rebuild,
            queen_info,
            queen_install,
            queen_uninstall,
            // Hive commands
            hive_install,
            hive_uninstall,
            hive_start,
            hive_stop,
            hive_list,
            hive_get,
            hive_status,
            hive_refresh_capabilities,
            // Worker commands
            worker_spawn,
            worker_process_list,
            worker_process_get,
            worker_process_delete,
            // Model commands
            model_download,
            model_list,
            model_get,
            model_delete,
            // Inference
            infer,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

async fn handle_command(cli: Cli) -> Result<()> {
    let config = Config::load()?;
    let queen_url = config.queen_url();

    // ============================================================================
    // COMMAND ROUTING
    // ============================================================================
    //
    // For each CLI command, delegate to the appropriate handler module.
    // All business logic lives in the handlers/ directory.
    //
    // Pattern:
    //   1. Extract CLI arguments
    //   2. Call handler::handle_xxx()
    //   3. Handler constructs Operation and submits to queen
    // ============================================================================
    
    // TEAM-295: Command is now Option, unwrap it here since we checked is_some earlier
    let command = cli.command.expect("Command should be Some if we reach here");

    match command {
        Commands::Status => handle_status(&queen_url).await,
        Commands::Queen { action } => handle_queen(action, &queen_url).await,
        Commands::Hive { action } => handle_hive(action, &queen_url).await,
        Commands::Worker { hive_id, action } => handle_worker(hive_id, action, &queen_url).await,
        Commands::Model { hive_id, action } => handle_model(hive_id, action, &queen_url).await,
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
            handle_infer(
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
                &queen_url,
            )
            .await
        }

        // ========================================================================
        // TEAM-284: DELETED PACKAGE MANAGER COMMANDS
        // ========================================================================
        // Sync, PackageStatus, Validate, Migrate removed (SSH/remote operations)
    }
}
