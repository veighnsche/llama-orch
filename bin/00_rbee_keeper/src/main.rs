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

// TEAM-332: Use library modules instead of redefining them
// TEAM-334: Removed unused config import
use rbee_keeper::{cli, handlers, Config};

use anyhow::Result;
use clap::Parser;

// TEAM-334: DELETED NarrationFormatter - RULE ZERO VIOLATION
// Formatting belongs in narration-core/src/format.rs, not in main.rs
// Use standard tracing compact formatter instead

use cli::{Cli, Commands};
use handlers::{
    handle_hive, handle_infer, handle_model, handle_queen, handle_self_check, handle_status,
    handle_worker,
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
// TEAM-334: Cleaned up - only ssh_list command active, rest to be re-implemented
// TEAM-336: Narration events stream to React sidebar via Tauri events
//
// TEAM-334 NOTE: Desktop entry integration
// - This function is called when rbee-keeper is launched with no arguments
// - Desktop entry: ~/.local/share/applications/rbee-dev.desktop
// - Flow: Desktop entry → ./rbee → xtask rbee → rbee-keeper (no args) → launch_gui()
// - Status: ⚠️ NOT WORKING YET from desktop entry (GUI doesn't appear)
// - Works fine when launched directly: ./rbee
// - Possible issues: Display environment, Wayland/X11, permissions
// - See: DESKTOP_ENTRY.md for debugging
fn launch_gui() {
    use rbee_keeper::tauri_commands::*;

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            // TEAM-336: Test command for narration
            test_narration,
            // TEAM-333: SSH list command
            ssh_list,
            // TEAM-367: Get installed hives
            get_installed_hives,
            // TEAM-338: SSH config editor
            ssh_open_config,
            // TEAM-335: Queen lifecycle commands (thin wrappers)
            queen_start,
            queen_stop,
            queen_status,
            queen_install,
            queen_rebuild,
            queen_uninstall,
            // TEAM-338: Hive lifecycle commands
            hive_start,
            hive_stop,
            hive_status,
            hive_install,
            hive_uninstall,
            hive_rebuild,
        ])
        .setup(|app| {
            // TEAM-336: Initialize tracing with Tauri event streaming
            // Events emitted on "narration" channel for React sidebar
            rbee_keeper::init_gui_tracing(app.handle().clone());
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

async fn handle_command(cli: Cli) -> Result<()> {
    // ============================================================
    // TEAM-336: CLI tracing setup (stderr only)
    // ============================================================
    rbee_keeper::init_cli_tracing();

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
        Commands::SelfCheck => handle_self_check().await,
        Commands::QueenCheck => {
            // TEAM-312: Deep narration test through queen job server
            use operations_contract::Operation;
            use rbee_keeper::job_client::submit_and_stream_job;

            submit_and_stream_job(&queen_url, Operation::QueenCheck).await
        }
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
        } // ========================================================================
          // TEAM-284: DELETED PACKAGE MANAGER COMMANDS
          // ========================================================================
          // Sync, PackageStatus, Validate, Migrate removed (SSH/remote operations)
    }
}
