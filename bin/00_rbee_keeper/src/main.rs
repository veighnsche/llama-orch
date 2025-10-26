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
mod process_utils; // TEAM-301: Process output streaming

use anyhow::Result;
use clap::Parser;
use config::Config;

// TEAM-309: Custom narration formatter for clean output
// TEAM-310: Now uses centralized format_message from narration-core
use tracing_subscriber::fmt::format::{self, FormatEvent, FormatFields};
use tracing_subscriber::fmt::FmtContext;
use tracing_subscriber::registry::LookupSpan;

/// Custom formatter for narration events in CLI mode.
/// 
/// TEAM-310: This formatter now delegates to `observability_narration_core::format::format_message()`
/// for consistent formatting across all narration outputs (SSE, CLI, logs).
struct NarrationFormatter;

impl<S, N> FormatEvent<S, N> for NarrationFormatter
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        _ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        use tracing::field::{Field, Visit};
        
        // Extract fields from the event
        // TEAM-311: Added fn_name field
        struct FieldVisitor {
            actor: Option<String>,
            action: Option<String>,
            target: Option<String>,
            human: Option<String>,
            fn_name: Option<String>,
        }
        
        impl Visit for FieldVisitor {
            fn record_str(&mut self, field: &Field, value: &str) {
                match field.name() {
                    "actor" => self.actor = Some(value.to_string()),
                    "action" => self.action = Some(value.to_string()),
                    "target" => self.target = Some(value.to_string()),
                    "human" => self.human = Some(value.to_string()),
                    "fn_name" => self.fn_name = Some(value.to_string()),
                    _ => {}
                }
            }
            
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                match field.name() {
                    "actor" => self.actor = Some(format!("{:?}", value).trim_matches('"').to_string()),
                    "action" => self.action = Some(format!("{:?}", value).trim_matches('"').to_string()),
                    "target" => self.target = Some(format!("{:?}", value).trim_matches('"').to_string()),
                    "human" => self.human = Some(format!("{:?}", value).trim_matches('"').to_string()),
                    "fn_name" => self.fn_name = Some(format!("{:?}", value).trim_matches('"').to_string()),
                    _ => {}
                }
            }
        }
        
        let mut visitor = FieldVisitor {
            actor: None,
            action: None,
            target: None,
            human: None,
            fn_name: None,
        };
        
        event.record(&mut visitor);
        
        // TEAM-310: Use centralized format_message from narration-core
        // TEAM-311: Now uses format_message_with_fn to show function names
        // TEAM-312: Removed actor from formatting - fn_name provides full trace
        // Format: Bold fn_name (40 chars), dimmed action (20 chars), message on second line
        if let (Some(action), Some(human)) = (visitor.action, visitor.human) {
            // TEAM-311: Use format_message_with_fn to include function name
            let formatted = observability_narration_core::format::format_message_with_fn(
                &action, 
                &human,
                visitor.fn_name.as_deref().unwrap_or("unknown")
            );
            write!(writer, "{}", formatted)
        } else {
            // Fallback for non-narration events
            writeln!(writer, "{:?}", event)
        }
    }
}

use cli::{Cli, Commands};
use handlers::{
    handle_hive, handle_infer, handle_model, handle_queen,
    handle_self_check, handle_status, handle_worker,
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
    // ============================================================
    // BUG FIX: TEAM-309 | Narration not visible in CLI mode
    // ============================================================
    // SUSPICION:
    // - Initially thought narration macros were broken
    // - Suspected n!() macro wasn't calling the right functions
    //
    // INVESTIGATION:
    // - Read debugging-rules.md (mandatory before fixing bugs)
    // - Ran self-check and saw NO narration output
    // - Traced code path: n!() → macro_emit() → narrate() → narrate_at_level()
    // - Found TEAM-299 removed stderr output for multi-tenant privacy (emit.rs:77-103)
    // - Checked privacy_isolation_tests.rs - confirms NO stderr by design
    // - Discovered narration only goes to: SSE sink, Tracing, Capture adapter
    // - self-check has NO SSE channel, NO job_id, NO tracing subscriber
    //
    // ROOT CAUSE:
    // - TEAM-299 removed stderr for security (correct for queen-rbee server)
    // - rbee-keeper CLI has NO tracing subscriber configured
    // - Narration is emitted but goes nowhere visible
    // - Not a bug in narration-core, missing integration in rbee-keeper
    //
    // FIX:
    // - Initialize tracing-subscriber for CLI mode (stdout output)
    // - Use FmtSubscriber with human-readable format
    // - Only for CLI mode (GUI uses different logging)
    // - This is safe: rbee-keeper is single-user, isolated process
    // - No privacy violation (not multi-tenant like queen-rbee)
    //
    // TESTING:
    // - Run: cargo build --bin rbee-keeper && ./target/debug/rbee-keeper self-check
    // - Expect: All narration tests show visible output
    // - Verify: 10 test narrations appear in terminal
    // - Check: Different modes (human/cute/story) display correctly
    // ============================================================
    
    // TEAM-309: Set up tracing subscriber for CLI narration visibility
    // This makes narration-core's tracing events visible to users.
    // Safe for CLI: single-user, isolated process (not multi-tenant server).
    //
    // DESIRED FORMAT (from narration-core/src/api/emit.rs:83):
    //   [actor     ] action         : message
    //
    // Example:
    //   [rbee-keeper] self_check_start: Starting rbee-keeper self-check
    use tracing_subscriber::{fmt, EnvFilter, Layer};
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    
    // Custom formatter for narration events
    let narration_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .event_format(NarrationFormatter)
        .with_filter(EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info")));
    
    tracing_subscriber::registry()
        .with(narration_layer)
        .init();
    
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
            use crate::job_client::submit_and_stream_job;
            
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
        }

        // ========================================================================
        // TEAM-284: DELETED PACKAGE MANAGER COMMANDS
        // ========================================================================
        // Sync, PackageStatus, Validate, Migrate removed (SSH/remote operations)
    }
}
