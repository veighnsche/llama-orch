//! rbee-keeper library - Shared code for CLI and GUI
//!
//! TEAM-293: Created library to support both CLI and Tauri GUI modes
//!
//! This library exposes all the core functionality that both the CLI
//! and GUI interfaces can use.

pub mod cli;
pub mod config;
pub mod handlers;
pub mod job_client;
pub mod platform; // TEAM-293: Cross-platform abstraction layer
pub mod ssh_resolver; // TEAM-332: SSH config resolver middleware
pub mod tracing_init; // TEAM-336: Consolidated tracing setup for CLI and GUI

// Re-export commonly used types
pub use config::Config;
pub use handlers::{
    handle_hive, handle_infer, handle_model, handle_queen, handle_status, handle_worker,
};

// TEAM-293: Tauri-specific module (always available for GUI binary)
pub mod process_utils;
pub mod tauri_commands; // TEAM-301: Process output streaming utilities

// TEAM-336: Re-export tracing init functions for convenience
pub use tracing_init::{init_cli_tracing, init_gui_tracing, NarrationEvent};
