//! Pool Control CLI - Local pool management
//!
//! TEAM-024 CLARIFICATION:
//! This is `rbee-hive` - the LOCAL pool management CLI
//! It runs ON the pool machine (mac.home.arpa, workstation.home.arpa, etc.)
//! Called by `llorch` (remote CLI) via SSH
//!
//! Binary: `rbee-hive` (this file)
//! Purpose: Local pool operations (models, workers, status)
//! Location: Runs on pool machines
//!
//! Created by: TEAM-022
//! Modified by: TEAM-027, TEAM-029

mod cli;
mod commands;
mod http;
mod monitor;
mod provisioner;  // TEAM-029: Model provisioner
mod registry;
mod timeout;

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = cli::Cli::parse_args();
    cli::handle_command(cli).await
}
