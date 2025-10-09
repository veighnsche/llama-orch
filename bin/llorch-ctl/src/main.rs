//! Orchestrator Control CLI - Remote pool control via SSH
//!
//! Created by: TEAM-022

mod cli;
mod commands;
mod ssh;

use anyhow::Result;

fn main() -> Result<()> {
    let cli = cli::Cli::parse_args();
    cli::handle_command(cli)
}
