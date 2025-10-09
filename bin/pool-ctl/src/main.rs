//! Pool Control CLI - Local pool management
//!
//! Created by: TEAM-022

mod cli;
mod commands;

use anyhow::Result;

fn main() -> Result<()> {
    let cli = cli::Cli::parse_args();
    cli::handle_command(cli)
}
