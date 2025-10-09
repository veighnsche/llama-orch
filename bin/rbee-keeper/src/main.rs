//! Orchestrator Control CLI - Remote pool control via SSH
//!
//! TEAM-024 CLARIFICATION:
//! This is `llorch` - the CLI tool for OPERATORS (humans)
//! It uses SSH to control remote pools
//! 
//! This is NOT the orchestrator daemon!
//! The orchestrator HTTP daemon is `orchestratord` (separate binary, M2)
//! 
//! Binary: `llorch` (this file)
//! Purpose: SSH-based remote pool control for operators
//! Protocol: SSH
//! 
//! vs.
//! 
//! Binary: `orchestratord` (not yet built, M2)
//! Purpose: HTTP daemon that routes inference requests
//! Protocol: HTTP
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
