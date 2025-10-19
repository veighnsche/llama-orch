//! SSH client wrapper
//!
//! Created by: TEAM-022
//! Refactored by: TEAM-022 (using system SSH for now - ssh2 auth issues with config)
//! Modified by: TEAM-023 (removed dead ssh2 code with missing imports)

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::process::Command;

/// Execute a command on a remote host and stream output to stdout
///
/// TEAM-023: Uses system SSH to respect ~/.ssh/config (User, IdentityFile, etc.)
pub fn execute_remote_command_streaming(host: &str, command: &str) -> Result<()> {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(ProgressStyle::default_spinner().template("{spinner:.cyan} {msg}").unwrap());
    spinner.set_message(format!("→ SSH to {}", host));
    spinner.finish_and_clear();

    let status = Command::new("ssh").arg(host).arg(command).status()?;

    if !status.success() {
        anyhow::bail!("SSH command failed");
    }

    Ok(())
}
