//! SSH client wrapper
//!
//! Created by: TEAM-022

use anyhow::Result;
use std::process::Command;

/// Execute a command on a remote host via SSH (returns output)
#[allow(dead_code)] // TEAM-022: Will be used in future checkpoints
pub fn execute_remote_command(host: &str, command: &str) -> Result<String> {
    let output = Command::new("ssh")
        .arg(host)
        .arg(command)
        .output()?;

    if !output.status.success() {
        anyhow::bail!(
            "SSH command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(String::from_utf8(output.stdout)?)
}

/// Execute a command on a remote host and stream output
pub fn execute_remote_command_streaming(host: &str, command: &str) -> Result<()> {
    let status = Command::new("ssh").arg(host).arg(command).status()?;

    if !status.success() {
        anyhow::bail!("SSH command failed");
    }

    Ok(())
}
