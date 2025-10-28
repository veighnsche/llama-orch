//! SSH and SCP helper functions for remote operations
//!
//! Created by: TEAM-330
//!
//! # Purpose
//! Provides reusable SSH and SCP operations for all remote-daemon-lifecycle functions.
//! These helpers are used by start, stop, install, uninstall, and shutdown operations.
//!
//! # Design
//! - Simple, focused functions
//! - Clear error messages
//! - Uses tokio::process::Command for async operations
//! - Returns stdout for ssh_exec, () for scp_upload
//!
//! # Usage
//! ```rust,ignore
//! use crate::ssh::{ssh_exec, scp_upload};
//!
//! // Execute SSH command
//! let output = ssh_exec(&ssh_config, "ls -la ~/.local/bin").await?;
//!
//! // Upload file via SCP
//! scp_upload(&ssh_config, &local_path, "~/.local/bin/daemon").await?;
//! ```

use super::local::{local_copy, local_exec};
use crate::SshConfig;
use anyhow::{Context, Result};
use std::path::PathBuf; // TEAM-331: Localhost bypass

/// Execute SSH command on remote machine
///
/// TEAM-330: Shared helper function for SSH operations
/// TEAM-331: Bypasses SSH for localhost operations
///
/// # Arguments
/// * `ssh_config` - SSH connection configuration
/// * `command` - Command to execute on remote machine
///
/// # Returns
/// * `Ok(String)` - stdout from command
/// * `Err` - SSH connection failed or command failed
///
/// # Example
/// ```rust,ignore
/// let output = ssh_exec(&ssh_config, "whoami").await?;
/// println!("Remote user: {}", output.trim());
/// ```
pub async fn ssh_exec(ssh_config: &SshConfig, command: &str) -> Result<String> {
    // TEAM-331: Bypass SSH for localhost
    if ssh_config.is_localhost() {
        return local_exec(command).await;
    }

    use tokio::process::Command;

    let output = Command::new("ssh")
        .arg("-p")
        .arg(ssh_config.port.to_string())
        .arg(format!("{}@{}", ssh_config.user, ssh_config.hostname))
        .arg(command)
        .output()
        .await
        .context("Failed to execute SSH command")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("SSH command failed: {}", stderr);
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Upload file to remote machine via SCP
///
/// TEAM-330: Shared helper function for SCP operations
/// TEAM-331: Bypasses SCP for localhost operations
///
/// # Arguments
/// * `ssh_config` - SSH connection configuration
/// * `local_path` - Path to local file to upload
/// * `remote_path` - Destination path on remote machine
///
/// # Returns
/// * `Ok(())` - File uploaded successfully
/// * `Err` - SCP connection failed or upload failed
///
/// # Example
/// ```rust,ignore
/// let local = PathBuf::from("target/release/daemon");
/// scp_upload(&ssh_config, &local, "~/.local/bin/daemon").await?;
/// ```
pub async fn scp_upload(
    ssh_config: &SshConfig,
    local_path: &PathBuf,
    remote_path: &str,
) -> Result<()> {
    // TEAM-331: Bypass SCP for localhost
    if ssh_config.is_localhost() {
        return local_copy(local_path, remote_path).await;
    }

    use tokio::process::Command;

    let remote_target = format!("{}@{}:{}", ssh_config.user, ssh_config.hostname, remote_path);

    let output = Command::new("scp")
        .arg("-P")
        .arg(ssh_config.port.to_string())
        .arg(local_path)
        .arg(&remote_target)
        .output()
        .await
        .context("Failed to execute SCP command")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("SCP failed: {}", stderr);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These are integration tests that require actual SSH access
    // They are marked as ignored by default

    #[tokio::test]
    #[ignore]
    async fn test_ssh_exec() {
        let ssh = SshConfig::new("localhost".to_string(), "test".to_string(), 22);
        let result = ssh_exec(&ssh, "echo 'hello'").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().trim(), "hello");
    }

    #[tokio::test]
    #[ignore]
    async fn test_scp_upload() {
        let ssh = SshConfig::new("localhost".to_string(), "test".to_string(), 22);
        let local = PathBuf::from("/tmp/test.txt");
        let result = scp_upload(&ssh, &local, "/tmp/test_remote.txt").await;
        assert!(result.is_ok());
    }
}
