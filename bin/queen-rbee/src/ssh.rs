//! SSH Connection Validation
//!
//! Created by: TEAM-043
//!
//! Validates SSH connections to remote rbee-hive nodes

use anyhow::{Context, Result};
use std::process::Stdio;
use tokio::process::Command;
use tracing::{info, error};

/// Test SSH connection to a remote host
pub async fn test_ssh_connection(
    host: &str,
    port: u16,
    user: &str,
    key_path: Option<&str>,
) -> Result<bool> {
    info!("🔌 Testing SSH connection to {}@{}:{}", user, host, port);

    let mut cmd = Command::new("ssh");
    
    // Add key if provided
    if let Some(key) = key_path {
        cmd.arg("-i").arg(key);
    }

    // SSH options
    cmd.arg("-o").arg("ConnectTimeout=10")
        .arg("-o").arg("BatchMode=yes")
        .arg("-o").arg("StrictHostKeyChecking=no")
        .arg("-p").arg(port.to_string())
        .arg(format!("{}@{}", user, host))
        .arg("echo 'connection test'")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = cmd.output().await
        .context("Failed to execute SSH command")?;

    if output.status.success() {
        info!("✅ SSH connection successful!");
        Ok(true)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        error!("❌ SSH connection failed: {}", stderr);
        Ok(false)
    }
}

/// Execute a command on a remote host via SSH
pub async fn execute_remote_command(
    host: &str,
    port: u16,
    user: &str,
    key_path: Option<&str>,
    command: &str,
) -> Result<(bool, String, String)> {
    info!("📡 Executing remote command on {}@{}: {}", user, host, command);

    let mut cmd = Command::new("ssh");
    
    if let Some(key) = key_path {
        cmd.arg("-i").arg(key);
    }

    cmd.arg("-o").arg("ConnectTimeout=10")
        .arg("-o").arg("BatchMode=yes")
        .arg("-o").arg("StrictHostKeyChecking=no")
        .arg("-p").arg(port.to_string())
        .arg(format!("{}@{}", user, host))
        .arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let output = cmd.output().await
        .context("Failed to execute SSH command")?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok((output.status.success(), stdout, stderr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires SSH setup
    async fn test_ssh_connection_localhost() {
        // This test requires SSH server running on localhost
        let result = test_ssh_connection("localhost", 22, "testuser", None).await;
        // We don't assert success because it depends on SSH setup
        assert!(result.is_ok());
    }
}
