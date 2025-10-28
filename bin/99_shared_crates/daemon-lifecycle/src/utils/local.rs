//! Local process execution helpers (bypasses SSH for localhost)
//!
//! Created by: TEAM-331
//!
//! # Purpose
//! Provides direct process execution for localhost operations, bypassing SSH overhead.
//! Used when SshConfig::is_localhost() returns true.
//!
//! # Design
//! - Direct tokio::process::Command execution
//! - Same interface as ssh.rs for easy switching
//! - No SSH/SCP overhead for local operations
//!
//! # Usage
//! ```rust,ignore
//! use crate::utils::local::{local_exec, local_copy};
//!
//! // Execute local command
//! let output = local_exec("ls -la ~/.local/bin").await?;
//!
//! // Copy file locally
//! local_copy(&local_path, "~/.local/bin/daemon").await?;
//! ```

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Execute command locally (bypasses SSH)
///
/// TEAM-331: Direct process execution for localhost operations
///
/// # Arguments
/// * `command` - Command to execute locally
///
/// # Returns
/// * `Ok(String)` - stdout from command
/// * `Err` - Command execution failed
///
/// # Example
/// ```rust,ignore
/// let output = local_exec("whoami").await?;
/// println!("Local user: {}", output.trim());
/// ```
pub async fn local_exec(command: &str) -> Result<String> {
    use tokio::process::Command;

    let output = Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .await
        .context("Failed to execute local command")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Local command failed: {}", stderr);
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Copy file locally (bypasses SCP)
///
/// TEAM-331: Direct file copy for localhost operations
///
/// # Arguments
/// * `local_path` - Source file path
/// * `dest_path` - Destination path (supports ~ expansion)
///
/// # Returns
/// * `Ok(())` - File copied successfully
/// * `Err` - Copy failed
///
/// # Example
/// ```rust,ignore
/// let local = PathBuf::from("target/release/daemon");
/// local_copy(&local, "~/.local/bin/daemon").await?;
/// ```
pub async fn local_copy(local_path: &Path, dest_path: &str) -> Result<()> {
    use tokio::fs;

    // Expand ~ in destination path
    let expanded_dest = if dest_path.starts_with('~') {
        let home = std::env::var("HOME").context("HOME environment variable not set")?;
        dest_path.replacen('~', &home, 1)
    } else {
        dest_path.to_string()
    };

    let dest = PathBuf::from(expanded_dest);

    // Create parent directory if needed
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).await.context("Failed to create destination directory")?;
    }

    // Copy file
    fs::copy(local_path, &dest).await.context("Failed to copy file")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_local_exec() {
        let result = local_exec("echo 'hello'").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().trim(), "hello");
    }

    #[tokio::test]
    async fn test_local_exec_failure() {
        let result = local_exec("false").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_local_copy() {
        // Create temp file
        let temp_dir = std::env::temp_dir();
        let src = temp_dir.join("test_src.txt");
        let dest = temp_dir.join("test_dest.txt");

        std::fs::write(&src, "test content").unwrap();

        let result = local_copy(&src, dest.to_str().unwrap()).await;
        assert!(result.is_ok());

        let content = std::fs::read_to_string(&dest).unwrap();
        assert_eq!(content, "test content");

        // Cleanup
        std::fs::remove_file(&src).ok();
        std::fs::remove_file(&dest).ok();
    }
}
