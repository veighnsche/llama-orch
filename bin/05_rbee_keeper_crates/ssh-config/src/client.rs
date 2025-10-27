//! SSH client for remote command execution
//!
//! TEAM-314: Migrated from hive-lifecycle to shared ssh-config crate
//!
//! This module provides an SSH client that:
//! - Uses host SSH config (~/.ssh/config)
//! - Executes commands on remote hosts
//! - Uploads/downloads files via SCP
//! - Provides narration for all operations

use anyhow::{Context, Result};
use observability_narration_core::n;
use std::path::Path;
use tokio::process::Command;

/// SSH client that uses host SSH config
///
/// TEAM-290: Piggybacks on ~/.ssh/config and ~/.ssh/id_rsa
/// TEAM-314: Migrated to shared ssh-config crate
///
/// # Example
///
/// ```rust,ignore
/// use ssh_config::SshClient;
///
/// let client = SshClient::connect("workstation").await?;
/// let output = client.execute("uname -a").await?;
/// println!("Remote OS: {}", output);
/// ```
pub struct SshClient {
    /// SSH host alias (from ~/.ssh/config)
    host: String,
}

impl SshClient {
    /// Connect to remote host using SSH config
    ///
    /// # Arguments
    /// * `host` - SSH host alias from ~/.ssh/config
    ///
    /// # Example
    /// ```rust,ignore
    /// let client = SshClient::connect("gpu-server").await?;
    /// ```
    pub async fn connect(host: &str) -> Result<Self> {
        n!("ssh_connect", "🔌 Connecting to SSH host '{}'", host);

        // Verify SSH connection works
        let output = Command::new("ssh")
            .arg("-o")
            .arg("ConnectTimeout=5")
            .arg("-o")
            .arg("BatchMode=yes") // No password prompts
            .arg(host)
            .arg("echo")
            .arg("connected")
            .output()
            .await
            .context("Failed to execute SSH command")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "SSH connection to '{}' failed:\n{}\n\n\
                 Make sure:\n\
                 1. Host is in ~/.ssh/config\n\
                 2. SSH key is configured\n\
                 3. Public key is on remote machine (~/.ssh/authorized_keys)",
                host,
                stderr
            );
        }

        n!("ssh_connected", "✅ Connected to '{}'", host);

        Ok(Self {
            host: host.to_string(),
        })
    }

    /// Execute command on remote host
    ///
    /// # Arguments
    /// * `command` - Command to execute
    ///
    /// # Returns
    /// * `Ok(String)` - Command output (stdout)
    /// * `Err` - If command fails
    ///
    /// # Example
    /// ```rust,ignore
    /// let output = client.execute("ls -la").await?;
    /// println!("{}", output);
    /// ```
    pub async fn execute(&self, command: &str) -> Result<String> {
        n!("ssh_exec", "🔧 Executing on '{}': {}", self.host, command);

        let output = Command::new("ssh")
            .arg(&self.host)
            .arg(command)
            .output()
            .await
            .context("Failed to execute SSH command")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Command failed on '{}':\n{}", self.host, stderr);
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();

        n!("ssh_exec_complete", "✅ Command completed on '{}'", self.host);

        Ok(stdout)
    }

    /// Upload file to remote host
    ///
    /// # Arguments
    /// * `local_path` - Local file path
    /// * `remote_path` - Remote file path
    ///
    /// # Example
    /// ```rust,ignore
    /// client.upload_file("./binary", "/usr/local/bin/binary").await?;
    /// ```
    pub async fn upload_file(&self, local_path: &str, remote_path: &str) -> Result<()> {
        n!("ssh_upload", "📤 Uploading '{}' to '{}' on '{}'", local_path, remote_path, self.host);

        // Verify local file exists
        if !Path::new(local_path).exists() {
            anyhow::bail!("Local file not found: {}", local_path);
        }

        // Use scp to upload
        let output = Command::new("scp")
            .arg(local_path)
            .arg(format!("{}:{}", self.host, remote_path))
            .output()
            .await
            .context("Failed to execute scp command")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Upload failed:\n{}", stderr);
        }

        n!("ssh_upload_complete", "✅ Upload complete to '{}'", self.host);

        Ok(())
    }

    /// Download file from remote host
    ///
    /// # Arguments
    /// * `remote_path` - Remote file path
    /// * `local_path` - Local file path
    ///
    /// # Example
    /// ```rust,ignore
    /// client.download_file("/var/log/app.log", "./app.log").await?;
    /// ```
    pub async fn download_file(&self, remote_path: &str, local_path: &str) -> Result<()> {
        n!("ssh_download", "📥 Downloading '{}' from '{}' to '{}'", remote_path, self.host, local_path);

        let output = Command::new("scp")
            .arg(format!("{}:{}", self.host, remote_path))
            .arg(local_path)
            .output()
            .await
            .context("Failed to execute scp command")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Download failed:\n{}", stderr);
        }

        n!("ssh_download_complete", "✅ Download complete from '{}'", self.host);

        Ok(())
    }

    /// Check if remote file exists
    ///
    /// # Arguments
    /// * `remote_path` - Remote file path
    ///
    /// # Returns
    /// * `Ok(true)` - File exists
    /// * `Ok(false)` - File does not exist
    ///
    /// # Example
    /// ```rust,ignore
    /// if client.file_exists("/usr/local/bin/app").await? {
    ///     println!("App is installed");
    /// }
    /// ```
    pub async fn file_exists(&self, remote_path: &str) -> Result<bool> {
        let output = Command::new("ssh")
            .arg(&self.host)
            .arg("test")
            .arg("-f")
            .arg(remote_path)
            .output()
            .await
            .context("Failed to execute SSH command")?;

        Ok(output.status.success())
    }

    /// Get remote host name
    ///
    /// # Returns
    /// The SSH host alias
    pub fn host(&self) -> &str {
        &self.host
    }
}
