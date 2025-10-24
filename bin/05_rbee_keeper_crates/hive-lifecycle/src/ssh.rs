//! SSH client that uses host SSH config
//!
//! TEAM-290: Piggybacks on ~/.ssh/config and ~/.ssh/id_rsa

use anyhow::{Context, Result};
use observability_narration_core::NarrationFactory;
use std::path::Path;
use tokio::process::Command;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-ssh");

/// SSH client that uses host SSH config
///
/// TEAM-290: Piggybacks on ~/.ssh/config and ~/.ssh/id_rsa
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
        NARRATE
            .action("ssh_connect")
            .context(host)
            .human("🔌 Connecting to SSH host '{}'")
            .emit();

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

        NARRATE
            .action("ssh_connected")
            .context(host)
            .human("✅ Connected to '{}'")
            .emit();

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
    pub async fn execute(&self, command: &str) -> Result<String> {
        NARRATE
            .action("ssh_exec")
            .context(&self.host)
            .context(command)
            .human("🔧 Executing on '{}': {}")
            .emit();

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

        NARRATE
            .action("ssh_exec_complete")
            .context(&self.host)
            .human("✅ Command completed on '{}'")
            .emit();

        Ok(stdout)
    }

    /// Upload file to remote host
    ///
    /// # Arguments
    /// * `local_path` - Local file path
    /// * `remote_path` - Remote file path
    pub async fn upload_file(&self, local_path: &str, remote_path: &str) -> Result<()> {
        NARRATE
            .action("ssh_upload")
            .context(&self.host)
            .context(local_path)
            .context(remote_path)
            .human("📤 Uploading '{}' to '{}:{}'")
            .emit();

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

        NARRATE
            .action("ssh_upload_complete")
            .context(&self.host)
            .human("✅ Upload complete to '{}'")
            .emit();

        Ok(())
    }

    /// Download file from remote host
    ///
    /// # Arguments
    /// * `remote_path` - Remote file path
    /// * `local_path` - Local file path
    pub async fn download_file(&self, remote_path: &str, local_path: &str) -> Result<()> {
        NARRATE
            .action("ssh_download")
            .context(&self.host)
            .context(remote_path)
            .context(local_path)
            .human("📥 Downloading '{}:{}' to '{}'")
            .emit();

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

        NARRATE
            .action("ssh_download_complete")
            .context(&self.host)
            .human("✅ Download complete from '{}'")
            .emit();

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
    pub fn host(&self) -> &str {
        &self.host
    }
}
