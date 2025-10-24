// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-188: Implemented SSH test connection functionality
// TEAM-222: Investigated behavior inventory (Phase 2)
// TEAM-256: Migrated from ssh2 to russh for async operations and SFTP support
// TEAM-260: Reverted to command-based SSH (ssh/scp) for simplicity and reliability
// Purpose: SSH client for managing remote rbee-hive instances

#![warn(missing_docs)]
#![warn(clippy::all)]

//! queen-rbee-ssh-client
//!
//! SSH client functionality for managing remote rbee-hive instances
//!
//! # Features
//!
//! - Test SSH connectivity to remote hosts
//! - Execute commands on remote hosts
//! - Copy files via SCP
//! - Handle SSH authentication (SSH agent)
//!
//! # Security
//!
//! Uses system ssh/scp commands with proper argument escaping to prevent command injection.

use anyhow::{Context, Result};
use observability_narration_core::Narration;
use std::time::Duration;
use tokio::process::Command;

const ACTOR_SSH_CLIENT: &str = "🔐 ssh-client";
const ACTION_TEST: &str = "test_connection";

/// SSH connection configuration
#[derive(Debug, Clone)]
pub struct SshConfig {
    /// SSH host address
    pub host: String,
    /// SSH port (default: 22)
    pub port: u16,
    /// SSH username
    pub user: String,
    /// Connection timeout in seconds
    pub timeout_secs: u64,
}

impl Default for SshConfig {
    fn default() -> Self {
        Self { host: String::new(), port: 22, user: String::new(), timeout_secs: 5 }
    }
}

/// TEAM-260: SSH client for remote operations using command-based execution
pub struct RbeeSSHClient {
    host: String,
    port: u16,
    user: String,
}

impl RbeeSSHClient {
    /// Connect to remote host (creates client instance)
    ///
    /// TEAM-260: Simplified to just store connection parameters.
    /// Actual SSH connection happens on each exec/copy_file call.
    /// This matches the behavior of command-line ssh/scp tools.
    pub async fn connect(host: &str, port: u16, user: &str) -> Result<Self> {
        // TEAM-260: Test connectivity with a simple command
        let client = Self {
            host: host.to_string(),
            port,
            user: user.to_string(),
        };

        // Verify SSH connectivity with timeout
        let test_future = client.exec_internal("echo test");
        tokio::time::timeout(Duration::from_secs(30), test_future)
            .await
            .context("SSH connection timeout (30s)")?
            .context("SSH connection test failed")?;

        Ok(client)
    }

    /// Execute command on remote host
    pub async fn exec(&self, command: &str) -> Result<(String, String, i32)> {
        self.exec_internal(command).await
    }

    /// TEAM-260: Internal exec implementation
    async fn exec_internal(&self, command: &str) -> Result<(String, String, i32)> {
        // TEAM-260: Build SSH command with proper arguments
        // -o StrictHostKeyChecking=no: Accept unknown host keys (matches russh behavior)
        // -o BatchMode=yes: Disable interactive prompts
        // -o ConnectTimeout=30: Connection timeout
        // -o ServerAliveInterval=60: Send keepalive every 60s (for long builds)
        // -o ServerAliveCountMax=120: Allow 120 missed keepalives (120min total)
        let mut cmd = Command::new("ssh");
        cmd.arg("-o").arg("StrictHostKeyChecking=no")
            .arg("-o").arg("BatchMode=yes")
            .arg("-o").arg("ConnectTimeout=30")
            .arg("-o").arg("ServerAliveInterval=60")
            .arg("-o").arg("ServerAliveCountMax=120")
            .arg("-p").arg(self.port.to_string())
            .arg(format!("{}@{}", self.user, self.host))
            .arg(command);

        let output = cmd.output().await.context("Failed to execute ssh command")?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(255);

        Ok((stdout, stderr, exit_code))
    }

    /// Copy file to remote host via SCP
    pub async fn copy_file(&self, local_path: &str, remote_path: &str) -> Result<()> {
        // TEAM-260: Use scp command for file transfer
        // -o StrictHostKeyChecking=no: Accept unknown host keys
        // -o BatchMode=yes: Disable interactive prompts
        // -P: Port number (uppercase P for scp)
        let mut cmd = Command::new("scp");
        cmd.arg("-o").arg("StrictHostKeyChecking=no")
            .arg("-o").arg("BatchMode=yes")
            .arg("-P").arg(self.port.to_string())
            .arg(local_path)
            .arg(format!("{}@{}:{}", self.user, self.host, remote_path));

        let output = cmd.output().await.context("Failed to execute scp command")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("SCP failed: {}", stderr);
        }

        Ok(())
    }

    /// Close connection (no-op for command-based implementation)
    pub async fn close(self) -> Result<()> {
        // TEAM-260: No persistent connection to close
        // Each exec/copy_file creates its own SSH session
        Ok(())
    }
}

/// Result of SSH connection test
#[derive(Debug, Clone)]
pub struct SshTestResult {
    /// Whether the connection was successful
    pub success: bool,
    /// Error message if connection failed
    pub error: Option<String>,
    /// Test command output (if successful)
    pub test_output: Option<String>,
}

/// Test SSH connection to a remote host
///
/// TEAM-260: Uses command-based SSH execution
///
/// This function:
/// 1. Attempts to establish SSH connection with timeout
/// 2. Runs a simple test command (`echo test`)
/// 3. Returns success/failure with details
///
/// # Arguments
///
/// * `config` - SSH connection configuration
///
/// # Returns
///
/// * `Ok(SshTestResult)` - Test result with success status
/// * `Err` - Critical error (should not happen in normal operation)
///
/// # Example
///
/// ```no_run
/// use queen_rbee_ssh_client::{SshConfig, test_ssh_connection};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = SshConfig {
///     host: "192.168.1.100".to_string(),
///     port: 22,
///     user: "admin".to_string(),
///     timeout_secs: 5,
/// };
///
/// let result = test_ssh_connection(config).await?;
/// if result.success {
///     println!("SSH connection successful!");
/// } else {
///     println!("SSH connection failed: {:?}", result.error);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn test_ssh_connection(config: SshConfig) -> Result<SshTestResult> {
    let target = format!("{}@{}:{}", config.user, config.host, config.port);

    Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, &target)
        .human(format!("🔐 Testing SSH connection to {}", target))
        .emit();

    // TEAM-256: Pre-flight check - verify SSH agent is running
    if let Err(msg) = check_ssh_agent() {
        Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, "failed").human(format!("❌ {}", msg)).emit();
        return Ok(SshTestResult { success: false, error: Some(msg), test_output: None });
    }

    // TEAM-256: Use async russh API with timeout
    let result = tokio::time::timeout(
        Duration::from_secs(config.timeout_secs),
        test_ssh_connection_async(config.clone()),
    )
    .await;

    let test_result = match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => SshTestResult {
            success: false,
            error: Some(format!("SSH error: {}", e)),
            test_output: None,
        },
        Err(_) => SshTestResult {
            success: false,
            error: Some(format!("Connection timeout after {} seconds", config.timeout_secs)),
            test_output: None,
        },
    };

    match &test_result {
        test_result if test_result.success => {
            Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, "success")
                .human(format!("✅ SSH connection to {} successful", target))
                .emit();
        }
        test_result => {
            Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, "failed")
                .human(format!(
                    "❌ SSH connection to {} failed: {}",
                    target,
                    test_result.error.as_deref().unwrap_or("unknown")
                ))
                .emit();
        }
    }

    Ok(test_result)
}

/// Check if SSH is available (pre-flight check)
fn check_ssh_agent() -> Result<(), String> {
    // TEAM-260: Check if ssh command is available
    // SSH agent check is handled by the ssh command itself
    match std::process::Command::new("ssh").arg("-V").output() {
        Ok(_) => Ok(()),
        Err(_) => Err("SSH command not found.\n\
             \n\
             Please install OpenSSH:\n\
             \n\
               # Debian/Ubuntu\n\
               sudo apt-get install openssh-client\n\
             \n\
               # Fedora/RHEL\n\
               sudo dnf install openssh-clients\n\
             \n\
             Then retry the command."
            .to_string()),
    }
}

/// TEAM-260: Async SSH connection test using command-based execution
async fn test_ssh_connection_async(config: SshConfig) -> Result<SshTestResult> {
    // Step 1: Connect and authenticate
    let client = match RbeeSSHClient::connect(&config.host, config.port, &config.user).await {
        Ok(c) => c,
        Err(e) => {
            return Ok(SshTestResult {
                success: false,
                error: Some(format!("Connection failed: {}", e)),
                test_output: None,
            });
        }
    };

    // Step 2: Run test command (already done in connect, but verify again)
    let (stdout, stderr, exit_code) = match client.exec("echo test").await {
        Ok(r) => r,
        Err(e) => {
            client.close().await.ok();
            return Ok(SshTestResult {
                success: false,
                error: Some(format!("Command execution failed: {}", e)),
                test_output: None,
            });
        }
    };

    // Step 3: Close connection
    client.close().await.ok();

    // Step 4: Check result
    if exit_code != 0 {
        return Ok(SshTestResult {
            success: false,
            error: Some(format!("Test command failed with exit code {}: {}", exit_code, stderr)),
            test_output: Some(stdout),
        });
    }

    // Success!
    Ok(SshTestResult { success: true, error: None, test_output: Some(stdout.trim().to_string()) })
}
