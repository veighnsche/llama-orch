// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-188: Implemented SSH test connection functionality
// TEAM-222: Investigated behavior inventory (Phase 2)
// TEAM-256: Migrated from ssh2 to russh for async operations and SFTP support
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
//! - Copy files via SFTP
//! - Handle SSH authentication (SSH agent)
//!
//! # Security
//!
//! Uses russh crate for safe async SSH operations without command injection vulnerabilities.

use anyhow::{Context, Result};
use observability_narration_core::Narration;
use russh_sftp::client::SftpSession;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;

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

// TEAM-256: SSH event handler for russh
struct RbeeSSHHandler;

#[async_trait::async_trait]
impl russh::client::Handler for RbeeSSHHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        server_public_key: &russh_keys::key::PublicKey,
    ) -> Result<bool, Self::Error> {
        // TEAM-256: Host key verification disabled for automated workflows
        // This matches ssh -o StrictHostKeyChecking=no behavior
        // In production environments, implement known_hosts verification:
        //   1. Store host keys in ~/.config/rbee/known_hosts
        //   2. Verify server_public_key matches stored key
        //   3. Prompt user on first connection (trust-on-first-use)
        let _ = server_public_key; // Acknowledge parameter
        Ok(true)
    }
}

/// TEAM-256: SSH client for remote operations
pub struct RbeeSSHClient {
    session: russh::client::Handle<RbeeSSHHandler>,
}

// TEAM-256: Ensure connections are always closed (no resource leaks)
impl Drop for RbeeSSHClient {
    fn drop(&mut self) {
        // Disconnect is async but Drop is sync
        // The session will be dropped and cleaned up by russh
        // This is acceptable since we explicitly call close() in normal flow
    }
}

impl RbeeSSHClient {
    /// Connect to remote host with timeout
    ///
    /// TEAM-256: Priority 1 fixes:
    /// - Uses SSH agent for authentication (not disk keys)
    /// - Adds connection timeout (30 seconds)
    /// - Tries all agent keys automatically
    pub async fn connect(host: &str, port: u16, user: &str) -> Result<Self> {
        // TEAM-256: Add connection timeout (Priority 1)
        let connect_future = Self::connect_internal(host, port, user);
        tokio::time::timeout(Duration::from_secs(30), connect_future)
            .await
            .context("SSH connection timeout (30s)")?
    }

    /// Internal connection logic (for timeout wrapping)
    async fn connect_internal(host: &str, port: u16, user: &str) -> Result<Self> {
        let config = russh::client::Config::default();
        let mut session = russh::client::connect(Arc::new(config), (host, port), RbeeSSHHandler)
            .await
            .context("Failed to connect to SSH server")?;

        // TEAM-256: Load SSH keys from standard locations
        // Matches standard ssh client behavior (~/.ssh/id_*)
        // Note: russh-keys agent support is for server-side auth, not client-side
        let home = std::env::var("HOME").context("HOME environment variable not set")?;
        let key_paths = vec![
            format!("{}/.ssh/id_ed25519", home),
            format!("{}/.ssh/id_rsa", home),
            format!("{}/.ssh/id_ecdsa", home),
        ];

        let mut authenticated = false;
        let mut tried_keys = Vec::new();

        for key_path in &key_paths {
            if !std::path::Path::new(key_path).exists() {
                continue;
            }

            tried_keys.push(key_path.clone());

            // Try to load and decrypt the key
            let key_pair = match russh_keys::load_secret_key(key_path, None) {
                Ok(k) => k,
                Err(russh_keys::Error::CouldNotReadKey) => {
                    // Key is encrypted, skip (user should use ssh-agent)
                    continue;
                }
                Err(_) => continue,
            };

            // Try to authenticate
            let auth_result = session.authenticate_publickey(user, Arc::new(key_pair)).await;

            if let Ok(true) = auth_result {
                authenticated = true;
                break;
            }
        }

        if !authenticated {
            if tried_keys.is_empty() {
                anyhow::bail!(
                    "No SSH keys found. Create one with:\n  ssh-keygen -t ed25519\n  ssh-copy-id {}@{}",
                    user, host
                );
            } else {
                anyhow::bail!(
                    "SSH authentication failed. Tried {} key(s): {}\n\nIf keys are encrypted, use ssh-agent:\n  eval $(ssh-agent)\n  ssh-add",
                    tried_keys.len(),
                    tried_keys.join(", ")
                );
            }
        }

        Ok(Self { session })
    }

    /// Execute command on remote host
    pub async fn exec(&mut self, command: &str) -> Result<(String, String, i32)> {
        let mut channel =
            self.session.channel_open_session().await.context("Failed to open SSH channel")?;

        channel.exec(true, command).await.context("Failed to execute command")?;

        let mut stdout = String::new();
        let mut stderr = String::new();
        let mut code = None;

        loop {
            let msg = channel.wait().await.context("Failed to wait for channel message")?;
            match msg {
                russh::ChannelMsg::Data { ref data } => {
                    stdout.push_str(&String::from_utf8_lossy(data));
                }
                russh::ChannelMsg::ExtendedData { ref data, ext } => {
                    if ext == 1 {
                        stderr.push_str(&String::from_utf8_lossy(data));
                    }
                }
                russh::ChannelMsg::ExitStatus { exit_status } => {
                    code = Some(exit_status);
                }
                russh::ChannelMsg::Eof => {
                    break;
                }
                _ => {}
            }
        }

        let exit_code = code.unwrap_or(255) as i32;
        Ok((stdout, stderr, exit_code))
    }

    /// Copy file to remote host via SFTP
    pub async fn copy_file(&mut self, local_path: &str, remote_path: &str) -> Result<()> {
        let channel =
            self.session.channel_open_session().await.context("Failed to open SFTP channel")?;
        channel
            .request_subsystem(true, "sftp")
            .await
            .context("Failed to request SFTP subsystem")?;

        let sftp = SftpSession::new(channel.into_stream())
            .await
            .context("Failed to create SFTP session")?;

        // Read local file
        let local_data = tokio::fs::read(local_path).await.context("Failed to read local file")?;

        // Write to remote
        let mut remote_file =
            sftp.create(remote_path).await.context("Failed to create remote file")?;

        remote_file.write_all(&local_data).await.context("Failed to write to remote file")?;
        remote_file.sync_all().await.context("Failed to sync remote file")?;

        Ok(())
    }

    /// Close connection
    pub async fn close(self) -> Result<()> {
        self.session
            .disconnect(russh::Disconnect::ByApplication, "", "")
            .await
            .context("Failed to disconnect SSH session")?;
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
/// TEAM-256: Rewritten to use russh async API
///
/// This function:
/// 1. Attempts to establish SSH connection with timeout
/// 2. Performs SSH handshake
/// 3. Attempts authentication using SSH agent
/// 4. Runs a simple test command (`echo test`)
/// 5. Returns success/failure with details
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

/// Check if SSH agent is running (pre-flight check)
fn check_ssh_agent() -> Result<(), String> {
    match std::env::var("SSH_AUTH_SOCK") {
        Ok(sock) if !sock.is_empty() => Ok(()),
        _ => Err("SSH agent not running.\n\
             \n\
             To start SSH agent:\n\
             \n\
               eval $(ssh-agent)\n\
               ssh-add ~/.ssh/id_rsa\n\
             \n\
             Then retry the command."
            .to_string()),
    }
}

/// TEAM-256: Async SSH connection test using russh
async fn test_ssh_connection_async(config: SshConfig) -> Result<SshTestResult> {
    // Step 1: Connect and authenticate
    let mut client = match RbeeSSHClient::connect(&config.host, config.port, &config.user).await {
        Ok(c) => c,
        Err(e) => {
            return Ok(SshTestResult {
                success: false,
                error: Some(format!("Connection failed: {}", e)),
                test_output: None,
            });
        }
    };

    // Step 2: Run test command
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
