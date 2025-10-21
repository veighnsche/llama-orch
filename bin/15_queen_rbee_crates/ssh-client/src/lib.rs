// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-188: Implemented SSH test connection functionality
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
//! - Handle SSH authentication (keys, passwords)
//!
//! # Security
//!
//! Uses ssh2 crate for safe SSH operations without command injection vulnerabilities.

use anyhow::{Context, Result};
use observability_narration_core::Narration;
use ssh2::Session;
use std::io::Read;
use std::net::TcpStream;
use std::time::Duration;

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
        Self {
            host: String::new(),
            port: 22,
            user: String::new(),
            timeout_secs: 5,
        }
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
/// This function:
/// 1. Attempts to establish TCP connection with timeout
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

    // Spawn blocking task for SSH operations (ssh2 is sync)
    let result = tokio::task::spawn_blocking(move || {
        test_ssh_connection_blocking(config)
    })
    .await
    .context("SSH test task panicked")?;

    match &result {
        Ok(test_result) if test_result.success => {
            Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, "success")
                .human(format!("✅ SSH connection to {} successful", target))
                .emit();
        }
        Ok(test_result) => {
            Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, "failed")
                .human(format!("❌ SSH connection to {} failed: {}", target, test_result.error.as_deref().unwrap_or("unknown")))
                .emit();
        }
        Err(e) => {
            Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, "error")
                .human(format!("❌ SSH test error: {}", e))
                .emit();
        }
    }

    result
}

/// Blocking SSH connection test (called from tokio::spawn_blocking)
fn test_ssh_connection_blocking(config: SshConfig) -> Result<SshTestResult> {
    let _target = format!("{}:{}", config.host, config.port);

    // Step 1: Establish TCP connection with timeout
    let tcp = match TcpStream::connect_timeout(
        &format!("{}:{}", config.host, config.port)
            .parse()
            .context("Invalid host:port")?,
        Duration::from_secs(config.timeout_secs),
    ) {
        Ok(tcp) => tcp,
        Err(e) => {
            return Ok(SshTestResult {
                success: false,
                error: Some(format!("TCP connection failed: {}", e)),
                test_output: None,
            });
        }
    };

    // Set read/write timeouts
    tcp.set_read_timeout(Some(Duration::from_secs(config.timeout_secs)))
        .context("Failed to set read timeout")?;
    tcp.set_write_timeout(Some(Duration::from_secs(config.timeout_secs)))
        .context("Failed to set write timeout")?;

    // Step 2: Perform SSH handshake
    let mut session = Session::new().context("Failed to create SSH session")?;
    session.set_tcp_stream(tcp);
    
    if let Err(e) = session.handshake() {
        return Ok(SshTestResult {
            success: false,
            error: Some(format!("SSH handshake failed: {}", e)),
            test_output: None,
        });
    }

    // Step 3: Authenticate using SSH agent
    // This respects the user's SSH keys in ~/.ssh/
    if let Err(e) = session.userauth_agent(&config.user) {
        return Ok(SshTestResult {
            success: false,
            error: Some(format!("SSH authentication failed: {}. Ensure SSH agent is running and keys are loaded.", e)),
            test_output: None,
        });
    }

    if !session.authenticated() {
        return Ok(SshTestResult {
            success: false,
            error: Some("SSH authentication failed: Not authenticated after userauth_agent".to_string()),
            test_output: None,
        });
    }

    // Step 4: Run test command
    let mut channel = match session.channel_session() {
        Ok(ch) => ch,
        Err(e) => {
            return Ok(SshTestResult {
                success: false,
                error: Some(format!("Failed to open SSH channel: {}", e)),
                test_output: None,
            });
        }
    };

    if let Err(e) = channel.exec("echo test") {
        return Ok(SshTestResult {
            success: false,
            error: Some(format!("Failed to execute test command: {}", e)),
            test_output: None,
        });
    }

    let mut output = String::new();
    if let Err(e) = channel.read_to_string(&mut output) {
        return Ok(SshTestResult {
            success: false,
            error: Some(format!("Failed to read command output: {}", e)),
            test_output: None,
        });
    }

    channel.wait_close().ok();

    let exit_status = channel.exit_status().unwrap_or(-1);
    if exit_status != 0 {
        return Ok(SshTestResult {
            success: false,
            error: Some(format!("Test command failed with exit code: {}", exit_status)),
            test_output: Some(output),
        });
    }

    // Success!
    Ok(SshTestResult {
        success: true,
        error: None,
        test_output: Some(output.trim().to_string()),
    })
}
