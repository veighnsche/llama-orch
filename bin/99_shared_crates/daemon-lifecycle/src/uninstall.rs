//! Uninstall daemon binary from remote machine
//!
//! # Types/Utils Used (from daemon-lifecycle)
//! - utils::ssh::ssh_exec() - Execute SSH commands (rm command)
//! - daemon_lifecycle::status::check_daemon_health() - HTTP health check
//!
//! # Requirements
//!
//! ## Input
//! - `daemon_name`: Name of daemon binary to remove
//! - `ssh_config`: SSH connection details
//! - `health_url`: Optional health URL to check if daemon is running
//!
//! ## Process
//! 1. Check if daemon is running (HTTP, NO SSH)
//!    - If running: return error (must stop daemon first)
//!    - If not running: continue
//!
//! 2. Remove binary from remote machine (ONE ssh call)
//!    - Use: `rm -f ~/.local/bin/{daemon_name}`
//!    - Return Ok even if file doesn't exist
//!
//! ## SSH Calls
//! - Total: 1 SSH call (rm command)
//! - Health check: HTTP only (no SSH)
//!
//! ## Error Handling
//! - Daemon still running (must stop first)
//! - SSH connection failed
//! - Permission denied (can't delete file)
//!
//! ## Example
//! ```rust,no_run
//! use remote_daemon_lifecycle::{UninstallConfig, SshConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
//! let config = UninstallConfig {
//!     daemon_name: "rbee-hive".to_string(),
//!     ssh_config: ssh,
//!     health_url: Some("http://192.168.1.100:7835".to_string()),
//!     health_timeout_secs: Some(2),
//!     job_id: None,
//! };
//! uninstall_daemon(config).await?;
//! # Ok(())
//! # }
//! ```

use crate::utils::ssh::ssh_exec;
use crate::SshConfig;
use anyhow::{Context, Result};
use observability_narration_core::n;
use observability_narration_macros::with_job_id;
use timeout_enforcer::with_timeout;

/// Configuration for remote daemon uninstallation
///
/// TEAM-330: Includes optional job_id for SSE narration routing
///
/// # Example
/// ```rust,ignore
/// use remote_daemon_lifecycle::{UninstallConfig, SshConfig};
///
/// let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
/// let config = UninstallConfig {
///     daemon_name: "llm-worker-rbee".to_string(),
///     ssh_config: ssh,
///     health_url: Some("http://192.168.1.100:7835".to_string()),
///     health_timeout_secs: Some(2),
///     job_id: Some("job-123".to_string()),  // For SSE routing
/// };
/// ```
#[derive(Debug, Clone)]
pub struct UninstallConfig {
    /// Name of the daemon binary
    pub daemon_name: String,

    /// SSH connection configuration
    pub ssh_config: SshConfig,

    /// Optional health URL to check if daemon is running
    /// If provided, will verify daemon is stopped before uninstalling
    pub health_url: Option<String>,

    /// Health check timeout in seconds (default: 2)
    pub health_timeout_secs: Option<u64>,

    /// Optional job ID for SSE narration routing
    /// When set, all narration (including timeout countdown) goes through SSE
    pub job_id: Option<String>,
}

/// Uninstall daemon binary from remote machine
///
/// TEAM-330: Enforces 1-minute timeout for uninstallation process
///
/// # Implementation
/// 1. Check if daemon is running (HTTP health check)
/// 2. Remove binary via SSH
/// 3. Verify removal
///
/// # Timeout Strategy
/// - Total timeout: 1 minute (uninstall is fast, just rm command)
/// - Health check: 2 seconds (configurable)
/// - SSH commands: <1 second each
///
/// # Job ID Support (TEAM-330)
/// When called from hive daemon managing worker lifecycle, set job_id in config:
///
/// ```rust,ignore
/// let config = UninstallConfig {
///     daemon_name: "llm-worker-rbee".to_string(),
///     ssh_config,
///     health_url: Some("http://192.168.1.100:7835".to_string()),
///     health_timeout_secs: Some(2),
///     job_id: Some(job_id),  // ← Routes narration + countdown through SSE!
/// };
/// uninstall_daemon(config).await?;
/// ```
///
/// The #[with_job_id] macro automatically wraps the function in NarrationContext,
/// routing ALL narration (including timeout countdown) through SSE!
#[with_job_id(config_param = "uninstall_config")]
#[with_timeout(secs = 60, label = "Uninstall daemon")]
pub async fn uninstall_daemon(uninstall_config: UninstallConfig) -> Result<()> {
    let daemon_name = &uninstall_config.daemon_name;
    let ssh_config = &uninstall_config.ssh_config;

    n!(
        "uninstall_start",
        "🗑️  Uninstalling {} from {}@{}",
        daemon_name,
        ssh_config.user,
        ssh_config.hostname
    );

    // Step 1: Check if daemon is running (if health_url provided)
    if let Some(health_url) = &uninstall_config.health_url {
        n!("health_check", "🔍 Checking if daemon is running");

        // Build full health URL (append /health if not already present)
        let full_health_url = if health_url.ends_with("/health") {
            health_url.clone()
        } else {
            format!("{}/health", health_url)
        };

        // TEAM-338: RULE ZERO - Updated to new signature
        let status =
            crate::status::check_daemon_health(&full_health_url, daemon_name, ssh_config).await;

        if status.is_running {
            n!(
                "daemon_still_running",
                "⚠️  Daemon '{}' is currently running. Stop it first.",
                daemon_name
            );
            anyhow::bail!("Daemon {} is still running at {}", daemon_name, health_url);
        }

        n!("daemon_stopped", "✅ Daemon is not running, safe to uninstall");
    }

    // Step 2: Remove binary from remote machine
    let remote_path = format!("~/.local/bin/{}", daemon_name);
    n!("removing", "🗑️  Removing {} from {}", daemon_name, remote_path);

    // Use rm -f to not fail if file doesn't exist
    let rm_cmd = format!("rm -f ~/.local/bin/{}", daemon_name);
    ssh_exec(&ssh_config, &rm_cmd).await.context("Failed to remove binary from remote")?;

    // Step 3: Verify removal (optional, but good practice)
    n!("verify", "✅ Verifying removal");
    let verify_cmd = format!("test ! -f ~/.local/bin/{} && echo 'REMOVED'", daemon_name);
    let output = ssh_exec(&ssh_config, &verify_cmd).await.context("Failed to verify removal")?;

    if !output.trim().contains("REMOVED") {
        n!("verify_warning", "⚠️  Could not verify removal (file may still exist)");
    }

    n!(
        "uninstall_complete",
        "🎉 {} uninstalled successfully from {}@{}",
        daemon_name,
        ssh_config.user,
        ssh_config.hostname
    );

    Ok(())
}

// TEAM-330: ssh_exec() is in utils/ssh.rs
// See: src/utils/ssh.rs
