//! Start daemon on remote machine via SSH
//!
//! # Types/Utils Used
//! - types::start::HttpDaemonConfig - Daemon configuration
//! - utils::ssh::ssh_exec() - Execute SSH commands
//! - health_poll::poll_health() - Health polling with exponential backoff (TEAM-358)
//!
//! # Requirements
//!
//! ## Input
//! - `daemon_name`: Name of daemon binary (e.g., "rbee-hive")
//! - `ssh_config`: SSH connection details (hostname, user, port)
//! - `daemon_config`: Daemon configuration (args, health_url, etc.)
//!
//! ## Process
//! 1. Find binary on remote machine (ONE ssh call)
//!    - Try: ~/.local/bin/{daemon}
//!    - Try: target/debug/{daemon}
//!    - Try: target/release/{daemon}
//!    - Return error if not found
//!
//! 2. Start daemon in background (ONE ssh call)
//!    - Use: `nohup {binary} {args} > /dev/null 2>&1 & echo $!`
//!    - Capture PID from stdout
//!
//! 3. Poll health endpoint via HTTP (NO SSH)
//!    - Use: `http://{hostname}:{port}/health`
//!    - Retry with exponential backoff (30 attempts)
//!    - Use health-poll crate (TEAM-358)
//!
//! 4. Return PID
//!
//! ## SSH Calls
//! - Total: 2 SSH calls (find binary, start daemon)
//! - Health polling: HTTP only (no SSH)
//!
//! ## Error Handling
//! - Binary not found on remote
//! - SSH connection failed
//! - Daemon failed to start
//! - Health check timeout
//!
//! ## Example
//! ```rust,no_run
//! use remote_daemon_lifecycle::{start_daemon, SshConfig};
//! use daemon_lifecycle::HttpDaemonConfig;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
//! let config = HttpDaemonConfig::new("rbee-hive", "http://192.168.1.100:7835")
//!     .with_args(vec!["--port".to_string(), "7835".to_string()]);
//!
//! let pid = start_daemon(ssh, config).await?;
//! println!("Started daemon with PID: {}", pid);
//! # Ok(())
//! # }
//! ```

use crate::utils::ssh::ssh_exec;
use crate::SshConfig;
use anyhow::{Context, Result};
use observability_narration_core::n;
use observability_narration_macros::with_job_id;
use std::path::PathBuf;
use timeout_enforcer::with_timeout;

// TEAM-367: Import shared types and utilities
pub use lifecycle_shared::{build_start_command, find_binary_command, HttpDaemonConfig};

/// Configuration for starting daemon on remote machine
///
/// TEAM-330: Includes optional job_id for SSE narration routing
#[derive(Debug, Clone)]
pub struct StartConfig {
    /// SSH connection configuration
    pub ssh_config: SshConfig,

    /// Daemon configuration (name, health_url, args)
    pub daemon_config: HttpDaemonConfig,

    /// Optional job ID for SSE narration routing
    /// When set, all narration (including timeout) goes through SSE
    pub job_id: Option<String>,
}

/// Start daemon on remote machine via SSH
///
/// TEAM-330: Enforces 2-minute timeout for entire start process
///
/// # Implementation
/// 1. Find binary on remote machine (ssh call)
/// 2. Start daemon in background (ssh call)
/// 3. Poll health endpoint via HTTP (no ssh)
/// 4. Return PID
///
/// # Timeout Strategy
/// - Total timeout: 2 minutes (covers find + start + health check)
/// - Find binary: <1 second
/// - Start daemon: <1 second
/// - Health polling: up to 30 seconds (daemon startup time)
/// - Buffer: extra time for slow networks
///
/// # Job ID Support
/// When called with job_id in StartConfig, all narration routes through SSE:
///
/// ```rust,ignore
/// let config = StartConfig {
///     ssh_config,
///     daemon_config,
///     job_id: Some("job-123".to_string()),  // ← Routes through SSE!
/// };
/// start_daemon(config).await?;
/// ```
#[with_job_id(config_param = "start_config")]
#[with_timeout(secs = 120, label = "Start daemon")]
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
    let ssh_config = &start_config.ssh_config;
    let daemon_config = &start_config.daemon_config;
    let daemon_name = &daemon_config.daemon_name;

    n!("start_begin", "🚀 Starting {} on {}@{}", daemon_name, ssh_config.user, ssh_config.hostname);

    // Step 1: Find binary on remote machine
    n!("find_binary", "🔍 Locating {} binary on remote...", daemon_name);

    // TEAM-367: Use shared find_binary_command function
    let find_cmd = find_binary_command(daemon_name);
    let binary_path =
        ssh_exec(ssh_config, &find_cmd).await.context("Failed to find binary on remote")?;

    let binary_path = binary_path.trim();

    if binary_path == "NOT_FOUND" || binary_path.is_empty() {
        n!("binary_not_found", "❌ Binary '{}' not found on remote", daemon_name);
        anyhow::bail!(
            "Binary '{}' not found on remote machine. Install it first with install_daemon()",
            daemon_name
        );
    }

    n!("found_binary", "✅ Found binary at: {}", binary_path);

    // Step 2: Start daemon in background
    n!("starting", "▶️  Starting daemon in background...");

    if !daemon_config.args.is_empty() {
        n!("start_with_args", "⚙️  Starting with args: {}", daemon_config.args.join(" "));
    }

    // TEAM-367: Use shared build_start_command function
    let start_cmd = build_start_command(binary_path, &daemon_config.args);
    let pid_output = ssh_exec(ssh_config, &start_cmd).await.context("Failed to start daemon")?;

    let pid: u32 = pid_output.trim().parse().context("Failed to parse PID from daemon start")?;

    n!("started", "✅ Daemon started with PID: {}", pid);

    // Step 3: Poll health endpoint via HTTP
    n!("health_check", "🏥 Polling health endpoint: {}", daemon_config.health_url);

    // TEAM-358: Use health-poll crate instead of duplicated polling logic
    health_poll::poll_health(
        &daemon_config.health_url,
        30,    // max_attempts
        200,   // initial_delay_ms
        1.5,   // backoff_multiplier
    ).await
    .context("Daemon started but failed health check")?;

    n!("healthy", "✅ Daemon is healthy and responding");
    n!(
        "start_complete",
        "🎉 {} started successfully on {}@{} (PID: {})",
        daemon_name,
        ssh_config.user,
        ssh_config.hostname,
        pid
    );

    Ok(pid)
}
