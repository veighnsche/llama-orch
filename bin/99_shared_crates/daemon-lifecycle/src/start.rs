//! Start daemon on remote machine via SSH
//!
//! # Types/Utils Used (from daemon-lifecycle)
//! - types::start::HttpDaemonConfig - Daemon configuration
//! - types::status::HealthPollConfig - Health polling configuration
//! - utils::ssh::ssh_exec() - Execute SSH commands
//! - utils::poll::poll_daemon_health() - Poll health endpoint
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
//!    - Retry with exponential backoff (10 attempts)
//!    - Use daemon-lifecycle's poll_daemon_health()
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

use crate::utils::poll::{poll_daemon_health, HealthPollConfig};
use crate::utils::ssh::ssh_exec;
use crate::SshConfig;
use anyhow::{Context, Result};
use observability_narration_core::n;
use observability_narration_macros::with_job_id;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use timeout_enforcer::with_timeout;

/// Configuration for HTTP-based daemons
///
/// TEAM-330: Moved from types/start.rs (RULE ZERO - inline it)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpDaemonConfig {
    /// Daemon name for narration (e.g., "queen-rbee", "rbee-hive")
    pub daemon_name: String,

    /// Health check URL (e.g., "http://localhost:7833/health")
    pub health_url: String,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,

    /// Path to daemon binary (optional - auto-resolved from daemon_name if not provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<PathBuf>,

    /// Command-line arguments for daemon
    #[serde(default)]
    pub args: Vec<String>,

    /// Maximum health check attempts (default: 10)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_health_attempts: Option<usize>,

    /// Initial health check delay in ms (default: 200)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_initial_delay_ms: Option<u64>,

    /// Process ID (required for signal-based shutdown)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,

    /// Graceful shutdown timeout in seconds (default: 5)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graceful_timeout_secs: Option<u64>,
}

impl HttpDaemonConfig {
    /// Create a new HTTP daemon config
    pub fn new(daemon_name: impl Into<String>, health_url: impl Into<String>) -> Self {
        Self {
            daemon_name: daemon_name.into(),
            health_url: health_url.into(),
            job_id: None,
            binary_path: None,
            args: Vec::new(),
            max_health_attempts: None,
            health_initial_delay_ms: None,
            pid: None,
            graceful_timeout_secs: None,
        }
    }

    /// Set explicit binary path (optional - auto-resolved if not set)
    pub fn with_binary_path(mut self, path: PathBuf) -> Self {
        self.binary_path = Some(path);
        self
    }

    /// Set command-line arguments
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set job_id for narration
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    /// Set process ID (required for signal-based shutdown)
    pub fn with_pid(mut self, pid: u32) -> Self {
        self.pid = Some(pid);
        self
    }

    /// Set graceful shutdown timeout
    pub fn with_graceful_timeout_secs(mut self, secs: u64) -> Self {
        self.graceful_timeout_secs = Some(secs);
        self
    }

    /// Set max health check attempts
    pub fn with_max_health_attempts(mut self, attempts: usize) -> Self {
        self.max_health_attempts = Some(attempts);
        self
    }

    /// Set initial health check delay
    pub fn with_health_initial_delay_ms(mut self, delay_ms: u64) -> Self {
        self.health_initial_delay_ms = Some(delay_ms);
        self
    }
}

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

    let find_cmd = format!(
        "which {} 2>/dev/null || \
         (test -x ~/.local/bin/{} && echo ~/.local/bin/{}) || \
         (test -x target/release/{} && echo target/release/{}) || \
         (test -x target/debug/{} && echo target/debug/{}) || \
         echo 'NOT_FOUND'",
        daemon_name, daemon_name, daemon_name, daemon_name, daemon_name, daemon_name, daemon_name
    );

    let binary_path =
        ssh_exec(ssh_config, &find_cmd).await.context("Failed to find binary on remote")?;

    let binary_path = binary_path.trim();

    if binary_path == "NOT_FOUND" || binary_path.is_empty() {
        anyhow::bail!(
            "Binary '{}' not found on remote machine. Install it first with install_daemon()",
            daemon_name
        );
    }

    n!("found_binary", "✅ Found binary at: {}", binary_path);

    // Step 2: Start daemon in background
    n!("starting", "▶️  Starting daemon in background...");

    // Build command with args
    let args = daemon_config.args.join(" ");
    let start_cmd = if args.is_empty() {
        format!("nohup {} > /dev/null 2>&1 & echo $!", binary_path)
    } else {
        format!("nohup {} {} > /dev/null 2>&1 & echo $!", binary_path, args)
    };

    let pid_output = ssh_exec(ssh_config, &start_cmd).await.context("Failed to start daemon")?;

    let pid: u32 = pid_output.trim().parse().context("Failed to parse PID from daemon start")?;

    n!("started", "✅ Daemon started with PID: {}", pid);

    // Step 3: Poll health endpoint via HTTP
    n!("health_check", "🏥 Polling health endpoint: {}", daemon_config.health_url);

    // Use local health polling
    let poll_config = HealthPollConfig::new(&daemon_config.health_url).with_max_attempts(30);

    poll_daemon_health(poll_config).await.context("Daemon started but failed health check")?;

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
