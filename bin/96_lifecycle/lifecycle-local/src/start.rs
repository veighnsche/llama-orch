//! Start daemon on LOCAL machine
//!
//! TEAM-358: Refactored to remove SSH code (lifecycle-local = LOCAL only)
//!
//! # Types/Utils Used
//! - HttpDaemonConfig - Daemon configuration
//! - health_poll::poll_health() - Health polling with exponential backoff
//! - utils::local::local_exec() - Execute local commands
//!
//! # Requirements
//!
//! ## Input
//! - `daemon_name`: Name of daemon binary (e.g., "rbee-hive")
//! - `daemon_config`: Daemon configuration (args, health_url, etc.)
//!
//! ## Process
//! 1. Find binary on LOCAL machine
//!    - Try: target/debug/{daemon}
//!    - Try: target/release/{daemon}
//!    - Try: ~/.local/bin/{daemon}
//!    - Return error if not found
//!
//! 2. Start daemon in background (local process)
//!    - Use: `nohup {binary} {args} > /dev/null 2>&1 & echo $!`
//!    - Capture PID from stdout
//!
//! 3. Poll health endpoint via HTTP
//!    - Use: `http://localhost:{port}/health`
//!    - Retry with exponential backoff (30 attempts)
//!    - Use health-poll crate
//!
//! 4. Return PID
//!
//! ## Error Handling
//! - Binary not found locally
//! - Daemon failed to start
//! - Health check timeout
//!
//! ## Example
//! ```rust,no_run
//! use lifecycle_local::{start_daemon, HttpDaemonConfig, StartConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835")
//!     .with_args(vec!["--port".to_string(), "7835".to_string()]);
//!
//! let start_config = StartConfig {
//!     daemon_config: config,
//!     job_id: None,
//! };
//!
//! let pid = start_daemon(start_config).await?;
//! println!("Started daemon with PID: {}", pid);
//! # Ok(())
//! # }
//! ```

use crate::utils::local::local_exec;
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

    /// TEAM-359: Monitoring group (e.g., "llm", "queen", "hive")
    /// Used for cgroup organization on Linux
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitor_group: Option<String>,

    /// TEAM-359: Monitoring instance (e.g., port number)
    /// Used for cgroup organization on Linux
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitor_instance: Option<String>,
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
            monitor_group: None,  // TEAM-359: Monitoring fields
            monitor_instance: None,
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

    /// TEAM-359: Set monitoring group and instance for cgroup organization
    pub fn with_monitoring(mut self, group: impl Into<String>, instance: impl Into<String>) -> Self {
        self.monitor_group = Some(group.into());
        self.monitor_instance = Some(instance.into());
        self
    }
}

/// Configuration for starting daemon on LOCAL machine
///
/// TEAM-358: Removed ssh_config (lifecycle-local = LOCAL only)
#[derive(Debug, Clone)]
pub struct StartConfig {
    /// Daemon configuration (name, health_url, args)
    pub daemon_config: HttpDaemonConfig,

    /// Optional job ID for SSE narration routing
    /// When set, all narration (including timeout) goes through SSE
    pub job_id: Option<String>,
}

/// Start daemon on LOCAL machine
///
/// TEAM-358: Refactored to remove SSH code, use health-poll crate
///
/// # Implementation
/// 1. Find binary on local machine
/// 2. Start daemon in background (local process)
/// 3. Poll health endpoint via HTTP
/// 4. Return PID
///
/// # Timeout Strategy
/// - Total timeout: 2 minutes (covers find + start + health check)
/// - Find binary: <1 second
/// - Start daemon: <1 second
/// - Health polling: up to 30 seconds (daemon startup time)
///
/// # Job ID Support
/// When called with job_id in StartConfig, all narration routes through SSE:
///
/// ```rust,ignore
/// let config = StartConfig {
///     daemon_config,
///     job_id: Some("job-123".to_string()),  // ← Routes through SSE!
/// };
/// start_daemon(config).await?;
/// ```
#[with_job_id(config_param = "start_config")]
#[with_timeout(secs = 120, label = "Start daemon")]
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
    let daemon_config = &start_config.daemon_config;
    let daemon_name = &daemon_config.daemon_name;

    n!("start_begin", "🚀 Starting {} locally", daemon_name);

    // Step 1: Find binary on local machine
    n!("find_binary", "🔍 Locating {} binary locally...", daemon_name);

    // TEAM-358: Prioritize target/ over ~/.local/bin for development
    let find_cmd = format!(
        "(test -x target/debug/{} && echo target/debug/{}) || \
         (test -x target/release/{} && echo target/release/{}) || \
         (test -x ~/.local/bin/{} && echo ~/.local/bin/{}) || \
         which {} 2>/dev/null || \
         echo 'NOT_FOUND'",
        daemon_name, daemon_name, daemon_name, daemon_name, daemon_name, daemon_name, daemon_name
    );

    let binary_path = local_exec(&find_cmd).await.context("Failed to find binary locally")?;

    let binary_path = binary_path.trim();

    if binary_path == "NOT_FOUND" || binary_path.is_empty() {
        n!("binary_not_found", "❌ Binary '{}' not found locally", daemon_name);
        anyhow::bail!(
            "Binary '{}' not found locally. Install it first with install_daemon()",
            daemon_name
        );
    }

    n!("found_binary", "✅ Found binary at: {}", binary_path);

    // Step 2: Start daemon in background (with monitoring)
    n!("starting", "▶️  Starting daemon in background...");

    // TEAM-359: Use ProcessMonitor for cgroup-based spawning
    let pid = if let (Some(group), Some(instance)) = (&daemon_config.monitor_group, &daemon_config.monitor_instance) {
        n!("monitored_spawn", "📊 Spawning with monitoring: group={}, instance={}", group, instance);
        
        let monitor_config = rbee_hive_monitor::MonitorConfig {
            group: group.clone(),
            instance: instance.clone(),
            cpu_limit: None,  // TODO: Make configurable
            memory_limit: None,  // TODO: Make configurable
        };

        if !daemon_config.args.is_empty() {
            n!("start_with_args", "⚙️  Starting with args: {}", daemon_config.args.join(" "));
        }

        rbee_hive_monitor::ProcessMonitor::spawn_monitored(
            monitor_config,
            binary_path,
            daemon_config.args.clone(),
        )
        .await
        .context("Failed to spawn monitored daemon")?
    } else {
        // Fallback: Plain spawn without monitoring (for backwards compatibility)
        n!("unmonitored_spawn", "⚠️  Spawning WITHOUT monitoring (no group/instance specified)");
        
        let args = daemon_config.args.join(" ");
        let start_cmd = if args.is_empty() {
            format!("nohup {} > /dev/null 2>&1 & echo $!", binary_path)
        } else {
            n!("start_with_args", "⚙️  Starting with args: {}", args);
            format!("nohup {} {} > /dev/null 2>&1 & echo $!", binary_path, args)
        };

        let pid_output = local_exec(&start_cmd).await.context("Failed to start daemon")?;
        pid_output.trim().parse().context("Failed to parse PID from daemon start")?
    };

    n!("started", "✅ Daemon started with PID: {}", pid);

    // Step 3: Poll health endpoint via HTTP
    n!("health_check", "🏥 Polling health endpoint: {}", daemon_config.health_url);

    // TEAM-358: Use health-poll crate instead of duplicated polling logic
    health_poll::poll_health(
        &daemon_config.health_url,
        30,  // max_attempts
        200, // initial_delay_ms
        1.5, // backoff_multiplier
    )
    .await
    .context("Daemon started but failed health check")?;

    n!("healthy", "✅ Daemon is healthy and responding");
    n!("start_complete", "🎉 {} started successfully (PID: {})", daemon_name, pid);

    Ok(pid)
}
