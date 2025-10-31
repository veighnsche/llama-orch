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
use std::path::PathBuf;
use timeout_enforcer::with_timeout;

// TEAM-367: Import shared types and utilities
// TEAM-378: Removed find_binary_command (moved to lifecycle-ssh)
pub use lifecycle_shared::{build_start_command, HttpDaemonConfig};

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

    // Step 1: Find binary on local machine (smart mode-aware selection)
    n!("find_binary", "🔍 Locating {} binary locally...", daemon_name);

    // TEAM-378: Phase 3 - Use smart check_binary_exists (prefers production, falls back to dev)
    use crate::utils::{check_binary_exists, CheckMode};
    if !check_binary_exists(daemon_name, CheckMode::Any).await {
        anyhow::bail!("Binary '{}' not found locally. Install it first with install_daemon()", daemon_name);
    }

    // Binary exists, now resolve its path
    use lifecycle_shared::BINARY_INSTALL_DIR;
    use std::path::PathBuf;
    
    let binary_path = if let Ok(home) = std::env::var("HOME") {
        let installed_path = PathBuf::from(&home).join(BINARY_INSTALL_DIR).join(daemon_name);
        if installed_path.exists() {
            installed_path
        } else {
            PathBuf::from(format!("target/debug/{}", daemon_name))
        }
    } else {
        PathBuf::from(format!("target/debug/{}", daemon_name))
    };

    // Step 2: Start daemon in background (with monitoring)
    n!("starting", "▶️  Starting daemon in background...");

    // TEAM-359: Use ProcessMonitor for cgroup-based spawning
    let pid = if let (Some(group), Some(instance)) =
        (&daemon_config.monitor_group, &daemon_config.monitor_instance)
    {
        n!(
            "monitored_spawn",
            "📊 Spawning with monitoring: group={}, instance={}",
            group,
            instance
        );

        let monitor_config = rbee_hive_monitor::MonitorConfig {
            group: group.clone(),
            instance: instance.clone(),
            cpu_limit: None,    // TODO: Make configurable
            memory_limit: None, // TODO: Make configurable
        };

        if !daemon_config.args.is_empty() {
            n!("start_with_args", "⚙️  Starting with args: {}", daemon_config.args.join(" "));
        }

        // TEAM-378: Convert PathBuf to &str for spawn_monitored
        let binary_path_str = binary_path.to_str()
            .context("Binary path contains invalid UTF-8")?;

        rbee_hive_monitor::ProcessMonitor::spawn_monitored(
            monitor_config,
            binary_path_str,
            daemon_config.args.clone(),
        )
        .await
        .context("Failed to spawn monitored daemon")?
    } else {
        // Fallback: Plain spawn without monitoring (for backwards compatibility)
        n!("unmonitored_spawn", "⚠️  Spawning WITHOUT monitoring (no group/instance specified)");

        if !daemon_config.args.is_empty() {
            n!("start_with_args", "⚙️  Starting with args: {}", daemon_config.args.join(" "));
        }

        // TEAM-378: Convert PathBuf to str for shell command
        let binary_path_str = binary_path.to_str()
            .context("Binary path contains invalid UTF-8")?;
        
        // TEAM-367: Use shared build_start_command function
        let start_cmd = build_start_command(binary_path_str, &daemon_config.args);
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
