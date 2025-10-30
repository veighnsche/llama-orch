//! Rebuild and hot-reload daemon on remote machine
//!
//! # Types/Utils Used (from daemon-lifecycle)
//! - types::start::HttpDaemonConfig - Daemon configuration
//! - Calls build_daemon(), stop_daemon(), install_daemon(), start_daemon()
//!
//! # Requirements
//!
//! ## Input
//! - `daemon_name`: Name of daemon to rebuild
//! - `ssh_config`: SSH connection details
//! - `daemon_config`: Daemon configuration (for restart)
//!
//! ## Process
//! 1. Build binary locally
//!    - Call: `build_daemon(daemon_name, None)`
//!
//! 2. Stop running daemon (if running)
//!    - Call: `stop_daemon(ssh_config, daemon_name, health_url)`
//!
//! 3. Install new binary
//!    - Call: `install_daemon(daemon_name, ssh_config, Some(binary_path))`
//!
//! 4. Start daemon with new binary
//!    - Call: `start_daemon(ssh_config, daemon_config)`
//!
//! ## SSH/SCP Calls
//! - Total: 3-4 calls (stop + install + start)
//! - Build: local only (no SSH)
//!
//! ## Error Handling
//! - Build failed
//! - Stop failed (daemon stuck)
//! - Install failed (SCP error)
//! - Start failed (new binary broken)
//!
//! ## Example
//! ```rust,no_run
//! use remote_daemon_lifecycle::{RebuildConfig, SshConfig};
//! use daemon_lifecycle::HttpDaemonConfig;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
//! let daemon_config = HttpDaemonConfig::new("rbee-hive", "http://192.168.1.100:7835")
//!     .with_args(vec!["--port".to_string(), "7835".to_string()]);
//!
//! let config = RebuildConfig {
//!     daemon_name: "rbee-hive".to_string(),
//!     ssh_config: ssh,
//!     daemon_config,
//!     job_id: None,
//! };
//!
//! rebuild_daemon(config).await?;
//! # Ok(())
//! # }
//! ```

use crate::build::{build_daemon, BuildConfig};
use crate::install::{install_daemon, InstallConfig};
use crate::start::HttpDaemonConfig; // TEAM-330: Moved from types/
use crate::start::{start_daemon, StartConfig};
use crate::stop::{stop_daemon, StopConfig};
use anyhow::{Context, Result};
use observability_narration_core::n;
use observability_narration_macros::with_job_id;
use timeout_enforcer::with_timeout;

/// Configuration for rebuilding daemon on remote machine
///
/// TEAM-330: Includes optional job_id for SSE narration routing
///
/// # Example
/// ```rust,ignore
/// use remote_daemon_lifecycle::{RebuildConfig, SshConfig};
/// use daemon_lifecycle::HttpDaemonConfig;
///
/// let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
/// let daemon_config = HttpDaemonConfig::new("llm-worker-rbee", "http://192.168.1.100:7836")
///     .with_args(vec!["--port".to_string(), "7836".to_string()]);
///
/// let config = RebuildConfig {
///     daemon_name: "llm-worker-rbee".to_string(),
///     ssh_config: ssh,
///     daemon_config,
///     job_id: Some("job-123".to_string()),  // For SSE routing
/// };
/// ```
// TEAM-358: Removed ssh_config field (lifecycle-local = LOCAL only)
#[derive(Debug, Clone)]
pub struct RebuildConfig {
    /// Name of the daemon binary
    pub daemon_name: String,

    /// Daemon configuration (for restart)
    pub daemon_config: HttpDaemonConfig,

    /// Optional job ID for SSE narration routing
    /// When set, all narration (including timeout countdown) goes through SSE
    pub job_id: Option<String>,
}

/// Rebuild and hot-reload daemon on remote machine
///
/// TEAM-330: Enforces 10-minute timeout for entire rebuild process
///
/// # Implementation
/// 1. Build binary locally (build_daemon)
/// 2. Stop running daemon (stop_daemon)
/// 3. Install new binary (install_daemon)
/// 4. Start daemon with new binary (start_daemon)
///
/// # Timeout Strategy
/// - Total timeout: 10 minutes (covers build + stop + install + start)
/// - Build: up to 5 minutes (large binaries)
/// - Stop: 20 seconds (handled by stop_daemon)
/// - Install: up to 5 minutes (handled by install_daemon)
/// - Start: 2 minutes (handled by start_daemon)
///
/// # Job ID Support (TEAM-330)
/// When called with job_id in RebuildConfig, all narration routes through SSE:
///
/// ```rust,ignore
/// let config = RebuildConfig {
///     daemon_name: "llm-worker-rbee".to_string(),
///     ssh_config,
///     daemon_config,
///     job_id: Some(job_id),  // ← Routes narration + countdown through SSE!
/// };
/// rebuild_daemon(config).await?;
/// ```
///
/// The #[with_job_id] macro automatically wraps the function in NarrationContext,
/// routing ALL narration (including timeout countdown) through SSE!
#[with_job_id(config_param = "rebuild_config")]
#[with_timeout(secs = 600, label = "Rebuild daemon")]
pub async fn rebuild_daemon(rebuild_config: RebuildConfig) -> Result<()> {
    let daemon_name = &rebuild_config.daemon_name;
    let daemon_config = &rebuild_config.daemon_config;

    n!("rebuild_start", "🔄 Rebuilding {} locally", daemon_name);

    // Step 1: Build binary locally
    n!("rebuild_build", "🔨 Building {} locally", daemon_name);
    let build_config = BuildConfig {
        daemon_name: daemon_name.clone(),
        target: None,
        job_id: rebuild_config.job_id.clone(),
    };

    let binary_path = build_daemon(build_config).await.context("Failed to build daemon")?;

    n!("rebuild_built", "✅ Built: {}", binary_path.display());

    // Step 2: Stop running daemon (if running)
    n!("rebuild_stop", "🛑 Stopping running daemon");

    // Extract base URL from daemon_config.health_url
    let health_url = daemon_config.health_url.clone();
    let shutdown_url = format!("{}/v1/shutdown", health_url.trim_end_matches("/health"));
    n!("rebuild_shutdown_url", "📡 Using shutdown URL: {}", shutdown_url);

    let stop_config = StopConfig {
        daemon_name: daemon_name.clone(),
        shutdown_url,
        health_url: health_url.clone(),
        job_id: rebuild_config.job_id.clone(),
    };

    // Ignore errors if daemon is not running
    if let Err(e) = stop_daemon(stop_config).await {
        n!("rebuild_stop_warning", "⚠️  Stop failed (daemon may not be running): {}", e);
    } else {
        n!("rebuild_stopped", "✅ Daemon stopped");
    }

    // Step 3: Install new binary
    n!("rebuild_install", "📦 Installing new binary");
    let install_config = InstallConfig {
        daemon_name: daemon_name.clone(),
        local_binary_path: Some(binary_path),
        job_id: rebuild_config.job_id.clone(),
    };

    install_daemon(install_config).await.context("Failed to install new binary")?;

    n!("rebuild_installed", "✅ New binary installed");

    // Step 4: Start daemon with new binary
    n!("rebuild_start_daemon", "🚀 Starting daemon with new binary");
    let start_config = StartConfig {
        daemon_config: daemon_config.clone(),
        job_id: rebuild_config.job_id.clone(),
    };

    let pid = start_daemon(start_config).await.context("Failed to start daemon with new binary")?;

    n!("rebuild_started", "✅ Daemon started with PID: {}", pid);

    n!("rebuild_complete", "🎉 {} rebuilt and restarted successfully locally", daemon_name);

    Ok(())
}

// TEAM-330: Orchestrates build, stop, install, start operations
// See: src/build.rs, src/stop.rs, src/install.rs, src/start.rs
