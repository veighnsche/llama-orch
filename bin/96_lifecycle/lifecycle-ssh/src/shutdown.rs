//! SSH-based shutdown of daemon on remote machine
//!
//! # Types/Utils Used (from daemon-lifecycle)
//! - utils::ssh::ssh_exec() - Execute SSH commands (SIGTERM/SIGKILL)
//! - Called as fallback from stop.rs when HTTP shutdown fails
//!
//! # Requirements
//!
//! ## Input
//! - `daemon_name`: Name of daemon to shutdown
//! - `shutdown_url`: HTTP shutdown endpoint URL (e.g., "http://192.168.1.100:7835/v1/shutdown")
//! - `ssh_config`: SSH connection details (for fallback)
//!
//! ## Process
//! 1. Try graceful shutdown via HTTP shutdown endpoint (NO SSH)
//!    - POST to: `{shutdown_url}`
//!    - Timeout: 10 seconds
//!    - If succeeds: return Ok
//!
//! 2. Wait for daemon to stop (HTTP polling, NO SSH)
//!    - Poll health endpoint every 500ms
//!    - Max wait: 5 seconds
//!    - If stops: return Ok
//!
//! 3. Fallback to SIGTERM via SSH (ONE ssh call)
//!    - Use: `pkill -TERM -f {daemon_name}`
//!    - Wait 2 seconds
//!
//! 4. Fallback to SIGKILL via SSH (ONE ssh call)
//!    - Use: `pkill -KILL -f {daemon_name}`
//!
//! ## SSH Calls
//! - Best case: 0 SSH calls (HTTP shutdown succeeds)
//! - Worst case: 2 SSH calls (SIGTERM + SIGKILL)
//!
//! ## Error Handling
//! - HTTP shutdown failed (continue to SSH)
//! - SIGTERM failed (continue to SIGKILL)
//! - SIGKILL failed (return error)
//!
//! ## Example
//! ```rust,no_run
//! use remote_daemon_lifecycle::{shutdown_daemon, SshConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
//! shutdown_daemon(
//!     "rbee-hive",
//!     "http://192.168.1.100:7835/health",
//!     ssh
//! ).await?;
//! # Ok(())
//! # }
//! ```

use crate::utils::ssh::ssh_exec;
use crate::SshConfig;
use anyhow::{Context, Result};
use observability_narration_core::n;
use observability_narration_macros::with_job_id;
use std::time::Duration;
use tokio::time::sleep;

/// Configuration for shutting down daemon on remote machine
///
/// TEAM-330: Includes optional job_id for SSE narration routing
#[derive(Debug, Clone)]
pub struct ShutdownConfig {
    /// Name of daemon to shutdown
    pub daemon_name: String,

    /// HTTP shutdown endpoint URL
    pub shutdown_url: String,

    /// Health endpoint URL (for polling)
    pub health_url: String,

    /// SSH connection configuration (for fallback)
    pub ssh_config: SshConfig,

    /// Optional job ID for SSE narration routing
    pub job_id: Option<String>,
}

/// SSH-based shutdown of daemon on remote machine
///
/// TEAM-330: Enforces 15-second timeout, SSH-only (no HTTP)
///
/// # Implementation
/// 1. Send SIGTERM via SSH (graceful)
/// 2. Wait 5s and check if stopped
/// 3. Send SIGKILL via SSH (force kill)
///
/// # Timeout Strategy
/// - Total timeout: 15 seconds
/// - SIGTERM wait: 5 seconds
/// - SIGKILL: 2 seconds
/// - Buffer: 8 seconds
///
/// # Job ID Support
/// When called with job_id in ShutdownConfig, all narration routes through SSE
///
/// # Note
/// This is called as a fallback from stop_daemon() when HTTP shutdown fails
#[with_job_id(config_param = "shutdown_config")]
#[with_timeout(secs = 15, label = "SSH shutdown")]
pub async fn shutdown_daemon(shutdown_config: ShutdownConfig) -> Result<()> {
    let daemon_name = &shutdown_config.daemon_name;
    let health_url = &shutdown_config.health_url;
    let ssh_config = &shutdown_config.ssh_config;

    n!(
        "ssh_shutdown_start",
        "🔪 SSH-based shutdown for {} on {}@{}",
        daemon_name,
        ssh_config.user,
        ssh_config.hostname
    );

    // Step 1: Send SIGTERM via SSH
    n!("sigterm", "🔪 Sending SIGTERM via SSH...");

    let sigterm_cmd = format!("pkill -TERM -f {}", daemon_name);
    match ssh_exec(ssh_config, &sigterm_cmd).await {
        Ok(_) => {
            n!("sigterm_sent", "✅ SIGTERM sent, waiting 5s for graceful shutdown...");

            // Wait for graceful shutdown
            sleep(Duration::from_secs(5)).await;

            // Check if daemon stopped
            n!("checking_stopped", "🔍 Checking if daemon stopped after SIGTERM...");
            let client = reqwest::Client::builder().timeout(Duration::from_secs(2)).build()?;

            match client.get(health_url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    n!("still_alive", "⚠️  Daemon still running after SIGTERM, sending SIGKILL");
                }
                _ => {
                    n!("stopped_sigterm", "✅ Daemon stopped after SIGTERM");
                    n!("shutdown_complete", "🎉 {} shutdown complete", daemon_name);
                    return Ok(());
                }
            }
        }
        Err(e) => {
            n!("sigterm_failed", "⚠️  SIGTERM failed: {}, sending SIGKILL", e);
        }
    }

    // Step 2: Send SIGKILL via SSH (force kill)
    n!("sigkill", "💀 Sending SIGKILL via SSH (force kill)...");

    let sigkill_cmd = format!("pkill -KILL -f {}", daemon_name);
    ssh_exec(ssh_config, &sigkill_cmd).await.context("Failed to send SIGKILL")?;

    n!("sigkill_sent", "✅ SIGKILL sent");

    // Give it a moment
    sleep(Duration::from_secs(2)).await;

    n!("shutdown_complete", "🎉 {} force killed via SSH", daemon_name);

    Ok(())
}
