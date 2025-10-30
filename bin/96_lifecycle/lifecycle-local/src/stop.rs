//! Stop daemon on remote machine via HTTP
//!
//! # Types/Utils Used (from daemon-lifecycle)
//! - reqwest for HTTP shutdown endpoint
//! - Falls back to shutdown_daemon() for SSH-based shutdown
//!
//! # Requirements
//!
//! ## Input
//! - `daemon_name`: Name of daemon to stop
//! - `ssh_config`: SSH connection details
//! - `shutdown_url`: HTTP shutdown endpoint URL (e.g., "http://192.168.1.100:7835/v1/shutdown")
//!
//! ## Process
//! 1. Try graceful shutdown via HTTP shutdown endpoint (NO SSH)
//!    - POST to: `{shutdown_url}`
//!    - Timeout: 5 seconds
//!    - If succeeds: return Ok
//!    - If fails: continue to step 2
//!
//! 2. Force kill via SSH (ONE ssh call)
//!    - Use: `pkill -f {daemon_name}`
//!    - Return Ok if successful
//!
//! ## SSH Calls
//! - Best case: 0 SSH calls (HTTP shutdown succeeds)
//! - Worst case: 1 SSH call (force kill)
//!
//! ## Error Handling
//! - SSH connection failed
//! - Process not found (not an error - daemon already stopped)
//! - Kill command failed
//!
//! ## Example
//! ```rust,no_run
//! use remote_daemon_lifecycle::{stop_daemon, SshConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
//! stop_daemon(ssh, "rbee-hive", "http://192.168.1.100:7835").await?;
//! # Ok(())
//! # }
//! ```

use crate::shutdown::{shutdown_daemon, ShutdownConfig};
use anyhow::{Context, Result};
use observability_narration_core::n;
use observability_narration_macros::with_job_id;
use std::time::Duration;
use timeout_enforcer::with_timeout;
use tokio::time::sleep;

/// Configuration for stopping daemon LOCALLY
///
/// TEAM-358: Removed ssh_config (lifecycle-local = LOCAL only)
#[derive(Debug, Clone)]
pub struct StopConfig {
    /// Name of daemon to stop
    pub daemon_name: String,

    /// HTTP shutdown endpoint URL
    pub shutdown_url: String,

    /// Health endpoint URL (for polling)
    pub health_url: String,

    /// Optional job ID for SSE narration routing
    pub job_id: Option<String>,
}

/// Stop daemon LOCALLY via HTTP
///
/// TEAM-358: Refactored to remove SSH fallback (lifecycle-local = LOCAL only)
///
/// # Implementation
/// 1. Try HTTP shutdown endpoint (10s timeout)
/// 2. Poll health endpoint to verify shutdown (5s max)
/// 3. If HTTP fails, use local process termination
///
/// # Timeout Strategy
/// - Total timeout: 20 seconds
/// - HTTP shutdown: 10 seconds
/// - Health polling: 5 seconds
/// - SSH fallback: handled by shutdown_daemon (30s)
///
/// # Job ID Support
/// When called with job_id in StopConfig, all narration routes through SSE
#[with_job_id(config_param = "stop_config")]
#[with_timeout(secs = 20, label = "Stop daemon")]
pub async fn stop_daemon(stop_config: StopConfig) -> Result<()> {
    let daemon_name = &stop_config.daemon_name;
    let shutdown_url = &stop_config.shutdown_url;
    let health_url = &stop_config.health_url;

    n!("stop_start", "🛑 Stopping {} locally", daemon_name);

    // Step 1: Try HTTP shutdown endpoint
    n!("http_shutdown", "📡 Attempting HTTP shutdown: {}", shutdown_url);

    let client = reqwest::Client::builder().timeout(Duration::from_secs(10)).build()?;

    match client.post(shutdown_url).send().await {
        Ok(response) if response.status().is_success() => {
            n!("http_success", "✅ HTTP shutdown request accepted");

            // Step 2: Poll health endpoint to verify shutdown
            n!("polling", "⏳ Waiting for daemon to stop (up to 10 attempts)...");

            for attempt in 1..=10 {
                sleep(Duration::from_millis(500)).await;

                match client.get(health_url).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        n!("still_running", "⏳ Daemon still running (attempt {}/10)", attempt);
                    }
                    _ => {
                        // Daemon stopped responding - success!
                        n!("stopped", "✅ Daemon stopped gracefully via HTTP");
                        n!("stop_complete", "🎉 {} stopped successfully", daemon_name);
                        return Ok(());
                    }
                }
            }

            n!("http_timeout", "⚠️  Daemon didn't stop after HTTP shutdown, falling back to SSH");
        }
        Ok(response) => {
            n!(
                "http_failed",
                "⚠️  HTTP shutdown failed: {}, falling back to SSH",
                response.status()
            );
        }
        Err(e) => {
            n!("http_error", "⚠️  HTTP shutdown error: {}, falling back to SSH", e);
        }
    }

    // Step 3: Fallback to SSH-based shutdown
    n!("ssh_fallback", "🔄 Falling back to SSH-based shutdown...");

    let shutdown_config = ShutdownConfig {
        daemon_name: daemon_name.to_string(),
        shutdown_url: shutdown_url.to_string(),
        health_url: health_url.to_string(),
        job_id: stop_config.job_id.clone(),
    };

    shutdown_daemon(shutdown_config).await.context("Local shutdown failed")?;

    n!("stop_complete", "🎉 {} stopped successfully", daemon_name);

    Ok(())
}
