//! Graceful daemon shutdown utilities
//!
//! TEAM-276: Extracted pattern from queen-lifecycle and hive-lifecycle
//! TEAM-316: Extended ShutdownConfig with lifecycle-specific fields

use anyhow::Result;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use std::time::Duration;
use tokio::time::sleep;

// TEAM-316: Use ShutdownConfig from daemon-contract as base
pub use daemon_contract::ShutdownConfig as ShutdownConfigBase;

/// Extended shutdown config with lifecycle-specific fields
///
/// TEAM-316: Extends daemon-contract::ShutdownConfig with health check URL
#[derive(Clone)]
pub struct ShutdownConfig {
    /// Base configuration from contract
    pub base: ShutdownConfigBase,

    /// Health check URL to verify daemon is running
    pub health_url: String,

    /// Shutdown endpoint (e.g., "/v1/shutdown")
    pub shutdown_endpoint: String,

    /// Timeout for graceful shutdown (SIGTERM) before force kill
    pub sigterm_timeout_secs: u64,
}

impl ShutdownConfig {
    /// Create a new shutdown config
    pub fn new(
        daemon_name: impl Into<String>,
        health_url: impl Into<String>,
        shutdown_endpoint: impl Into<String>,
    ) -> Self {
        let base = ShutdownConfigBase {
            daemon_name: daemon_name.into(),
            pid: 0, // Will be set later if needed
            graceful_timeout_secs: 5,
            job_id: None,
        };

        Self {
            base,
            health_url: health_url.into(),
            shutdown_endpoint: shutdown_endpoint.into(),
            sigterm_timeout_secs: 5,
        }
    }

    /// Set the SIGTERM timeout
    pub fn with_sigterm_timeout(mut self, secs: u64) -> Self {
        self.sigterm_timeout_secs = secs;
        self.base.graceful_timeout_secs = secs;
        self
    }

    /// Set the job_id for narration
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.base.job_id = Some(job_id.into());
        self
    }
}

/// Gracefully shutdown a daemon via HTTP endpoint
///
/// TEAM-276: Extracted from queen-lifecycle/stop.rs
///
/// Steps:
/// 1. Check if daemon is running (via health endpoint)
/// 2. Send shutdown request to shutdown endpoint
/// 3. Handle expected connection errors (daemon shuts down before responding)
///
/// # Arguments
/// * `config` - Shutdown configuration
///
/// # Returns
/// * `Ok(())` - Daemon stopped successfully (or was not running)
/// * `Err` - Unexpected error during shutdown
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{graceful_shutdown, ShutdownConfig};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = ShutdownConfig::new(
///     "queen-rbee",
///     "http://localhost:8500",
///     "http://localhost:8500/v1/shutdown",
/// ).with_job_id("job-123");
///
/// graceful_shutdown(config).await?;
/// # Ok(())
/// # }
/// ```
pub async fn graceful_shutdown(config: ShutdownConfig) -> Result<()> {
    // TEAM-311: Migrated to n!() macro
    // TEAM-316: Updated to use config.base fields
    let ctx = config.base.job_id.as_ref().map(|jid| NarrationContext::new().with_job_id(jid));
    
    let shutdown_impl = async {
        // Step 1: Check if daemon is running
        let is_running =
            crate::health::is_daemon_healthy(&config.health_url, None, Some(Duration::from_secs(2)))
                .await;

        if !is_running {
            n!("daemon_not_running", "⚠️  {} not running", config.base.daemon_name);
            return Ok(());
        }

        // Step 2: Send shutdown request
        n!("daemon_shutdown", "🛑 Shutting down {}...", config.base.daemon_name);

        let shutdown_client = reqwest::Client::builder().timeout(Duration::from_secs(30)).build()?;

        match shutdown_client.post(&config.shutdown_endpoint).send().await {
            Ok(_) => {
                n!("daemon_stopped", "✅ {} stopped", config.base.daemon_name);
                Ok(())
            }
            Err(e) => {
                // Connection closed/reset is expected - daemon shuts down before responding
                if e.is_connect() || e.to_string().contains("connection closed") {
                    n!("daemon_stopped", "✅ {} stopped", config.base.daemon_name);
                    Ok(())
                } else {
                    // Unexpected error
                    n!("daemon_shutdown_failed", "⚠️  Failed to stop {}: {}", config.base.daemon_name, e);
                    Err(e.into())
                }
            }
        }
    };
    
    // Execute with context if job_id provided
    if let Some(ctx) = ctx {
        with_narration_context(ctx, shutdown_impl).await
    } else {
        shutdown_impl.await
    }
}

/// Force shutdown a daemon process by PID (SIGTERM → SIGKILL)
///
/// TEAM-276: Extracted pattern from hive-lifecycle/stop.rs
///
/// Steps:
/// 1. Send SIGTERM to process
/// 2. Wait for graceful timeout
/// 3. If still running, send SIGKILL
///
/// # Arguments
/// * `pid` - Process ID to kill
/// * `daemon_name` - Daemon name for narration
/// * `timeout_secs` - Graceful timeout before SIGKILL
/// * `job_id` - Optional job_id for narration
///
/// # Returns
/// * `Ok(())` - Process terminated
/// * `Err` - Failed to kill process
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::force_shutdown;
///
/// # async fn example() -> anyhow::Result<()> {
/// force_shutdown(12345, "rbee-hive", 5, Some("job-123")).await?;
/// # Ok(())
/// # }
/// ```
pub async fn force_shutdown(
    pid: u32,
    daemon_name: &str,
    timeout_secs: u64,
    job_id: Option<&str>,
) -> Result<()> {
    // TEAM-311: Migrated to n!() macro
    let ctx = job_id.map(|jid| NarrationContext::new().with_job_id(jid));
    
    let force_impl = async {
        // Step 1: Send SIGTERM
        n!("daemon_sigterm", "🛑 Sending SIGTERM to {} (PID: {})", daemon_name, pid);

    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;

        let pid_nix = Pid::from_raw(pid as i32);

            // Send SIGTERM
            if let Err(e) = kill(pid_nix, Signal::SIGTERM) {
                n!("daemon_sigterm_failed", "⚠️  Failed to send SIGTERM: {}", e);
                anyhow::bail!("Failed to send SIGTERM: {}", e);
            }

            // Step 2: Wait for graceful shutdown
            sleep(Duration::from_secs(timeout_secs)).await;

            // Step 3: Check if still running, send SIGKILL if needed
            match kill(pid_nix, Signal::SIGTERM) {
                Err(nix::errno::Errno::ESRCH) => {
                    // Process not found = already terminated
                    n!("daemon_terminated", "✅ {} terminated gracefully (PID: {})", daemon_name, pid);
                    Ok(())
                }
                _ => {
                    // Still running, send SIGKILL
                    n!("daemon_sigkill", "⚠️  {} did not stop gracefully, sending SIGKILL (PID: {})", daemon_name, pid);

                    kill(pid_nix, Signal::SIGKILL)?;

                    n!("daemon_killed", "✅ {} killed (PID: {})", daemon_name, pid);
                    Ok(())
                }
            }
        }

        #[cfg(not(unix))]
        {
            anyhow::bail!("force_shutdown only supported on Unix systems")
        }
    };
    
    // Execute with context if job_id provided
    if let Some(ctx) = ctx {
        with_narration_context(ctx, force_impl).await
    } else {
        force_impl.await
    }
}
