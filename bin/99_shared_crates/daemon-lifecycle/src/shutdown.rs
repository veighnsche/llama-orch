//! Graceful daemon shutdown utilities
//!
//! TEAM-276: Extracted pattern from queen-lifecycle and hive-lifecycle

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use std::time::Duration;
use tokio::time::sleep;

const NARRATE: NarrationFactory = NarrationFactory::new("dmn-life");

/// Configuration for graceful shutdown
///
/// TEAM-276: Pattern from queen-lifecycle/stop.rs and hive-lifecycle/stop.rs
pub struct ShutdownConfig {
    /// Daemon name for narration
    pub daemon_name: String,

    /// Health check URL to verify daemon is running
    pub health_url: String,

    /// Shutdown endpoint (e.g., "/v1/shutdown")
    pub shutdown_endpoint: String,

    /// Timeout for graceful shutdown (SIGTERM) before force kill
    pub sigterm_timeout_secs: u64,

    /// Optional job_id for narration routing
    pub job_id: Option<String>,
}

impl ShutdownConfig {
    /// Create a new shutdown config
    pub fn new(
        daemon_name: impl Into<String>,
        health_url: impl Into<String>,
        shutdown_endpoint: impl Into<String>,
    ) -> Self {
        Self {
            daemon_name: daemon_name.into(),
            health_url: health_url.into(),
            shutdown_endpoint: shutdown_endpoint.into(),
            sigterm_timeout_secs: 5,
            job_id: None,
        }
    }

    /// Set the SIGTERM timeout
    pub fn with_sigterm_timeout(mut self, secs: u64) -> Self {
        self.sigterm_timeout_secs = secs;
        self
    }

    /// Set the job_id for narration
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
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
    // Step 1: Check if daemon is running
    let is_running = crate::health::is_daemon_healthy(
        &config.health_url,
        None,
        Some(Duration::from_secs(2)),
    )
    .await;

    if !is_running {
        // TEAM-276: Using .maybe_job_id() to reduce boilerplate
        NARRATE
            .action("daemon_not_running")
            .context(&config.daemon_name)
            .human(format!("⚠️  {} not running", config.daemon_name))
            .maybe_job_id(config.job_id.as_deref())
            .emit();
        return Ok(());
    }

    // Step 2: Send shutdown request
    // TEAM-276: Using .maybe_job_id() to reduce boilerplate
    NARRATE
        .action("daemon_shutdown")
        .context(&config.daemon_name)
        .human(format!("🛑 Shutting down {}...", config.daemon_name))
        .maybe_job_id(config.job_id.as_deref())
        .emit();

    let shutdown_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;

    match shutdown_client.post(&config.shutdown_endpoint).send().await {
        Ok(_) => {
            // TEAM-276: Using .maybe_job_id() to reduce boilerplate
            NARRATE
                .action("daemon_stopped")
                .context(&config.daemon_name)
                .human(format!("✅ {} stopped", config.daemon_name))
                .maybe_job_id(config.job_id.as_deref())
                .emit();
            Ok(())
        }
        Err(e) => {
            // Connection closed/reset is expected - daemon shuts down before responding
            if e.is_connect() || e.to_string().contains("connection closed") {
                // TEAM-276: Using .maybe_job_id() to reduce boilerplate
                NARRATE
                    .action("daemon_stopped")
                    .context(&config.daemon_name)
                    .human(format!("✅ {} stopped", config.daemon_name))
                    .maybe_job_id(config.job_id.as_deref())
                    .emit();
                Ok(())
            } else {
                // Unexpected error
                // TEAM-276: Using .maybe_job_id() to reduce boilerplate
                NARRATE
                    .action("daemon_shutdown_failed")
                    .context(&config.daemon_name)
                    .context(e.to_string())
                    .human(format!("⚠️  Failed to stop {}: {{}}", config.daemon_name))
                    .error_kind("shutdown_failed")
                    .maybe_job_id(config.job_id.as_deref())
                    .emit();
                Err(e.into())
            }
        }
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
    // Step 1: Send SIGTERM
    // TEAM-276: Using .maybe_job_id() to reduce boilerplate
    NARRATE
        .action("daemon_sigterm")
        .context(pid.to_string())
        .human(format!("🛑 Sending SIGTERM to {} (PID: {})", daemon_name, pid))
        .maybe_job_id(job_id)
        .emit();

    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;

        let pid_nix = Pid::from_raw(pid as i32);

        // Send SIGTERM
        if let Err(e) = kill(pid_nix, Signal::SIGTERM) {
            // TEAM-276: Using .maybe_job_id() to reduce boilerplate
            NARRATE
                .action("daemon_sigterm_failed")
                .context(e.to_string())
                .human(format!("⚠️  Failed to send SIGTERM: {{}}", ))
                .error_kind("sigterm_failed")
                .maybe_job_id(job_id)
                .emit();
            anyhow::bail!("Failed to send SIGTERM: {}", e);
        }

        // Step 2: Wait for graceful shutdown
        sleep(Duration::from_secs(timeout_secs)).await;

        // Step 3: Check if still running, send SIGKILL if needed
        match kill(pid_nix, Signal::SIGTERM) {
            Err(nix::errno::Errno::ESRCH) => {
                // Process not found = already terminated
                // TEAM-276: Using .maybe_job_id() to reduce boilerplate
                NARRATE
                    .action("daemon_terminated")
                    .context(pid.to_string())
                    .human(format!("✅ {} terminated gracefully (PID: {})", daemon_name, pid))
                    .maybe_job_id(job_id)
                    .emit();
                Ok(())
            }
            _ => {
                // Still running, send SIGKILL
                // TEAM-276: Using .maybe_job_id() to reduce boilerplate
                NARRATE
                    .action("daemon_sigkill")
                    .context(pid.to_string())
                    .human(format!("⚠️  {} did not stop gracefully, sending SIGKILL (PID: {})", daemon_name, pid))
                    .maybe_job_id(job_id)
                    .emit();

                kill(pid_nix, Signal::SIGKILL)?;

                // TEAM-276: Using .maybe_job_id() to reduce boilerplate
                NARRATE
                    .action("daemon_killed")
                    .context(pid.to_string())
                    .human(format!("✅ {} killed (PID: {})", daemon_name, pid))
                    .maybe_job_id(job_id)
                    .emit();
                Ok(())
            }
        }
    }

    #[cfg(not(unix))]
    {
        anyhow::bail!("force_shutdown only supported on Unix systems")
    }
}
