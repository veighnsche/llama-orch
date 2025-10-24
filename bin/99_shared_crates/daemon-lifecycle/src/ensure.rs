//! Ensure daemon running pattern
//!
//! TEAM-259: Extracted from lib.rs for better organization
//! TEAM-276: Enhanced to support Handle pattern for cleanup tracking
//!
//! Provides the "ensure daemon is running" pattern used across:
//! - rbee-keeper → queen-rbee (via queen-lifecycle)
//! - queen-rbee → rbee-hive (via hive-lifecycle)
//! - (future) rbee-hive → llm-worker

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use std::time::Duration;

use crate::health::is_daemon_healthy;

// TEAM-197: Migrated to narration-core v0.5.0 pattern
const NARRATE: NarrationFactory = NarrationFactory::new("dmn-life");

/// Ensure daemon is running, auto-start if needed
///
/// TEAM-259: Extracted from ensure_queen_running and ensure_hive_running
///
/// This function implements the "ensure daemon running" pattern:
/// 1. Check if daemon is healthy (HTTP /health endpoint)
/// 2. If not running, spawn daemon using provided callback
/// 3. Wait for health check to pass (with timeout)
///
/// # Arguments
/// * `daemon_name` - Name of daemon (for logging)
/// * `base_url` - Base URL of daemon (e.g., "http://localhost:8500")
/// * `job_id` - Optional job ID for narration routing
/// * `spawn_fn` - Async function to spawn the daemon
/// * `timeout` - Max time to wait for health (default: 30 seconds)
/// * `poll_interval` - Health check interval (default: 500ms)
///
/// # Returns
/// * `Ok(true)` - Daemon was already running
/// * `Ok(false)` - Daemon was started by us
/// * `Err` - Failed to start daemon or timeout
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::ensure_daemon_running;
/// use anyhow::Result;
///
/// # async fn example() -> Result<()> {
/// ensure_daemon_running(
///     "queen-rbee",
///     "http://localhost:8500",
///     None,
///     || async {
///         // Spawn queen daemon here
///         Ok(())
///     },
///     None,
///     None,
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn ensure_daemon_running<F, Fut>(
    daemon_name: &str,
    base_url: &str,
    job_id: Option<&str>,
    spawn_fn: F,
    timeout: Option<Duration>,
    poll_interval: Option<Duration>,
) -> Result<bool>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let timeout = timeout.unwrap_or(Duration::from_secs(30));
    let poll_interval = poll_interval.unwrap_or(Duration::from_millis(500));

    // Check if daemon is already healthy
    if is_daemon_healthy(base_url, None, None).await {
        let mut narration = NARRATE.action("daemon_check").context(daemon_name);
        if let Some(jid) = job_id {
            narration = narration.job_id(jid);
        }
        narration.human(&format!("{} is already running", daemon_name)).emit();
        return Ok(true); // Already running
    }

    // Daemon is not running, start it
    let mut narration = NARRATE.action("daemon_start").context(daemon_name);
    if let Some(jid) = job_id {
        narration = narration.job_id(jid);
    }
    narration.human(&format!("⚠️  {} is not running, starting...", daemon_name)).emit();

    // Spawn daemon
    spawn_fn().await?;

    // Wait for daemon to become healthy (with timeout)
    let start_time = std::time::Instant::now();

    loop {
        if is_daemon_healthy(base_url, None, None).await {
            let mut narration = NARRATE.action("daemon_start").context(daemon_name);
            if let Some(jid) = job_id {
                narration = narration.job_id(jid);
            }
            narration.human(&format!("✅ {} is now running and healthy", daemon_name)).emit();
            return Ok(false); // Started by us
        }

        if start_time.elapsed() > timeout {
            return Err(anyhow::anyhow!(
                "Timeout waiting for {} to become healthy (waited {:?})",
                daemon_name,
                timeout
            ));
        }

        tokio::time::sleep(poll_interval).await;
    }
}

/// Ensure daemon running with Handle pattern
///
/// TEAM-276: Generic ensure pattern that returns a handle for cleanup tracking
///
/// This is the preferred pattern for lifecycle crates. It:
/// 1. Checks if daemon is healthy
/// 2. If not running, calls spawn_fn to start it
/// 3. Returns a handle indicating if we started it (for cleanup)
///
/// # Arguments
/// * `daemon_name` - Name of daemon (for logging)
/// * `health_url` - Health check URL (e.g., "http://localhost:8500/health")
/// * `job_id` - Optional job ID for narration routing
/// * `spawn_fn` - Async function to spawn the daemon
/// * `handle_already_running` - Function to create handle for already-running daemon
/// * `handle_started_by_us` - Function to create handle for daemon we started
///
/// # Returns
/// * `Ok(H)` - Handle to daemon (tracks if we started it)
/// * `Err` - Failed to start daemon
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::ensure_daemon_with_handle;
/// use anyhow::Result;
///
/// struct MyHandle { started_by_us: bool }
///
/// # async fn example() -> Result<()> {
/// let handle = ensure_daemon_with_handle(
///     "queen-rbee",
///     "http://localhost:8500/health",
///     None,
///     || async {
///         // Spawn queen daemon here
///         Ok(())
///     },
///     || MyHandle { started_by_us: false },
///     || MyHandle { started_by_us: true },
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn ensure_daemon_with_handle<F, Fut, H, AR, SU>(
    daemon_name: &str,
    health_url: &str,
    job_id: Option<&str>,
    spawn_fn: F,
    handle_already_running: AR,
    handle_started_by_us: SU,
) -> Result<H>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
    AR: FnOnce() -> H,
    SU: FnOnce() -> H,
{
    // Check if daemon is already healthy
    if is_daemon_healthy(health_url, None, Some(Duration::from_secs(2))).await {
        let mut narration = NARRATE.action("daemon_already_running").context(daemon_name);
        if let Some(jid) = job_id {
            narration = narration.job_id(jid);
        }
        narration.human(format!("✅ {} is already running and healthy", daemon_name)).emit();
        return Ok(handle_already_running());
    }

    // Daemon is not running, start it
    let mut narration = NARRATE.action("daemon_not_running").context(daemon_name);
    if let Some(jid) = job_id {
        narration = narration.job_id(jid);
    }
    narration.human(format!("⚠️  {} is not running, starting...", daemon_name)).emit();

    // Spawn daemon
    spawn_fn().await?;

    // Daemon started successfully
    let mut narration = NARRATE.action("daemon_started").context(daemon_name);
    if let Some(jid) = job_id {
        narration = narration.job_id(jid);
    }
    narration.human(format!("✅ {} started and healthy", daemon_name)).emit();

    Ok(handle_started_by_us())
}
