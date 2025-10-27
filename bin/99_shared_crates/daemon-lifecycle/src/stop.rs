//! Stop HTTP-based daemons
//!
//! TEAM-316: Extracted from lifecycle.rs (RULE ZERO - single responsibility)
//! TEAM-327: Migrated to signal-based shutdown (no more HTTP /v1/shutdown endpoint)

use anyhow::Result;

use crate::shutdown::shutdown_daemon;
use crate::status::check_daemon_health; // TEAM-329: Renamed from health to status
use crate::types::start::HttpDaemonConfig; // TEAM-329: types/start.rs (renamed from lifecycle.rs)
use crate::utils::pid::{read_pid_file, remove_pid_file}; // TEAM-329: Centralized PID operations

/// Stop an HTTP-based daemon gracefully using signals
///
/// TEAM-276: High-level function for graceful daemon shutdown
/// TEAM-316: Extracted from lifecycle.rs
/// TEAM-327: Migrated to signal-based shutdown (SIGTERM → SIGKILL)
///
/// Steps:
/// 1. Check if daemon is running (via health endpoint)
/// 2. Read PID from file (~/.local/var/run/{daemon}.pid) for stateless shutdown
/// 3. Send SIGTERM to process
/// 4. Wait for graceful timeout, send SIGKILL if needed
/// 5. Clean up PID file
///
/// # Arguments
/// * `config` - HTTP daemon configuration (uses health_url and pid)
///
/// # Returns
/// * `Ok(())` - Daemon stopped successfully
/// * `Err` - Unexpected error during shutdown or PID not available
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{stop_http_daemon, HttpDaemonConfig};
/// use std::path::PathBuf;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = HttpDaemonConfig::new(
///     "queen-rbee",
///     PathBuf::from("target/release/queen-rbee"),
///     "http://localhost:8500",
/// )
/// .with_pid(12345)
/// .with_job_id("job-123");
///
/// stop_http_daemon(config).await?;
/// # Ok(())
/// # }
/// ```
pub async fn stop_daemon(config: HttpDaemonConfig) -> Result<()> {
    // Step 1: Check if daemon is running (via health check)
    let is_running = check_daemon_health(&config.health_url, None, None).await;
    
    if !is_running {
        use observability_narration_core::n;
        n!("daemon_not_running", "⚠️  {} not running", config.daemon_name);
        // Clean up stale PID file if it exists
        let _ = remove_pid_file(&config.daemon_name);
        return Ok(());
    }
    
    // Step 2: Get PID from PID file (stateless shutdown)
    // TEAM-327: Read from ~/.local/var/run/{daemon}.pid
    let pid = match config.pid {
        Some(pid) => pid, // Use provided PID if available
        None => read_pid_file(&config.daemon_name)?, // Otherwise read from file
    };
    
    // Step 3: Use signal-based shutdown (SIGTERM → SIGKILL)
    let timeout_secs = config.graceful_timeout_secs.unwrap_or(5);
    shutdown_daemon(pid, &config.daemon_name, timeout_secs, config.job_id.as_deref()).await?;
    
    // Step 4: Clean up PID file after successful shutdown
    remove_pid_file(&config.daemon_name)?;
    
    Ok(())
}

