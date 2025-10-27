//! Stop HTTP-based daemons
//!
//! TEAM-316: Extracted from lifecycle.rs (RULE ZERO - single responsibility)
//! TEAM-327: Migrated to signal-based shutdown (no more HTTP /v1/shutdown endpoint)

use anyhow::Result;
use daemon_contract::HttpDaemonConfig;
use std::path::PathBuf;

use crate::health::is_daemon_healthy;
use crate::shutdown::force_shutdown;

/// Get PID file path for a daemon
///
/// TEAM-327: Standard Unix pattern - PID files in ~/.local/var/run/
/// TEAM-328: Use centralized path function
fn get_pid_file_path(daemon_name: &str) -> Result<PathBuf> {
    crate::paths::get_pid_file_path(daemon_name)
}

/// Read PID from PID file
///
/// TEAM-327: Standard Unix pattern for stateless daemon management
fn read_pid_file(daemon_name: &str) -> Result<u32> {
    let pid_file = get_pid_file_path(daemon_name)?;
    
    if !pid_file.exists() {
        anyhow::bail!("PID file not found: {}. Is {} running?", pid_file.display(), daemon_name);
    }
    
    let pid_str = std::fs::read_to_string(&pid_file)?;
    let pid = pid_str.trim().parse::<u32>()
        .map_err(|e| anyhow::anyhow!("Invalid PID in {}: {}", pid_file.display(), e))?;
    
    Ok(pid)
}

/// Remove PID file after successful shutdown
///
/// TEAM-327: Cleanup PID file to avoid stale entries
fn remove_pid_file(daemon_name: &str) -> Result<()> {
    let pid_file = get_pid_file_path(daemon_name)?;
    if pid_file.exists() {
        std::fs::remove_file(&pid_file)?;
    }
    Ok(())
}

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
pub async fn stop_http_daemon(config: HttpDaemonConfig) -> Result<()> {
    // Step 1: Check if daemon is running (via health check)
    let is_running = is_daemon_healthy(&config.health_url, None, None).await;
    
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
    force_shutdown(pid, &config.daemon_name, timeout_secs, config.job_id.as_deref()).await?;
    
    // Step 4: Clean up PID file after successful shutdown
    remove_pid_file(&config.daemon_name)?;
    
    Ok(())
}

// TEAM-328: Renamed export for consistent naming
/// Alias for stop_http_daemon with consistent naming
pub use stop_http_daemon as stop_daemon;
