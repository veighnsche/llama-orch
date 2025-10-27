//! Daemon shutdown utilities
//!
//! TEAM-276: Extracted pattern from queen-lifecycle and hive-lifecycle
//! TEAM-329: Simplified to signal-based shutdown only (removed HTTP-based shutdown)

use anyhow::Result;
use observability_narration_core::n;
use std::time::Duration;
use tokio::time::sleep;

// TEAM-329: Re-export ShutdownConfig from types module
pub use crate::types::shutdown::ShutdownConfig;

/// Shutdown a daemon process by PID (SIGTERM → SIGKILL)
///
/// TEAM-276: Extracted pattern from hive-lifecycle/stop.rs
/// TEAM-329: Renamed from shutdown_daemon_force (simplified naming)
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
/// use daemon_lifecycle::shutdown_daemon;
///
/// # async fn example() -> anyhow::Result<()> {
/// shutdown_daemon(12345, "rbee-hive", 5, Some("job-123")).await?;
/// # Ok(())
/// # }
/// ```
pub async fn shutdown_daemon(
    pid: u32,
    daemon_name: &str,
    timeout_secs: u64,
    _job_id: Option<&str>,
) -> Result<()> {
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
        anyhow::bail!("shutdown_daemon only supported on Unix systems")
    }
}

