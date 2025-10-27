//! Graceful shutdown of daemon on remote machine
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
//! use remote_daemon_lifecycle::{shutdown_daemon_remote, SshConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
//! shutdown_daemon_remote(
//!     "rbee-hive",
//!     "http://192.168.1.100:7835/health",
//!     ssh
//! ).await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use crate::SshConfig;

/// Graceful shutdown of daemon on remote machine
///
/// TODO: Implement
/// - Try HTTP shutdown endpoint
/// - Wait for daemon to stop
/// - Fallback to SIGTERM via SSH
/// - Fallback to SIGKILL via SSH
pub async fn shutdown_daemon_remote(
    _daemon_name: &str,
    _shutdown_url: &str,
    _ssh_config: SshConfig,
) -> Result<()> {
    anyhow::bail!("shutdown_daemon_remote: NOT YET IMPLEMENTED")
}
