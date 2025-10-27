//! Stop daemon on remote machine
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
//! use remote_daemon_lifecycle::{stop_daemon_remote, SshConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
//! stop_daemon_remote(ssh, "rbee-hive", "http://192.168.1.100:7835").await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use crate::SshConfig;

/// Stop daemon on remote machine
///
/// TODO: Implement
/// - Try graceful shutdown via HTTP shutdown endpoint
/// - Fallback to force kill via SSH
pub async fn stop_daemon_remote(
    _ssh_config: SshConfig,
    _daemon_name: &str,
    _shutdown_url: &str,
) -> Result<()> {
    anyhow::bail!("stop_daemon_remote: NOT YET IMPLEMENTED")
}
