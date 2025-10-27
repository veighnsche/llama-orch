//! Uninstall daemon binary from remote machine
//!
//! # Requirements
//!
//! ## Input
//! - `daemon_name`: Name of daemon binary to remove
//! - `ssh_config`: SSH connection details
//! - `health_url`: Optional health URL to check if daemon is running
//!
//! ## Process
//! 1. Check if daemon is running (HTTP, NO SSH)
//!    - If running: return error (must stop daemon first)
//!    - If not running: continue
//!
//! 2. Remove binary from remote machine (ONE ssh call)
//!    - Use: `rm -f ~/.local/bin/{daemon_name}`
//!    - Return Ok even if file doesn't exist
//!
//! ## SSH Calls
//! - Total: 1 SSH call (rm command)
//! - Health check: HTTP only (no SSH)
//!
//! ## Error Handling
//! - Daemon still running (must stop first)
//! - SSH connection failed
//! - Permission denied (can't delete file)
//!
//! ## Example
//! ```rust,no_run
//! use remote_daemon_lifecycle::{uninstall_daemon_remote, SshConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
//! uninstall_daemon_remote(
//!     "rbee-hive",
//!     ssh,
//!     Some("http://192.168.1.100:7835/health")
//! ).await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use crate::SshConfig;

/// Uninstall daemon binary from remote machine
///
/// TODO: Implement
/// - Check if daemon is running (HTTP)
/// - Remove binary via SSH
pub async fn uninstall_daemon_remote(
    _daemon_name: &str,
    _ssh_config: SshConfig,
    _health_url: Option<&str>,
) -> Result<()> {
    anyhow::bail!("uninstall_daemon_remote: NOT YET IMPLEMENTED")
}
