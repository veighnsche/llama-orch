//! Start daemon on remote machine via SSH
//!
//! # Requirements
//!
//! ## Input
//! - `daemon_name`: Name of daemon binary (e.g., "rbee-hive")
//! - `ssh_config`: SSH connection details (hostname, user, port)
//! - `daemon_config`: Daemon configuration (args, health_url, etc.)
//!
//! ## Process
//! 1. Find binary on remote machine (ONE ssh call)
//!    - Try: ~/.local/bin/{daemon}
//!    - Try: target/debug/{daemon}
//!    - Try: target/release/{daemon}
//!    - Return error if not found
//!
//! 2. Start daemon in background (ONE ssh call)
//!    - Use: `nohup {binary} {args} > /dev/null 2>&1 & echo $!`
//!    - Capture PID from stdout
//!
//! 3. Poll health endpoint via HTTP (NO SSH)
//!    - Use: `http://{hostname}:{port}/health`
//!    - Retry with exponential backoff (10 attempts)
//!    - Use daemon-lifecycle's poll_daemon_health()
//!
//! 4. Return PID
//!
//! ## SSH Calls
//! - Total: 2 SSH calls (find binary, start daemon)
//! - Health polling: HTTP only (no SSH)
//!
//! ## Error Handling
//! - Binary not found on remote
//! - SSH connection failed
//! - Daemon failed to start
//! - Health check timeout
//!
//! ## Example
//! ```rust,no_run
//! use remote_daemon_lifecycle::{start_daemon_remote, SshConfig};
//! use daemon_lifecycle::HttpDaemonConfig;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
//! let config = HttpDaemonConfig::new("rbee-hive", "http://192.168.1.100:7835")
//!     .with_args(vec!["--port".to_string(), "7835".to_string()]);
//!
//! let pid = start_daemon_remote(ssh, config).await?;
//! println!("Started daemon with PID: {}", pid);
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use crate::SshConfig;
use daemon_lifecycle::HttpDaemonConfig;

/// Start daemon on remote machine via SSH
///
/// TODO: Implement
/// - Find binary on remote (ssh call 1)
/// - Start daemon in background (ssh call 2)
/// - Poll health via HTTP (no ssh)
/// - Return PID
pub async fn start_daemon_remote(
    _ssh_config: SshConfig,
    _daemon_config: HttpDaemonConfig,
) -> Result<u32> {
    anyhow::bail!("start_daemon_remote: NOT YET IMPLEMENTED")
}
