//! Check daemon status on remote machine
//!
//! # Types/Utils Used (from daemon-lifecycle)
//! - None (uses reqwest directly for HTTP health checks)
//! - Provides check_daemon_health() for utils::poll::poll_daemon_health()
//!
//! # Requirements
//!
//! ## Input
//! - `health_url`: Health endpoint URL (e.g., "http://192.168.1.100:7835/health")
//!
//! ## Process
//! 1. HTTP GET to health endpoint (NO SSH)
//!    - Use: `reqwest::get(health_url)`
//!    - Timeout: 2 seconds
//!    - Return: true if 200 OK, false otherwise
//!
//! ## SSH Calls
//! - Total: 0 SSH calls (HTTP only)
//!
//! ## Error Handling
//! - Connection timeout → return false (daemon not running)
//! - Connection refused → return false (daemon not running)
//! - HTTP error → return false (daemon unhealthy)
//!
//! ## Example
//! ```rust,no_run
//! use remote_daemon_lifecycle::check_daemon_status;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let is_running = check_daemon_status("http://192.168.1.100:7835/health").await?;
//! if is_running {
//!     println!("Daemon is running");
//! } else {
//!     println!("Daemon is not running");
//! }
//! # Ok(())
//! # }
//! ```

use crate::utils::binary::check_binary_installed;
use crate::SshConfig;

// TEAM-367: Import shared types and utilities
pub use lifecycle_shared::{check_health_http, DaemonStatus};

/// Check daemon health and installation status
///
/// TEAM-338: RULE ZERO - Updated existing function (was returning bool, now returns DaemonStatus)
/// BREAKING CHANGE: Callers must now handle DaemonStatus struct
///
/// # Arguments
/// - `health_url`: Health endpoint URL (e.g., "http://localhost:7833/health")
/// - `daemon_name`: Name of daemon binary (e.g., "queen-rbee")
/// - `ssh_config`: SSH config for remote check (use SshConfig::localhost() for local)
///
/// # Implementation
/// 1. Check if running via HTTP health endpoint (no SSH)
/// 2. Check if installed:
///    - If localhost: Check ~/.local/bin/{daemon_name} exists
///    - If remote: SSH command `test -f ~/.local/bin/{daemon_name}`
///
/// # Optimization
/// If daemon is running, skip installation check (must be installed to run)
pub async fn check_daemon_health(
    health_url: &str,
    daemon_name: &str,
    ssh_config: &SshConfig,
) -> DaemonStatus {
    // Step 1: Check if running via HTTP (TEAM-367: Use shared function)
    let is_running = check_health_http(health_url).await;

    // Step 2: Check if installed (only if not running - optimization)
    let is_installed = if is_running {
        true // If running, must be installed
    } else {
        check_binary_installed(daemon_name, ssh_config).await
    };

    DaemonStatus { is_running, is_installed }
}
