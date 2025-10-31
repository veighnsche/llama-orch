//! Check daemon status on LOCAL machine
//!
//! TEAM-358: Refactored to remove SSH code (lifecycle-local = LOCAL only)
//!
//! # Types/Utils Used
//! - None (uses reqwest directly for HTTP health checks)
//! - Provides check_daemon_health() for health status verification
//!
//! # Requirements
//!
//! ## Input
//! - `health_url`: Health endpoint URL (e.g., "http://localhost:7835/health")
//!
//! ## Process
//! 1. HTTP GET to health endpoint
//!    - Use: `reqwest::get(health_url)`
//!    - Timeout: 2 seconds
//!    - Return: true if 200 OK, false otherwise
//!
//! ## Error Handling
//! - Connection timeout → return false (daemon not running)
//! - Connection refused → return false (daemon not running)
//! - HTTP error → return false (daemon unhealthy)
//!
//! ## Example
//! ```rust,no_run
//! use lifecycle_local::check_daemon_health;
//!
//! # async fn example() {
//! let status = check_daemon_health("http://localhost:7835/health", "rbee-hive").await;
//! if status.is_running {
//!     println!("Daemon is running");
//! } else {
//!     println!("Daemon is not running");
//! }
//! # }
//! ```

// TEAM-377: Use check_binary_actually_installed to only check ~/.local/bin/
use crate::utils::binary::check_binary_actually_installed;

// TEAM-367: Import shared types and utilities
pub use lifecycle_shared::{check_health_http, DaemonStatus};

/// Check daemon health and installation status
///
/// TEAM-338: RULE ZERO - Updated existing function (was returning bool, now returns DaemonStatus)
/// TEAM-358: Removed ssh_config parameter (lifecycle-local = LOCAL only)
/// BREAKING CHANGE: Callers must now handle DaemonStatus struct
///
/// # Arguments
/// - `health_url`: Health endpoint URL (e.g., "http://localhost:7833/health")
/// - `daemon_name`: Name of daemon binary (e.g., "queen-rbee")
///
/// # Implementation
/// 1. Check if running via HTTP health endpoint
/// 2. Check if installed: Check ~/.local/bin/{daemon_name} or target/ exists
///
/// # Optimization
/// If daemon is running, skip installation check (must be installed to run)
pub async fn check_daemon_health(
    health_url: &str,
    daemon_name: &str,
) -> DaemonStatus {
    // Step 1: Check if running via HTTP (TEAM-367: Use shared function)
    let is_running = check_health_http(health_url).await;

    // Step 2: Check if installed (only if not running - optimization)
    // TEAM-377: Check if ACTUALLY installed (only ~/.local/bin/, not dev builds)
    // This determines if "Uninstall" button should be enabled
    let is_installed = if is_running {
        true // If running, must be installed (or dev build, but still "exists")
    } else {
        // Only check ~/.local/bin/, NOT target/debug/ or target/release/
        check_binary_actually_installed(daemon_name).await
    };

    DaemonStatus { is_running, is_installed }
}
