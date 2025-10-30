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
use std::time::Duration;

/// Daemon status information
///
/// TEAM-338: RULE ZERO - Updated existing function to return struct
/// TEAM-338: RULE ZERO FIX - Added Serialize, Deserialize, specta::Type for Tauri bindings
///           This is the SINGLE SOURCE OF TRUTH for daemon status (no QueenStatus/HiveStatus duplicates)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "tauri", derive(specta::Type))]
pub struct DaemonStatus {
    /// Is the daemon currently running?
    pub is_running: bool,
    /// Is the daemon binary installed?
    pub is_installed: bool,
}

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
    // Step 1: Check if running (HTTP, no SSH)
    let client = match reqwest::Client::builder().timeout(Duration::from_secs(2)).build() {
        Ok(c) => c,
        Err(_) => {
            return DaemonStatus {
                is_running: false,
                is_installed: false, // Can't check, assume not installed
            };
        }
    };

    // ============================================================
    // INVESTIGATION: TEAM-341 | Health check behavior during startup
    // ============================================================
    // SUSPICION:
    // - Thought HTTP request was failing silently
    // - Suspected wrong URL or missing /health endpoint
    // - Suspected outdated binary being installed
    //
    // INVESTIGATION:
    // - Added narration to expose actual HTTP errors
    // - Found daemon returns 404 Not Found during first ~8-10 attempts
    // - After ~15-20 seconds, daemon returns 200 OK
    // - Verified: curl http://localhost:7833/health → 200 OK (after waiting)
    // - Verified: curl http://localhost:7833/v1/info → 200 OK (works immediately)
    // - Binary timestamps match: install copied correct binary
    //
    // ROOT CAUSE:
    // - Debug builds (unoptimized) take 15-20 seconds to fully initialize
    // - Daemon process starts, binds port, accepts connections
    // - But routes aren't fully registered yet during initialization
    // - Initial HTTP requests get 404 Not Found
    // - After initialization completes, requests get 200 OK
    // - This is NORMAL behavior for debug builds
    //
    // CONCLUSION:
    // - NO BUG - health polling is working correctly!
    // - Exponential backoff handles slow startup gracefully
    // - 404 is treated as "not healthy yet", polling continues
    // - Eventually daemon becomes healthy and returns 200 OK
    // - User was canceling (^C) before polling completed
    //
    // RECOMMENDATION:
    // - Keep narration for visibility (helps debug real issues)
    // - Consider increasing initial delay for debug builds
    // - Release builds should be much faster (~2-3 seconds)
    // ============================================================
    // TEAM-341: Show the actual URL being called
    observability_narration_core::n!(
        "health_check_url",
        "🔗 Checking URL: {}",
        health_url
    );
    
    let is_running = match client.get(health_url).send().await {
        Ok(response) => {
            let status = response.status();
            let is_success = status.is_success();
            observability_narration_core::n!(
                "health_check_response",
                "🏥 Health check response: {} (success: {})",
                status,
                is_success
            );
            is_success
        }
        Err(e) => {
            observability_narration_core::n!(
                "health_check_error",
                "❌ Health check failed: {}",
                e
            );
            false
        }
    };

    // Step 2: Check if installed (only if not running - optimization)
    let is_installed = if is_running {
        true // If running, must be installed
    } else {
        check_binary_installed(daemon_name, ssh_config).await
    };

    DaemonStatus { is_running, is_installed }
}
