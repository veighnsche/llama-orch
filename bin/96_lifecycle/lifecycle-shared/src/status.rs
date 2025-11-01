//! Daemon status types and utilities shared between lifecycle-local and lifecycle-ssh
//!
//! TEAM-367: Extracted from lifecycle-local/src/status.rs and lifecycle-ssh/src/status.rs

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Daemon status information
///
/// TEAM-338: RULE ZERO - Updated existing function to return struct
/// TEAM-338: RULE ZERO FIX - Added Serialize, Deserialize, specta::Type for Tauri bindings
///           This is the SINGLE SOURCE OF TRUTH for daemon status (no QueenStatus/HiveStatus duplicates)
/// TEAM-378: RULE ZERO - Added SSH config fields (hostname, user, port) for iframe URL construction
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tauri", derive(specta::Type))]
pub struct DaemonStatus {
    /// Whether daemon is currently running
    pub is_running: bool,

    /// Whether daemon binary is installed
    pub is_installed: bool,

    /// Build mode of installed binary ("debug", "release", or None if not installed/unknown)
    /// TEAM-379: Added for UI display
    pub build_mode: Option<String>,

    /// TEAM-378: SSH hostname (IP address or domain) - needed for iframe URL
    /// For localhost, this will be "localhost" or "127.0.0.1"
    /// For remote hives, this is the actual IP/domain from SSH config
    pub hostname: String,

    /// TEAM-378: SSH username - needed for remote operations
    pub user: String,

    /// TEAM-378: SSH port - needed for remote operations
    pub port: u16,
}

impl DaemonStatus {
    /// Create new DaemonStatus
    /// TEAM-378: RULE ZERO - Added hostname, user, port parameters
    pub fn new(
        is_running: bool,
        is_installed: bool,
        build_mode: Option<String>,
        hostname: String,
        user: String,
        port: u16,
    ) -> Self {
        Self {
            is_running,
            is_installed,
            build_mode,
            hostname,
            user,
            port,
        }
    }

    /// Create status for running daemon (must be installed to run)
    /// TEAM-378: RULE ZERO - Added hostname, user, port parameters
    pub fn running(build_mode: Option<String>, hostname: String, user: String, port: u16) -> Self {
        Self {
            is_running: true,
            is_installed: true,
            build_mode,
            hostname,
            user,
            port,
        }
    }

    /// Create status for stopped daemon (installed but not running)
    /// TEAM-378: RULE ZERO - Added hostname, user, port parameters
    pub fn stopped_installed(
        build_mode: Option<String>,
        hostname: String,
        user: String,
        port: u16,
    ) -> Self {
        Self {
            is_running: false,
            is_installed: true,
            build_mode,
            hostname,
            user,
            port,
        }
    }

    /// Create status for not installed daemon
    /// TEAM-378: RULE ZERO - Added hostname, user, port parameters
    pub fn not_installed(hostname: String, user: String, port: u16) -> Self {
        Self {
            is_running: false,
            is_installed: false,
            build_mode: None,
            hostname,
            user,
            port,
        }
    }
}

/// Normalize health URL by ensuring it ends with /health
///
/// TEAM-367: Shared logic extracted from uninstall.rs
///
/// # Example
/// ```rust
/// # use lifecycle_shared::normalize_health_url;
/// assert_eq!(normalize_health_url("http://localhost:7833"), "http://localhost:7833/health");
/// assert_eq!(normalize_health_url("http://localhost:7833/health"), "http://localhost:7833/health");
/// ```
pub fn normalize_health_url(health_url: &str) -> String {
    if health_url.ends_with("/health") {
        health_url.to_string()
    } else {
        format!("{}/health", health_url)
    }
}

/// Check if daemon is running via HTTP health endpoint
///
/// TEAM-367: Shared logic extracted from lifecycle-local and lifecycle-ssh
///
/// # Arguments
/// - `health_url`: Health endpoint URL (e.g., "http://localhost:7833/health")
///
/// # Returns
/// - `true` if daemon responds with 200 OK
/// - `false` if connection fails or non-2xx response
///
/// # Implementation
/// - 2 second timeout
/// - Narrates URL and response for debugging
/// - Treats any error as "not running"
pub async fn check_health_http(health_url: &str) -> bool {
    // Build HTTP client with timeout
    let client = match reqwest::Client::builder().timeout(Duration::from_secs(2)).build() {
        Ok(c) => c,
        Err(_) => return false,
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
    
    match client.get(health_url).send().await {
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
    }
}
