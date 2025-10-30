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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tauri", derive(specta::Type))]
pub struct DaemonStatus {
    /// Whether daemon is currently running
    pub is_running: bool,

    /// Whether daemon binary is installed
    pub is_installed: bool,
}

impl DaemonStatus {
    /// Create new DaemonStatus
    pub fn new(is_running: bool, is_installed: bool) -> Self {
        Self { is_running, is_installed }
    }

    /// Create status for running daemon (must be installed to run)
    pub fn running() -> Self {
        Self { is_running: true, is_installed: true }
    }

    /// Create status for stopped daemon (installed but not running)
    pub fn stopped_installed() -> Self {
        Self { is_running: false, is_installed: true }
    }

    /// Create status for not installed daemon
    pub fn not_installed() -> Self {
        Self { is_running: false, is_installed: false }
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
