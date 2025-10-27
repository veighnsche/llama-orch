//! Daemon status checking
//!
//! TEAM-276: Extracted from queen-lifecycle and hive-lifecycle
//! TEAM-328: Consolidated status.rs and get.rs into health.rs (RULE ZERO)
//! TEAM-329: Renamed health.rs → status.rs (checking status, not health)
//! TEAM-329: Moved poll_daemon_health to utils/poll.rs (polling is a utility)
//!
//! Provides simple HTTP status checking for daemons.

use reqwest::Client;
use std::time::Duration;

// TEAM-329: Re-export status types from types module (PARITY)
pub use crate::types::status::{StatusRequest, StatusResponse};

/// Check if daemon is healthy by querying its health endpoint
///
/// # Arguments
/// * `base_url` - Base URL of daemon (e.g., "http://localhost:8500")
/// * `health_endpoint` - Optional health endpoint path (default: "/health")
/// * `timeout` - Optional timeout duration (default: 2 seconds)
///
/// # Returns
/// * `true` if daemon responds with 2xx status
/// * `false` if daemon is unreachable or returns error status
pub async fn check_daemon_health(
    base_url: &str,
    health_endpoint: Option<&str>,
    timeout: Option<Duration>,
) -> bool {
    let endpoint = health_endpoint.unwrap_or("/health");
    let timeout = timeout.unwrap_or(Duration::from_secs(2));

    let client = match Client::builder().timeout(timeout).build() {
        Ok(c) => c,
        Err(_) => return false,
    };

    let url = format!("{}{}", base_url, endpoint);

    match client.get(&url).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

// TEAM-329: poll_daemon_health moved to utils/poll.rs (polling is a utility)
