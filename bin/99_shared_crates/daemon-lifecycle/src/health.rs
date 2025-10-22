//! Daemon health checking
//!
//! TEAM-259: Extracted from lib.rs for better organization
//!
//! Provides HTTP-based health checking for daemons.

use std::time::Duration;

/// Check if daemon is healthy via HTTP health endpoint
///
/// TEAM-259: Extracted from is_queen_healthy and is_hive_healthy
///
/// # Arguments
/// * `base_url` - Base URL of daemon (e.g., "http://localhost:8500")
/// * `health_endpoint` - Health endpoint path (default: "/health")
/// * `timeout` - HTTP timeout (default: 2 seconds)
///
/// # Returns
/// * `true` - Daemon is healthy
/// * `false` - Daemon is not responding or unhealthy
pub async fn is_daemon_healthy(
    base_url: &str,
    health_endpoint: Option<&str>,
    timeout: Option<Duration>,
) -> bool {
    let endpoint = health_endpoint.unwrap_or("/health");
    let timeout = timeout.unwrap_or(Duration::from_secs(2));

    let client = match reqwest::Client::builder().timeout(timeout).build() {
        Ok(c) => c,
        Err(_) => return false,
    };

    let url = format!("{}{}", base_url, endpoint);

    match client.get(&url).send().await {
        Ok(response) if response.status().is_success() => true,
        _ => false,
    }
}
