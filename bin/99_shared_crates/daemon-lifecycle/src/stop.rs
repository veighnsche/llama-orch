//! Stop HTTP-based daemons
//!
//! TEAM-316: Extracted from lifecycle.rs (RULE ZERO - single responsibility)

use anyhow::Result;
use daemon_contract::HttpDaemonConfig;

use crate::shutdown::{graceful_shutdown, ShutdownConfig};

/// Stop an HTTP-based daemon gracefully
///
/// TEAM-276: High-level function for graceful daemon shutdown
/// TEAM-316: Extracted from lifecycle.rs
///
/// Steps:
/// 1. Check if daemon is running
/// 2. Send shutdown request to HTTP endpoint
/// 3. Handle expected connection errors
///
/// # Arguments
/// * `config` - HTTP daemon configuration (uses health_url and shutdown_endpoint)
///
/// # Returns
/// * `Ok(())` - Daemon stopped successfully
/// * `Err` - Unexpected error during shutdown
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{stop_http_daemon, HttpDaemonConfig};
/// use std::path::PathBuf;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = HttpDaemonConfig::new(
///     "queen-rbee",
///     PathBuf::from("target/release/queen-rbee"),
///     "http://localhost:8500",
/// )
/// .with_job_id("job-123");
///
/// stop_http_daemon(config).await?;
/// # Ok(())
/// # }
/// ```
pub async fn stop_http_daemon(config: HttpDaemonConfig) -> Result<()> {
    let shutdown_endpoint = config.shutdown_endpoint
        .unwrap_or_else(|| format!("{}/v1/shutdown", config.health_url));
    
    let shutdown_config =
        ShutdownConfig::new(config.daemon_name, config.health_url, shutdown_endpoint);

    let shutdown_config = if let Some(job_id) = config.job_id {
        shutdown_config.with_job_id(job_id)
    } else {
        shutdown_config
    };

    graceful_shutdown(shutdown_config).await
}
