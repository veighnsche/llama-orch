//! Start HTTP-based daemons
//!
//! TEAM-316: Extracted from lifecycle.rs (RULE ZERO - single responsibility)

use anyhow::Result;
use daemon_contract::HttpDaemonConfig;
use tokio::process::Child;

use crate::health::{poll_until_healthy, HealthPollConfig};
use crate::manager::DaemonManager;

/// Start an HTTP-based daemon (spawn + health polling)
///
/// TEAM-276: High-level function that combines spawn and health polling
/// TEAM-316: Extracted from lifecycle.rs
/// TEAM-319: Detaches child process internally (no mem::forget needed)
///
/// Steps:
/// 1. Spawn the daemon process
/// 2. Poll health endpoint until ready
/// 3. Detach the child process (daemon keeps running)
///
/// # Arguments
/// * `config` - HTTP daemon configuration
///
/// # Returns
/// * `Ok(())` - Daemon started, healthy, and detached
/// * `Err` - Failed to start or become healthy
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{start_http_daemon, HttpDaemonConfig};
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
/// start_http_daemon(config).await?;
/// // Daemon is now running and healthy (detached)
/// # Ok(())
/// # }
/// ```
pub async fn start_http_daemon(config: HttpDaemonConfig) -> Result<()> {
    // Step 1: Spawn the daemon
    let manager = DaemonManager::new(config.binary_path.clone(), config.args.clone());
    let mut child = manager.spawn().await?;

    // Step 2: Poll until healthy
    let mut health_config =
        HealthPollConfig::new(&config.health_url).with_daemon_name(&config.daemon_name);

    if let Some(attempts) = config.max_health_attempts {
        health_config = health_config.with_max_attempts(attempts);
    }

    if let Some(job_id) = config.job_id.as_deref() {
        health_config = health_config.with_job_id(job_id);
    }

    poll_until_healthy(health_config).await?;

    // Step 3: Detach the child process
    // The daemon will keep running independently
    // We use ManuallyDrop to prevent Drop from running without mem::forget
    let _ = std::mem::ManuallyDrop::new(child);

    Ok(())
}
