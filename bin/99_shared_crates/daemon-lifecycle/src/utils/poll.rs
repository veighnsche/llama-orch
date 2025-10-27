//! Polling utilities for daemon readiness
//!
//! TEAM-329: Extracted from health.rs - polling is a utility, not health checking

use observability_narration_core::n;
use observability_narration_macros::with_job_id;
use std::time::Duration;
use tokio::time::sleep;

use crate::status::check_daemon_health;
use crate::types::status::HealthPollConfig; // TEAM-329: types/status.rs (PARITY)

/// Poll daemon health endpoint until healthy or max attempts reached
///
/// TEAM-276: Added for daemon startup synchronization with exponential backoff
/// TEAM-329: Moved from health.rs to utils/poll.rs (polling is a utility)
///
/// Uses exponential backoff:
/// - Attempt 1: 200ms
/// - Attempt 2: 300ms
/// - Attempt 3: 450ms
/// - Attempt 4: 675ms
/// - ...
///
/// # Arguments
/// * `config` - Health polling configuration
///
/// # Returns
/// * `Ok(())` - Daemon is healthy
/// * `Err` - Daemon failed to become healthy after max attempts
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{poll_daemon_health, HealthPollConfig};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = HealthPollConfig::new("http://localhost:8500")
///     .with_max_attempts(10)
///     .with_job_id("job-123");
///
/// poll_daemon_health(config).await?;
/// # Ok(())
/// # }
/// ```
#[with_job_id] // TEAM-328: Eliminates job_id context boilerplate
pub async fn poll_daemon_health(config: HealthPollConfig) -> anyhow::Result<()> {
    let daemon_name = config.daemon_name.as_deref().unwrap_or("daemon");

    // Emit start narration
    n!(
        "daemon_health_poll",
        "⏳ Waiting for {} to become healthy at {}",
        daemon_name,
        config.base_url
    );

    for attempt in 1..=config.max_attempts {
        // Check health
        if check_daemon_health(
            &config.base_url,
            config.health_endpoint.as_deref(),
            Some(Duration::from_secs(2)),
        )
        .await
        {
            // Success!
            n!("daemon_healthy", "✅ {} is healthy (attempt {})", daemon_name, attempt);
            return Ok(());
        }

        // Not healthy yet, calculate delay with exponential backoff
        if attempt < config.max_attempts {
            let delay_ms = (config.initial_delay_ms as f64
                * config.backoff_multiplier.powi((attempt - 1) as i32))
                as u64;
            let delay = Duration::from_millis(delay_ms);

            // Emit progress narration
            n!(
                "daemon_poll_retry",
                "🔄 Attempt {}/{}, retrying in {}ms...",
                attempt,
                config.max_attempts,
                delay_ms
            );

            sleep(delay).await;
        }
    }

    // Failed after max attempts
    n!(
        "daemon_not_healthy",
        "❌ {} failed to become healthy after {} attempts",
        daemon_name,
        config.max_attempts
    );

    anyhow::bail!(
        "{} at {} failed to become healthy after {} attempts",
        daemon_name,
        config.base_url,
        config.max_attempts
    )
}
