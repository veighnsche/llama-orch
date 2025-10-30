

//! Health polling utility for daemon lifecycle management
//!
//! Provides HTTP health check polling with exponential backoff.
//! Used by all lifecycle crates to verify daemon startup.

use anyhow::{Context, Result};
use std::time::Duration;

/// Poll a health endpoint until it responds successfully
///
/// # Arguments
/// * `url` - Full health endpoint URL (e.g., "http://localhost:7833/health")
/// * `max_attempts` - Maximum number of polling attempts
/// * `initial_delay_ms` - Initial delay between attempts in milliseconds
/// * `backoff_multiplier` - Multiplier for exponential backoff (e.g., 1.5)
///
/// # Returns
/// * `Ok(())` - Health endpoint responded successfully
/// * `Err` - Health check failed after max attempts
///
/// # Example
/// ```rust,no_run
/// use health_poll::poll_health;
///
/// # async fn example() -> anyhow::Result<()> {
/// poll_health(
///     "http://localhost:7833/health",
///     30,    // max attempts
///     200,   // initial delay (ms)
///     1.5,   // backoff multiplier
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn poll_health(
    url: &str,
    max_attempts: usize,
    initial_delay_ms: u64,
    backoff_multiplier: f64,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("Failed to create HTTP client")?;

    let mut delay_ms = initial_delay_ms;

    for attempt in 1..=max_attempts {
        // Wait before attempt (except first)
        if attempt > 1 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            delay_ms = (delay_ms as f64 * backoff_multiplier) as u64;
        }

        tracing::debug!("Health check attempt {}/{}: {}", attempt, max_attempts, url);

        match client.get(url).send().await {
            Ok(response) if response.status().is_success() => {
                tracing::info!("✅ Health check passed: {}", url);
                return Ok(());
            }
            Ok(response) => {
                tracing::debug!(
                    "⏳ Health check failed (attempt {}/{}): HTTP {}",
                    attempt,
                    max_attempts,
                    response.status()
                );
            }
            Err(e) => {
                tracing::debug!(
                    "⏳ Health check failed (attempt {}/{}): {}",
                    attempt,
                    max_attempts,
                    e
                );
            }
        }
    }

    anyhow::bail!(
        "Health check failed after {} attempts: {}",
        max_attempts,
        url
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires running server
    async fn test_poll_health_success() {
        let result = poll_health("http://localhost:7833/health", 5, 100, 1.5).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_poll_health_failure() {
        let result = poll_health("http://localhost:99999/health", 3, 10, 1.5).await;
        assert!(result.is_err());
    }
}
