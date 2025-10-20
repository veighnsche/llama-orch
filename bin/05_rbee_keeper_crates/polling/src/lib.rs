//! Health polling utilities for rbee-keeper
//!
//! This crate provides polling functionality to wait for services (like queen-rbee)
//! to become healthy before proceeding with operations.
//!
//! Created by: TEAM-135
//! Modified by: TEAM-162

use anyhow::Result;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info};

/// Wait for queen-rbee to become healthy
///
/// Polls the health endpoint until it returns success or max attempts reached.
///
/// # Arguments
/// * `port` - The port queen-rbee is running on
/// * `max_attempts` - Maximum number of polling attempts (default: 20)
/// * `interval_ms` - Milliseconds between attempts (default: 500)
///
/// # Returns
/// * `Ok(())` if queen becomes healthy
/// * `Err` if max attempts exceeded or connection fails
pub async fn wait_for_queen_health(
    port: u16,
    max_attempts: Option<u32>,
    interval_ms: Option<u64>,
) -> Result<()> {
    let max_attempts = max_attempts.unwrap_or(20);
    let interval = Duration::from_millis(interval_ms.unwrap_or(500));

    info!("⏳ Waiting for queen-rbee health on port {}...", port);

    let client = reqwest::Client::new();
    let health_url = format!("http://localhost:{}/health", port);

    for attempt in 1..=max_attempts {
        debug!("Health check attempt {}/{}", attempt, max_attempts);

        match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                info!("✅ Queen healthy after {} attempts", attempt);
                return Ok(());
            }
            Ok(response) => {
                debug!("Health check returned status: {}", response.status());
            }
            Err(e) => {
                debug!("Health check failed: {}", e);
            }
        }

        if attempt < max_attempts {
            sleep(interval).await;
        }
    }

    anyhow::bail!(
        "Queen failed to become healthy after {} attempts ({}s timeout)",
        max_attempts,
        (max_attempts as u64 * interval.as_millis() as u64) / 1000
    )
}

/// Simple wrapper with default settings
pub async fn wait_for_queen(port: u16) -> Result<()> {
    wait_for_queen_health(port, None, None).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wait_for_queen_timeout() {
        // Should fail quickly on non-existent port
        let result = wait_for_queen_health(9999, Some(2), Some(100)).await;
        assert!(result.is_err());
    }
}
