//! Queen health checking
//!
//! TEAM-259: Extracted from rbee-keeper/src/queen_lifecycle.rs

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use std::time::Duration;
use tokio::time::sleep;

// TEAM-192: Local narration factory for queen lifecycle
const NARRATE: NarrationFactory = NarrationFactory::new("kpr-life");

/// Check if queen is healthy by calling /health endpoint
///
/// # Arguments
/// * `base_url` - Queen URL (e.g., "http://localhost:8500")
///
/// # Returns
/// * `Ok(true)` - Queen is running and healthy
/// * `Ok(false)` - Queen is not running (connection refused)
/// * `Err` - Other errors (timeout, invalid response, etc.)
pub async fn is_queen_healthy(base_url: &str) -> Result<bool> {
    let health_url = format!("{}/health", base_url);

    let client = reqwest::Client::builder().timeout(Duration::from_millis(500)).build()?;

    match client.get(&health_url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                Ok(true)
            } else {
                Ok(false)
            }
        }
        Err(e) => {
            // Connection refused means queen is not running
            if e.is_connect() {
                Ok(false)
            } else {
                Err(e.into())
            }
        }
    }
}

/// Poll health endpoint until queen is ready
///
/// Uses exponential backoff: 100ms, 200ms, 400ms, 800ms, 1600ms, 3200ms
///
/// # Arguments
/// * `base_url` - Queen URL
/// * `timeout` - Maximum time to wait
///
/// # Returns
/// * `Ok(())` - Queen became healthy
/// * `Err` - Timeout or other error
pub async fn poll_until_healthy(base_url: &str, timeout: Duration) -> Result<()> {
    let start = std::time::Instant::now();
    let mut delay = Duration::from_millis(100);
    let max_delay = Duration::from_millis(3200);
    let mut attempt = 0u32;

    loop {
        attempt += 1;

        // Check if we've exceeded timeout
        if start.elapsed() >= timeout {
            NARRATE
                .action("queen_start")
                .context(timeout.as_secs().to_string())
                .human("Queen failed to become healthy within {} seconds")
                .error_kind("startup_timeout")
                .emit();
            anyhow::bail!("Timeout waiting for queen to become healthy");
        }

        // Try health check
        match is_queen_healthy(base_url).await {
            Ok(true) => {
                NARRATE
                    .action("queen_poll")
                    .context(format!("{:?}", start.elapsed()))
                    .human("Queen health check succeeded after {}")
                    .emit();
                return Ok(());
            }
            Ok(false) => {
                NARRATE
                    .action("queen_poll")
                    .context(attempt.to_string())
                    .context(delay.as_millis().to_string())
                    .human("Polling queen health (attempt {}, delay {}ms)")
                    .emit();
            }
            Err(e) => {
                NARRATE
                    .action("queen_poll")
                    .context(e.to_string())
                    .human("Queen health check failed: {}")
                    .error_kind("health_check_failed")
                    .emit();
            }
        }

        // Wait before next attempt
        sleep(delay).await;

        // Exponential backoff (cap at max_delay)
        delay = std::cmp::min(delay * 2, max_delay);
    }
}
