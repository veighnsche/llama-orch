//! Queen stop operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs

use anyhow::Result;
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("queen-life");

/// Stop queen-rbee daemon
///
/// Sends shutdown request to queen's /v1/shutdown endpoint.
/// Handles expected connection errors (queen shuts down before responding).
///
/// # Arguments
/// * `queen_url` - Base URL for queen (e.g., "http://localhost:8500")
///
/// # Returns
/// * `Ok(())` - Queen stopped successfully (or was not running)
/// * `Err` - Unexpected error during shutdown
pub async fn stop_queen(queen_url: &str) -> Result<()> {
    let client = reqwest::Client::new();

    // First check if queen is running
    let health_check = client.get(format!("{}/health", queen_url)).send().await;

    let is_running = matches!(health_check, Ok(response) if response.status().is_success());

    if !is_running {
        NARRATE.action("queen_stop").human("⚠️  Queen not running").emit();
        return Ok(());
    }

    // Queen is running, send shutdown request
    let shutdown_client =
        reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(30)).build()?;

    match shutdown_client.post(format!("{}/v1/shutdown", queen_url)).send().await {
        Ok(_) => {
            NARRATE.action("queen_stop").human("✅ Queen stopped").emit();
            Ok(())
        }
        Err(e) => {
            // Connection closed/reset is expected - queen shuts down before responding
            // TEAM-296: Fixed error detection to handle all connection closure scenarios
            let error_str = e.to_string();
            let is_expected_shutdown = e.is_connect()
                || e.is_request()
                || error_str.contains("connection closed")
                || error_str.contains("connection reset")
                || error_str.contains("broken pipe");

            if is_expected_shutdown {
                NARRATE.action("queen_stop").human("✅ Queen stopped").emit();
                Ok(())
            } else {
                // Unexpected error
                NARRATE
                    .action("queen_stop")
                    .context(error_str)
                    .human("⚠️  Failed to stop queen: {}")
                    .error_kind("shutdown_failed")
                    .emit();
                Err(e.into())
            }
        }
    }
}
