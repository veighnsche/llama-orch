//! Queen status operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs

use anyhow::Result;
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("queen-life");

/// Check queen-rbee daemon status
///
/// Queries the /health endpoint to determine if queen is running.
///
/// # Arguments
/// * `queen_url` - Base URL for queen (e.g., "http://localhost:8500")
///
/// # Returns
/// * `Ok(())` - Status check completed (prints result to stdout)
/// * `Err` - Error during status check
pub async fn check_queen_status(queen_url: &str) -> Result<()> {
    // TEAM-186: Check queen-rbee health endpoint
    let client = reqwest::Client::builder()
        .timeout(tokio::time::Duration::from_secs(5))
        .build()?;

    match client.get(format!("{}/health", queen_url)).send().await {
        Ok(response) if response.status().is_success() => {
            NARRATE
                .action("queen_status")
                .context(queen_url)
                .human("✅ Queen is running on {}")
                .emit();

            // Try to get more details from the response
            if let Ok(body) = response.text().await {
                println!("Status: {}", body);
            }
            Ok(())
        }
        Ok(response) => {
            NARRATE
                .action("queen_status")
                .context(response.status().to_string())
                .human("⚠️  Queen responded with status: {}")
                .emit();
            Ok(())
        }
        Err(_) => {
            NARRATE
                .action("queen_status")
                .context(queen_url)
                .human("❌ Queen is not running on {}")
                .emit();
            Ok(())
        }
    }
}
