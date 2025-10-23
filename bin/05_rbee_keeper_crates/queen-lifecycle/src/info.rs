//! Queen build info operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs
//! TEAM-262: Query queen's /v1/build-info endpoint

use anyhow::Result;
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("queen-life");

/// Get queen-rbee build information
///
/// Queries the /v1/build-info endpoint to get build configuration details.
///
/// # Arguments
/// * `queen_url` - Base URL for queen (e.g., "http://localhost:8500")
///
/// # Returns
/// * `Ok(())` - Build info retrieved (prints to stdout)
/// * `Err` - Error during query
pub async fn get_queen_info(queen_url: &str) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(tokio::time::Duration::from_secs(5))
        .build()?;

    match client.get(format!("{}/v1/build-info", queen_url)).send().await {
        Ok(response) if response.status().is_success() => {
            NARRATE.action("queen_info").human("📋 Queen build information:").emit();
            if let Ok(body) = response.text().await {
                println!("{}", body);
            }
            Ok(())
        }
        Err(_) => {
            NARRATE
                .action("queen_info")
                .human("❌ Queen is not running or /v1/build-info not available")
                .emit();
            Ok(())
        }
        _ => {
            NARRATE.action("queen_info").human("⚠️  Failed to get build info").emit();
            Ok(())
        }
    }
}
