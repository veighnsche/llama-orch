//! Queen status operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs
//! TEAM-311: Migrated to n!() macro

use anyhow::Result;
use observability_narration_core::n;

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
    let client = reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(5)).build()?;

    match client.get(format!("{}/health", queen_url)).send().await {
        Ok(response) if response.status().is_success() => {
            n!("queen_status", "✅ Queen is running on {}", queen_url);

            // Try to get more details from the response
            if let Ok(body) = response.text().await {
                println!("Status: {}", body);
            }
            Ok(())
        }
        Ok(response) => {
            n!("queen_status", "⚠️  Queen responded with status: {}", response.status());
            Ok(())
        }
        Err(_) => {
            n!("queen_status", "❌ Queen is not running on {}", queen_url);
            Ok(())
        }
    }
}
