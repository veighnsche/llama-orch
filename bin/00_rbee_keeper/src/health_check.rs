//! Health check for queen-rbee
//!
//! Created by: TEAM-151
//! Date: 2025-10-20
//!
//! Simple health probe to check if queen is running.
//! Returns Ok(true) if queen is healthy, Ok(false) if connection refused.
//!
//! # Happy Flow
//! "bee keeper first tests if queen is running? by calling the health."

use anyhow::Result;

/// Check if queen-rbee is running by calling GET /health
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
    
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()?;
    
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
                // Other errors (timeout, etc.)
                Err(anyhow::anyhow!("Health check failed: {}", e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check_when_queen_is_off() {
        // This should return Ok(false) when queen is not running
        let result = is_queen_healthy("http://localhost:8500").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false);
    }
}
