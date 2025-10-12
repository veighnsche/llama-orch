// Created by: TEAM-DX-001
// HTTP client for fetching pages

use crate::error::{DxError, Result};
use reqwest::Client;
use std::time::Duration;

/// HTTP fetcher with timeout and error handling
pub struct Fetcher {
    client: Client,
    timeout: Duration,
}

impl Fetcher {
    /// Create a new fetcher with default timeout (2 seconds)
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(2))
    }
    
    /// Create a fetcher with custom timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to build HTTP client");
        
        Self { client, timeout }
    }
    
    /// Fetch a page and return HTML content
    pub async fn fetch_page(&self, url: &str) -> Result<String> {
        let response = self.client
            .get(url)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    DxError::Timeout {
                        timeout: self.timeout.as_secs(),
                    }
                } else {
                    DxError::Network(e)
                }
            })?;
        
        if !response.status().is_success() {
            return Err(DxError::Network(
                response.error_for_status().unwrap_err()
            ));
        }
        
        let body = response.text().await?;
        Ok(body)
    }
    
    /// Fetch a stylesheet from a URL
    pub async fn fetch_stylesheet(&self, url: &str) -> Result<String> {
        self.fetch_page(url).await
    }
}

impl Default for Fetcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_fetcher_creation() {
        let fetcher = Fetcher::new();
        assert_eq!(fetcher.timeout.as_secs(), 2);
    }
    
    #[tokio::test]
    async fn test_fetcher_with_custom_timeout() {
        let fetcher = Fetcher::with_timeout(Duration::from_secs(5));
        assert_eq!(fetcher.timeout.as_secs(), 5);
    }
}
