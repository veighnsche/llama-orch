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
    
    // TEAM-DX-002: Additional fetcher tests
    
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
    
    #[test]
    fn test_fetcher_default() {
        let fetcher = Fetcher::default();
        assert_eq!(fetcher.timeout.as_secs(), 2);
    }
    
    #[test]
    fn test_fetcher_timeout_values() {
        let fetcher_1s = Fetcher::with_timeout(Duration::from_secs(1));
        let fetcher_10s = Fetcher::with_timeout(Duration::from_secs(10));
        
        assert_eq!(fetcher_1s.timeout.as_secs(), 1);
        assert_eq!(fetcher_10s.timeout.as_secs(), 10);
    }
    
    #[test]
    fn test_fetcher_timeout_millis() {
        let fetcher = Fetcher::with_timeout(Duration::from_millis(500));
        assert_eq!(fetcher.timeout.as_millis(), 500);
    }
    
    // Integration tests require network access
    #[tokio::test]
    #[ignore]
    async fn test_fetch_page_success() {
        let fetcher = Fetcher::new();
        let result = fetcher.fetch_page("http://localhost:3000").await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    #[ignore]
    async fn test_fetch_stylesheet_success() {
        let fetcher = Fetcher::new();
        let result = fetcher.fetch_stylesheet("http://localhost:3000/style.css").await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_fetch_invalid_url() {
        let fetcher = Fetcher::new();
        let result = fetcher.fetch_page("not-a-valid-url").await;
        assert!(result.is_err());
    }
}
