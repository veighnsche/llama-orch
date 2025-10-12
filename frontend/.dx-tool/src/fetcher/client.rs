// Created by: TEAM-DX-001
// TEAM-DX-003: Added headless browser support for SPAs
// HTTP client for fetching pages and stylesheets

use crate::error::{DxError, Result};
use reqwest::Client;
use std::time::Duration;
use headless_chrome::{Browser, LaunchOptions};
use std::sync::Arc;

/// HTTP client wrapper with timeout and headless browser support
pub struct Fetcher {
    client: Client,
    timeout: Duration,
    use_browser: bool,
    browser_wait_ms: u64,
}

// TEAM-DX-003: Tuned for 12th Gen Intel i5-1240P, 62GB RAM, NVMe SSD
const DEFAULT_BROWSER_WAIT_MS: u64 = 3000;  // 3 seconds for SPA render
const MAX_BROWSER_WAIT_MS: u64 = 10000;     // 10 seconds maximum

impl Fetcher {
    pub fn new() -> Self {
        Self::with_timeout(Duration::from_secs(30))
    }
    
    pub fn with_timeout(timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");
        
        Self { 
            client, 
            timeout,
            use_browser: true, // TEAM-DX-003: Enable by default for SPA support
            browser_wait_ms: DEFAULT_BROWSER_WAIT_MS,
        }
    }
    
    pub fn without_browser(mut self) -> Self {
        self.use_browser = false;
        self
    }
    
    pub fn with_browser_wait(mut self, wait_ms: u64) -> Self {
        self.browser_wait_ms = wait_ms.min(MAX_BROWSER_WAIT_MS);
        self
    }
    
    /// Fetch HTML page (with headless browser for SPAs)
    pub async fn fetch_page(&self, url: &str) -> Result<String> {
        if self.use_browser {
            // TEAM-DX-003: Use headless Chrome for SPA support
            self.fetch_page_with_browser(url)
        } else {
            self.fetch_page_simple(url).await
        }
    }
    
    /// Fetch HTML with headless browser (for SPAs like Histoire/Storybook)
    fn fetch_page_with_browser(&self, url: &str) -> Result<String> {
        let options = LaunchOptions::default_builder()
            .headless(true)
            .build()
            .map_err(|e| DxError::Network(e.to_string()))?;
        
        let browser = Browser::new(options)
            .map_err(|e| DxError::Network(e.to_string()))?;
        
        let tab = browser.new_tab()
            .map_err(|e| DxError::Network(e.to_string()))?;
        
        // Navigate and wait for network idle
        tab.navigate_to(url)
            .map_err(|e| DxError::Network(e.to_string()))?;
        
        tab.wait_for_element("body")
            .map_err(|e| DxError::Network(e.to_string()))?;
        
        // Wait for JavaScript to execute and render (SPAs need time)
        // Poll for content with configurable timeout
        let poll_interval_ms = 500;
        let max_polls = (self.browser_wait_ms / poll_interval_ms) as usize;
        
        for _ in 0..max_polls {
            std::thread::sleep(Duration::from_millis(poll_interval_ms));
            
            if let Ok(result) = tab.evaluate("document.querySelector('#app')?.innerHTML?.length > 0", false) {
                if let Some(value) = result.value {
                    if let Some(has_content) = value.as_bool() {
                        if has_content {
                            // Give it a bit more time for full render
                            std::thread::sleep(Duration::from_millis(1000));
                            break;
                        }
                    }
                }
            }
        }
        
        // Check for iframes (Histoire/Storybook pattern)
        let has_iframe = tab.evaluate("document.querySelector('iframe') !== null", false)
            .ok()
            .and_then(|v| v.value)
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        if has_iframe {
            // Wait for iframe to load
            std::thread::sleep(Duration::from_millis(1500));
            
            // Try to get iframe src and navigate to it directly
            if let Ok(iframe_src) = tab.evaluate(
                "document.querySelector('iframe')?.src || ''",
                false
            ) {
                if let Some(src_value) = iframe_src.value {
                    if let Some(src_str) = src_value.as_str() {
                        if !src_str.is_empty() && src_str.starts_with("http") {
                            // Navigate to iframe URL directly
                            drop(tab);
                            let new_tab = browser.new_tab()
                                .map_err(|e| DxError::Network(e.to_string()))?;
                            
                            new_tab.navigate_to(src_str)
                                .map_err(|e| DxError::Network(e.to_string()))?;
                            
                            new_tab.wait_for_element("body")
                                .map_err(|e| DxError::Network(e.to_string()))?;
                            
                            // Wait for iframe content to render
                            std::thread::sleep(Duration::from_millis(self.browser_wait_ms));
                            
                            let html = new_tab.get_content()
                                .map_err(|e| DxError::Network(e.to_string()))?;
                            
                            return Ok(html);
                        }
                    }
                }
            }
        }
        
        // Get the rendered HTML
        let html = tab.get_content()
            .map_err(|e| DxError::Network(e.to_string()))?;
        
        Ok(html)
    }
    
    /// Fetch HTML page (simple HTTP request, no JS execution)
    async fn fetch_page_simple(&self, url: &str) -> Result<String> {
        let response = self.client
            .get(url)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    DxError::Timeout {
                        url: url.to_string(),
                        timeout_secs: self.timeout.as_secs(),
                    }
                } else if e.is_connect() {
                    DxError::Network(format!("Connection failed: {}", e))
                } else {
                    DxError::Network(e.to_string())
                }
            })?;
        
        if !response.status().is_success() {
            return Err(DxError::Network(format!(
                "HTTP {} for {}",
                response.status(),
                url
            )));
        }
        
        response
            .text()
            .await
            .map_err(|e| DxError::Network(e.to_string()))
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
