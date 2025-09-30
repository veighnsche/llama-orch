//! HTTP client for pool-managerd daemon
//!
//! Replaces embedded Registry with HTTP calls to daemon on port 9200

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct PoolManagerClient {
    base_url: String,
    client: reqwest::Client,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStatus {
    pub pool_id: String,
    pub live: bool,
    pub ready: bool,
    pub active_leases: i32,
    pub slots_total: i32,
    pub slots_free: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

impl Default for PoolStatus {
    fn default() -> Self {
        Self {
            pool_id: String::new(),
            live: false,
            ready: false,
            active_leases: 0,
            slots_total: 0,
            slots_free: 0,
        }
    }
}

impl PoolManagerClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .expect("failed to build reqwest client"),
        }
    }

    pub fn from_env() -> Self {
        let url = std::env::var("POOL_MANAGERD_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:9200".to_string());
        Self::new(url)
    }

    pub async fn get_pool_status(&self, pool_id: &str) -> Result<PoolStatus> {
        let url = format!("{}/pools/{}/status", self.base_url, pool_id);
        let resp = self.client.get(&url).send().await?;
        
        if !resp.status().is_success() {
            anyhow::bail!("pool status request failed: {}", resp.status());
        }
        
        let status = resp.json().await?;
        Ok(status)
    }

    pub async fn daemon_health(&self) -> Result<HealthResponse> {
        let url = format!("{}/health", self.base_url);
        let resp = self.client.get(&url).send().await?;
        
        if !resp.status().is_success() {
            anyhow::bail!("health check failed: {}", resp.status());
        }
        
        let health = resp.json().await?;
        Ok(health)
    }

    /// Check if daemon is reachable
    pub async fn is_available(&self) -> bool {
        self.daemon_health().await.is_ok()
    }
}
