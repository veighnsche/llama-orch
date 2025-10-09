//! HTTP client for rbee-hive pool manager
//!
//! Per test-001-mvp.md Phase 2: Pool Preflight
//! Communicates with rbee-hive daemon via HTTP
//!
//! Created by: TEAM-027

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// HTTP client for pool manager communication
pub struct PoolClient {
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

/// Health check response
///
/// Per test-001-mvp.md lines 49-54
#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub api_version: String,
}

/// Spawn worker request
///
/// Per test-001-mvp.md lines 136-143
#[derive(Debug, Serialize)]
pub struct SpawnWorkerRequest {
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub model_path: String,
}

/// Spawn worker response
///
/// Per test-001-mvp.md lines 162-166
#[derive(Debug, Deserialize)]
pub struct SpawnWorkerResponse {
    pub worker_id: String,
    pub url: String,
    pub state: String,
}

impl PoolClient {
    /// Create new pool client
    ///
    /// # Arguments
    /// * `base_url` - Pool manager base URL (e.g., "http://mac.home.arpa:8080")
    /// * `api_key` - API key for authentication
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            base_url,
            api_key,
            client: reqwest::Client::new(),
        }
    }

    /// Health check
    ///
    /// Per test-001-mvp.md Phase 2.1: Version Check
    ///
    /// # Returns
    /// Health response with status, version, and API version
    pub async fn health_check(&self) -> Result<HealthResponse> {
        let response = self
            .client
            .get(&format!("{}/v1/health", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Health check failed: HTTP {}",
                response.status()
            );
        }

        Ok(response.json().await?)
    }

    /// Spawn worker
    ///
    /// Per test-001-mvp.md Phase 5: Worker Startup
    ///
    /// # Arguments
    /// * `request` - Spawn worker request
    ///
    /// # Returns
    /// Spawn worker response with worker ID, URL, and state
    pub async fn spawn_worker(&self, request: SpawnWorkerRequest) -> Result<SpawnWorkerResponse> {
        let response = self
            .client
            .post(&format!("{}/v1/workers/spawn", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Failed to spawn worker: HTTP {} - {}",
                status,
                body
            );
        }

        Ok(response.json().await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_client_creation() {
        let client = PoolClient::new(
            "http://localhost:8080".to_string(),
            "test-key".to_string(),
        );
        assert_eq!(client.base_url, "http://localhost:8080");
        assert_eq!(client.api_key, "test-key");
    }
}
