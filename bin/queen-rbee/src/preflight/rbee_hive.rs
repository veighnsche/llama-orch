// TEAM-110: Audited 2025-10-18 - ✅ CLEAN - Standard HTTP client for preflight checks with proper timeouts
//
//! rbee-hive Preflight Validation
//!
//! Created by: TEAM-079
//!
//! Validates rbee-hive readiness before spawning workers.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Backend {
    pub name: String,
    pub available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub ram_total_gb: u32,
    pub ram_available_gb: u32,
    pub disk_total_gb: u32,
    pub disk_available_gb: u32,
}

#[derive(Debug, Clone)]
pub struct RbeeHivePreflight {
    pub base_url: String,
    client: reqwest::Client,
}

impl RbeeHivePreflight {
    /// Create a new rbee-hive preflight checker
    pub fn new(base_url: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");
        
        Self { base_url, client }
    }

    /// Check rbee-hive health endpoint
    pub async fn check_health(&self) -> Result<HealthResponse> {
        let url = format!("{}/health", self.base_url);
        tracing::info!("Checking rbee-hive health: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to connect to rbee-hive")?;

        if !response.status().is_success() {
            anyhow::bail!("Health check failed: HTTP {}", response.status());
        }

        let health: HealthResponse = response.json().await
            .context("Failed to parse health response")?;

        Ok(health)
    }

    /// Validate version compatibility
    pub async fn check_version_compatibility(&self, required: &str) -> Result<bool> {
        let health = self.check_health().await?;
        
        // Simple version comparison (in production, use semver crate)
        let compatible = health.version >= required.to_string();
        
        tracing::info!("Version check: {} >= {} = {}", health.version, required, compatible);
        Ok(compatible)
    }

    /// Query available backends
    pub async fn query_backends(&self) -> Result<Vec<Backend>> {
        let url = format!("{}/v1/backends", self.base_url);
        tracing::info!("Querying backends: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to query backends")?;

        let backends: Vec<Backend> = response.json().await
            .context("Failed to parse backends response")?;

        Ok(backends)
    }

    /// Query available resources
    pub async fn query_resources(&self) -> Result<ResourceInfo> {
        let url = format!("{}/v1/resources", self.base_url);
        tracing::info!("Querying resources: {}", url);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to query resources")?;

        let resources: ResourceInfo = response.json().await
            .context("Failed to parse resources response")?;

        Ok(resources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preflight_creation() {
        let preflight = RbeeHivePreflight::new("http://localhost:8081".to_string());
        assert_eq!(preflight.base_url, "http://localhost:8081");
    }
}
