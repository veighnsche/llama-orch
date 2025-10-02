//! node-registration-client — Client for worker node registration
//!
//! Handles worker registration with orchestrator, heartbeats, and capability reporting.
//!
//! # ⚠️ SECURITY: Worker Token Management
//!
//! For worker registration tokens, use `secrets-management`:
//! ```rust,ignore
//! use secrets_management::Secret;
//! let worker_token = Secret::load_from_file("/etc/llorch/secrets/worker-token")?;
//! // Use worker_token.expose() for Authorization header
//! ```
//! See: `bin/shared-crates/secrets-management/README.md`odic heartbeats.

use anyhow::{Context, Result};
use pool_registry_types::NodeCapabilities;
use service_registry::{HeartbeatPoolStatus, HeartbeatRequest, RegisterRequest};
use std::time::Duration;
use tracing::{info, warn};

pub mod client;

pub use client::RegistrationClient;

/// Configuration for node registration
#[derive(Debug, Clone)]
pub struct NodeRegistrationConfig {
    pub node_id: String,
    pub machine_id: String,
    pub address: String,
    pub orchestratord_url: String,
    pub pools: Vec<String>,
    pub capabilities: NodeCapabilities,
    pub heartbeat_interval_secs: u64,
    pub api_token: Option<String>,
}

/// Node registration service
pub struct NodeRegistration {
    config: NodeRegistrationConfig,
    client: RegistrationClient,
}

impl NodeRegistration {
    /// Create a new node registration service
    pub fn new(config: NodeRegistrationConfig) -> Self {
        let client =
            RegistrationClient::new(config.orchestratord_url.clone(), config.api_token.clone());

        Self { config, client }
    }

    /// Register with orchestratord
    pub async fn register(&self) -> Result<()> {
        info!(
            node_id = %self.config.node_id,
            orchestratord_url = %self.config.orchestratord_url,
            "Registering with control plane"
        );

        let request = RegisterRequest {
            node_id: self.config.node_id.clone(),
            machine_id: self.config.machine_id.clone(),
            address: self.config.address.clone(),
            pools: self.config.pools.clone(),
            capabilities: self.config.capabilities.clone(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
        };

        self.client.register(request).await.context("Failed to register with orchestratord")?;

        info!(node_id = %self.config.node_id, "Successfully registered");
        Ok(())
    }

    /// Deregister from orchestratord
    pub async fn deregister(&self) -> Result<()> {
        info!(
            node_id = %self.config.node_id,
            "Deregistering from control plane"
        );

        self.client.deregister(&self.config.node_id).await.context("Failed to deregister")?;

        info!(node_id = %self.config.node_id, "Successfully deregistered");
        Ok(())
    }

    /// Spawn heartbeat task
    pub fn spawn_heartbeat<F>(self, get_pool_status: F) -> tokio::task::JoinHandle<()>
    where
        F: Fn() -> Vec<HeartbeatPoolStatus> + Send + 'static,
    {
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(self.config.heartbeat_interval_secs));

            info!(
                node_id = %self.config.node_id,
                interval_secs = self.config.heartbeat_interval_secs,
                "Starting heartbeat task"
            );

            loop {
                ticker.tick().await;

                let pools = get_pool_status();
                let request =
                    HeartbeatRequest { timestamp: chrono::Utc::now().to_rfc3339(), pools };

                match self.client.heartbeat(&self.config.node_id, request).await {
                    Ok(_) => {
                        tracing::debug!(node_id = %self.config.node_id, "Heartbeat sent");
                    }
                    Err(e) => {
                        warn!(
                            node_id = %self.config.node_id,
                            error = %e,
                            "Failed to send heartbeat"
                        );
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> NodeRegistrationConfig {
        NodeRegistrationConfig {
            node_id: "node-1".to_string(),
            machine_id: "machine-1".to_string(),
            address: "http://localhost:9200".to_string(),
            orchestratord_url: "http://localhost:8080".to_string(),
            pools: vec!["pool-0".to_string()],
            capabilities: NodeCapabilities {
                gpus: vec![],
                cpu_cores: Some(8),
                ram_total_bytes: Some(32_000_000_000),
            },
            heartbeat_interval_secs: 10,
            api_token: None,
        }
    }

    #[tokio::test]
    async fn test_create_registration() {
        let config = test_config();
        let _registration = NodeRegistration::new(config);
    }
}
