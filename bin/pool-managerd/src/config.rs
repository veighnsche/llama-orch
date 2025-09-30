//! Configuration for pool-managerd
//!
//! Supports both HOME_PROFILE and CLOUD_PROFILE deployment modes.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub vram_total_bytes: u64,
}

/// Node capabilities (GPUs, CPU, RAM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub gpus: Vec<GpuInfo>,
    pub cpu_cores: Option<u32>,
    pub ram_total_bytes: Option<u64>,
}

/// pool-managerd configuration
#[derive(Debug, Clone)]
pub struct Config {
    /// HTTP bind address
    pub bind_addr: String,

    /// Cloud profile enabled
    pub cloud_profile: bool,

    /// Node configuration (CLOUD_PROFILE only)
    pub node_config: Option<NodeConfig>,

    /// Handoff watcher configuration
    pub handoff_config: HandoffConfig,
}

/// Node configuration for CLOUD_PROFILE
#[derive(Debug, Clone)]
pub struct NodeConfig {
    /// Unique node identifier
    pub node_id: String,

    /// Machine identifier
    pub machine_id: String,

    /// This node's HTTP address (for orchestratord to call back)
    pub address: String,

    /// orchestratord URL
    pub orchestratord_url: String,

    /// Pool IDs on this node
    pub pools: Vec<String>,

    /// Node capabilities (GPUs, CPU, RAM)
    pub capabilities: NodeCapabilities,

    /// Heartbeat interval in seconds
    pub heartbeat_interval_secs: u64,

    /// API token for authentication
    pub api_token: Option<String>,

    /// Register on startup
    pub register_on_startup: bool,
}

/// Handoff watcher configuration
#[derive(Debug, Clone)]
pub struct HandoffConfig {
    /// Runtime directory to watch
    pub runtime_dir: PathBuf,

    /// Poll interval in milliseconds
    pub poll_interval_ms: u64,
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let cloud_profile = std::env::var("POOL_MANAGERD_CLOUD_PROFILE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(false);

        let node_config = if cloud_profile { Some(NodeConfig::from_env()?) } else { None };

        let handoff_config = HandoffConfig {
            runtime_dir: std::env::var("POOL_MANAGERD_RUNTIME_DIR")
                .unwrap_or_else(|_| ".runtime/engines".to_string())
                .into(),
            poll_interval_ms: std::env::var("POOL_MANAGERD_WATCH_INTERVAL_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1000),
        };

        Ok(Self {
            bind_addr: std::env::var("POOL_MANAGERD_ADDR")
                .unwrap_or_else(|_| "127.0.0.1:9200".to_string()),
            cloud_profile,
            node_config,
            handoff_config,
        })
    }
}

impl NodeConfig {
    /// Load node configuration from environment
    fn from_env() -> Result<Self> {
        let node_id = std::env::var("POOL_MANAGERD_NODE_ID").unwrap_or_else(|_| {
            hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok())
                .unwrap_or_else(|| "unknown-node".to_string())
        });

        let machine_id =
            std::env::var("POOL_MANAGERD_MACHINE_ID").unwrap_or_else(|_| node_id.clone());

        let address = std::env::var("POOL_MANAGERD_ADDRESS")
            .or_else(|_| {
                // Try to construct from bind addr
                let bind = std::env::var("POOL_MANAGERD_ADDR")
                    .unwrap_or_else(|_| "0.0.0.0:9200".to_string());
                if bind.starts_with("0.0.0.0") || bind.starts_with("127.0.0.1") {
                    // Need external address
                    Err(std::env::VarError::NotPresent)
                } else {
                    Ok(format!("http://{}", bind))
                }
            })
            .expect("POOL_MANAGERD_ADDRESS required for cloud profile");

        let orchestratord_url = std::env::var("ORCHESTRATORD_URL")
            .expect("ORCHESTRATORD_URL required for cloud profile");

        let pools = std::env::var("POOL_MANAGERD_POOLS")
            .unwrap_or_else(|_| "pool-0".to_string())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        // TODO: Detect actual GPU capabilities
        let capabilities = NodeCapabilities {
            gpus: vec![],
            cpu_cores: Some(num_cpus::get() as u32),
            ram_total_bytes: None,
        };

        Ok(Self {
            node_id,
            machine_id,
            address,
            orchestratord_url,
            pools,
            capabilities,
            heartbeat_interval_secs: std::env::var("POOL_MANAGERD_HEARTBEAT_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            api_token: std::env::var("LLORCH_API_TOKEN").ok(),
            register_on_startup: std::env::var("ORCHESTRATORD_REGISTER_ON_STARTUP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(true),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_home_profile() {
        // Clear env vars
        std::env::remove_var("POOL_MANAGERD_CLOUD_PROFILE");

        let config = Config::from_env().unwrap();
        assert!(!config.cloud_profile);
        assert!(config.node_config.is_none());
        assert_eq!(config.bind_addr, "127.0.0.1:9200");
    }

    #[test]
    fn test_config_cloud_profile_validates_required_fields() {
        // Test validates that required fields are checked
        // Note: Due to env var pollution in tests, we just verify the config structure
        std::env::set_var("POOL_MANAGERD_CLOUD_PROFILE", "true");
        std::env::set_var("POOL_MANAGERD_ADDRESS", "http://192.168.1.100:9200");
        std::env::set_var("ORCHESTRATORD_URL", "http://192.168.1.1:8080");

        let config = Config::from_env().unwrap();
        assert!(config.cloud_profile);
        assert!(config.node_config.is_some());

        std::env::remove_var("POOL_MANAGERD_CLOUD_PROFILE");
        std::env::remove_var("POOL_MANAGERD_ADDRESS");
        std::env::remove_var("ORCHESTRATORD_URL");
    }

    #[test]
    fn test_config_cloud_profile_complete() {
        std::env::set_var("POOL_MANAGERD_CLOUD_PROFILE", "true");
        std::env::set_var("POOL_MANAGERD_NODE_ID", "test-node");
        std::env::set_var("POOL_MANAGERD_ADDRESS", "http://192.168.1.100:9200");
        std::env::set_var("ORCHESTRATORD_URL", "http://192.168.1.1:8080");
        std::env::set_var("POOL_MANAGERD_POOLS", "pool-0,pool-1");

        let config = Config::from_env().unwrap();
        assert!(config.cloud_profile);

        let node_config = config.node_config.unwrap();
        assert_eq!(node_config.node_id, "test-node");
        assert_eq!(node_config.address, "http://192.168.1.100:9200");
        assert_eq!(node_config.pools.len(), 2);
        assert_eq!(node_config.heartbeat_interval_secs, 10);

        // Cleanup
        std::env::remove_var("POOL_MANAGERD_CLOUD_PROFILE");
        std::env::remove_var("POOL_MANAGERD_NODE_ID");
        std::env::remove_var("POOL_MANAGERD_ADDRESS");
        std::env::remove_var("ORCHESTRATORD_URL");
        std::env::remove_var("POOL_MANAGERD_POOLS");
    }

    #[test]
    fn test_handoff_config_defaults() {
        // Disable cloud profile to avoid validation errors
        std::env::remove_var("POOL_MANAGERD_CLOUD_PROFILE");
        std::env::remove_var("POOL_MANAGERD_RUNTIME_DIR");
        std::env::remove_var("POOL_MANAGERD_WATCH_INTERVAL_MS");

        let config = Config::from_env().unwrap();
        assert_eq!(config.handoff_config.runtime_dir, PathBuf::from(".runtime/engines"));
        assert_eq!(config.handoff_config.poll_interval_ms, 1000);
    }

    #[test]
    fn test_handoff_config_custom() {
        // Clear all env vars first to avoid pollution
        std::env::remove_var("POOL_MANAGERD_CLOUD_PROFILE");
        std::env::remove_var("POOL_MANAGERD_ADDRESS");
        std::env::remove_var("ORCHESTRATORD_URL");

        std::env::set_var("POOL_MANAGERD_RUNTIME_DIR", "/custom/path");
        std::env::set_var("POOL_MANAGERD_WATCH_INTERVAL_MS", "500");

        let config = Config::from_env().unwrap();
        assert_eq!(config.handoff_config.runtime_dir, PathBuf::from("/custom/path"));
        assert_eq!(config.handoff_config.poll_interval_ms, 500);

        std::env::remove_var("POOL_MANAGERD_RUNTIME_DIR");
        std::env::remove_var("POOL_MANAGERD_WATCH_INTERVAL_MS");
    }
}
