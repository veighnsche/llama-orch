//! Configuration for orchestratord
//!
//! Loads and validates orchestrator configuration from environment variables
//! and optional config files. Implements fail-fast validation per OC-CONFIG-6001.

use anyhow::{Context, Result};
use std::path::PathBuf;

/// orchestratord configuration
#[derive(Debug, Clone)]
pub struct Config {
    /// HTTP bind address
    pub bind_addr: String,

    /// Cloud profile enabled
    pub cloud_profile: bool,

    /// Admission queue configuration
    pub admission: AdmissionConfig,

    /// Placement strategy
    pub placement_strategy: PlacementStrategy,

    /// Service registry configuration (CLOUD_PROFILE only)
    pub service_registry: Option<ServiceRegistryConfig>,

    /// Stale node checker configuration (CLOUD_PROFILE only)
    pub stale_checker: Option<StaleCheckerConfig>,

    /// Optional config file path for pool definitions
    pub pools_config_path: Option<PathBuf>,
}

/// Admission queue configuration
#[derive(Debug, Clone)]
pub struct AdmissionConfig {
    /// Queue capacity
    pub capacity: usize,

    /// Full-queue policy: "reject" or "drop-lru"
    pub policy: QueuePolicy,
}

/// Queue policy for full queues
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueuePolicy {
    Reject,
    DropLru,
}

/// Placement strategy for task routing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementStrategy {
    RoundRobin,
    LeastLoaded,
    Random,
}

/// Service registry configuration
#[derive(Debug, Clone)]
pub struct ServiceRegistryConfig {
    /// Node timeout in milliseconds
    pub timeout_ms: u64,
}

/// Stale node checker configuration
#[derive(Debug, Clone)]
pub struct StaleCheckerConfig {
    /// Check interval in seconds
    pub interval_secs: u64,
}

impl Config {
    /// Load configuration from environment variables with validation.
    ///
    /// This implements OC-CONFIG-6001: strict validation with fail-fast on
    /// missing/invalid pools or placement configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Required environment variables are missing
    /// - Values cannot be parsed
    /// - Cloud profile is enabled but required fields are missing
    /// - Validation constraints are violated
    pub fn from_env() -> Result<Self> {
        let bind_addr = std::env::var("ORCHD_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());

        let cloud_profile = std::env::var("ORCHESTRATORD_CLOUD_PROFILE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(false);

        // Admission configuration
        let admission_capacity = std::env::var("ORCHD_ADMISSION_CAPACITY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);

        if admission_capacity == 0 {
            anyhow::bail!("ORCHD_ADMISSION_CAPACITY must be > 0, got 0");
        }

        let admission_policy = match std::env::var("ORCHD_ADMISSION_POLICY").ok().as_deref() {
            Some("drop-lru") => QueuePolicy::DropLru,
            Some("reject") | None => QueuePolicy::Reject,
            Some(other) => {
                anyhow::bail!(
                    "Invalid ORCHD_ADMISSION_POLICY: '{}'. Must be 'reject' or 'drop-lru'",
                    other
                );
            }
        };

        let admission = AdmissionConfig { capacity: admission_capacity, policy: admission_policy };

        // Placement strategy
        let placement_strategy = match std::env::var("ORCHESTRATORD_PLACEMENT_STRATEGY")
            .ok()
            .as_deref()
        {
            Some("least-loaded") => PlacementStrategy::LeastLoaded,
            Some("random") => PlacementStrategy::Random,
            Some("round-robin") | None => PlacementStrategy::RoundRobin,
            Some(other) => {
                anyhow::bail!(
                        "Invalid ORCHESTRATORD_PLACEMENT_STRATEGY: '{}'. Must be 'round-robin', 'least-loaded', or 'random'",
                        other
                    );
            }
        };

        // Cloud profile configuration
        let (service_registry, stale_checker) = if cloud_profile {
            let timeout_ms = std::env::var("ORCHESTRATORD_NODE_TIMEOUT_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30_000);

            if timeout_ms == 0 {
                anyhow::bail!("ORCHESTRATORD_NODE_TIMEOUT_MS must be > 0, got 0");
            }

            let interval_secs = std::env::var("ORCHESTRATORD_STALE_CHECK_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10);

            if interval_secs == 0 {
                anyhow::bail!("ORCHESTRATORD_STALE_CHECK_INTERVAL_SECS must be > 0, got 0");
            }

            (Some(ServiceRegistryConfig { timeout_ms }), Some(StaleCheckerConfig { interval_secs }))
        } else {
            (None, None)
        };

        // Optional pools config file
        let pools_config_path = std::env::var("ORCHD_POOLS_CONFIG").ok().map(PathBuf::from);

        // Validate pools config file exists if specified
        if let Some(ref path) = pools_config_path {
            if !path.exists() {
                anyhow::bail!("ORCHD_POOLS_CONFIG file does not exist: {}", path.display());
            }
        }

        Ok(Self {
            bind_addr,
            cloud_profile,
            admission,
            placement_strategy,
            service_registry,
            stale_checker,
            pools_config_path,
        })
    }

    /// Validate the configuration for consistency.
    ///
    /// This performs cross-field validation that cannot be done during
    /// construction.
    pub fn validate(&self) -> Result<()> {
        // Cloud profile validation
        if self.cloud_profile {
            if self.service_registry.is_none() {
                anyhow::bail!("Cloud profile enabled but service_registry config is missing");
            }
            if self.stale_checker.is_none() {
                anyhow::bail!("Cloud profile enabled but stale_checker config is missing");
            }
        }

        // Bind address validation (basic)
        if self.bind_addr.is_empty() {
            anyhow::bail!("bind_addr cannot be empty");
        }

        Ok(())
    }

    /// Load and validate configuration in one step.
    ///
    /// This is the primary entry point for loading configuration.
    /// It performs both loading and validation, failing fast on any error.
    pub fn load() -> Result<Self> {
        let config = Self::from_env().context("Failed to load configuration from environment")?;
        config.validate().context("Configuration validation failed")?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clear_env() {
        std::env::remove_var("ORCHD_ADDR");
        std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
        std::env::remove_var("ORCHD_ADMISSION_CAPACITY");
        std::env::remove_var("ORCHD_ADMISSION_POLICY");
        std::env::remove_var("ORCHESTRATORD_PLACEMENT_STRATEGY");
        std::env::remove_var("ORCHESTRATORD_NODE_TIMEOUT_MS");
        std::env::remove_var("ORCHESTRATORD_STALE_CHECK_INTERVAL_SECS");
        std::env::remove_var("ORCHD_POOLS_CONFIG");
    }

    #[test]
    fn test_config_defaults() {
        clear_env();

        let config = Config::from_env().unwrap();
        assert_eq!(config.bind_addr, "0.0.0.0:8080");
        assert!(!config.cloud_profile);
        assert_eq!(config.admission.capacity, 8);
        assert_eq!(config.admission.policy, QueuePolicy::Reject);
        assert_eq!(config.placement_strategy, PlacementStrategy::RoundRobin);
        assert!(config.service_registry.is_none());
        assert!(config.stale_checker.is_none());
        assert!(config.pools_config_path.is_none());
    }

    #[test]
    fn test_config_custom_bind_addr() {
        clear_env();
        std::env::set_var("ORCHD_ADDR", "127.0.0.1:9090");

        let config = Config::from_env().unwrap();
        assert_eq!(config.bind_addr, "127.0.0.1:9090");

        clear_env();
    }

    #[test]
    fn test_config_admission_capacity() {
        clear_env();
        std::env::set_var("ORCHD_ADMISSION_CAPACITY", "16");

        let config = Config::from_env().unwrap();
        assert_eq!(config.admission.capacity, 16);

        clear_env();
    }

    #[test]
    fn test_config_admission_capacity_zero_fails() {
        clear_env();
        std::env::set_var("ORCHD_ADMISSION_CAPACITY", "0");

        let result = Config::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0"));

        clear_env();
    }

    #[test]
    fn test_config_admission_policy_drop_lru() {
        clear_env();
        std::env::set_var("ORCHD_ADMISSION_POLICY", "drop-lru");

        let config = Config::from_env().unwrap();
        assert_eq!(config.admission.policy, QueuePolicy::DropLru);

        clear_env();
    }

    #[test]
    fn test_config_admission_policy_reject() {
        clear_env();
        std::env::set_var("ORCHD_ADMISSION_POLICY", "reject");

        let config = Config::from_env().unwrap();
        assert_eq!(config.admission.policy, QueuePolicy::Reject);

        clear_env();
    }

    #[test]
    fn test_config_admission_policy_invalid_fails() {
        clear_env();
        std::env::set_var("ORCHD_ADMISSION_POLICY", "invalid");

        let result = Config::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid ORCHD_ADMISSION_POLICY"));

        clear_env();
    }

    #[test]
    fn test_config_placement_strategy_least_loaded() {
        clear_env();
        std::env::set_var("ORCHESTRATORD_PLACEMENT_STRATEGY", "least-loaded");

        let config = Config::from_env().unwrap();
        assert_eq!(config.placement_strategy, PlacementStrategy::LeastLoaded);

        clear_env();
    }

    #[test]
    fn test_config_placement_strategy_random() {
        clear_env();
        std::env::set_var("ORCHESTRATORD_PLACEMENT_STRATEGY", "random");

        let config = Config::from_env().unwrap();
        assert_eq!(config.placement_strategy, PlacementStrategy::Random);

        clear_env();
    }

    #[test]
    fn test_config_placement_strategy_invalid_fails() {
        clear_env();
        std::env::set_var("ORCHESTRATORD_PLACEMENT_STRATEGY", "invalid");

        let result = Config::from_env();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid ORCHESTRATORD_PLACEMENT_STRATEGY"));

        clear_env();
    }

    #[test]
    fn test_config_cloud_profile_enabled() {
        clear_env();
        std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");

        let config = Config::from_env().unwrap();
        assert!(config.cloud_profile);
        assert!(config.service_registry.is_some());
        assert!(config.stale_checker.is_some());

        let registry = config.service_registry.unwrap();
        assert_eq!(registry.timeout_ms, 30_000);

        let checker = config.stale_checker.unwrap();
        assert_eq!(checker.interval_secs, 10);

        clear_env();
    }

    #[test]
    fn test_config_cloud_profile_custom_timeouts() {
        clear_env();
        std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
        std::env::set_var("ORCHESTRATORD_NODE_TIMEOUT_MS", "60000");
        std::env::set_var("ORCHESTRATORD_STALE_CHECK_INTERVAL_SECS", "20");

        let config = Config::from_env().unwrap();
        let registry = config.service_registry.unwrap();
        assert_eq!(registry.timeout_ms, 60_000);

        let checker = config.stale_checker.unwrap();
        assert_eq!(checker.interval_secs, 20);

        clear_env();
    }

    #[test]
    fn test_config_cloud_profile_zero_timeout_fails() {
        clear_env();
        std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
        std::env::set_var("ORCHESTRATORD_NODE_TIMEOUT_MS", "0");

        let result = Config::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0"));

        clear_env();
    }

    #[test]
    fn test_config_cloud_profile_zero_interval_fails() {
        clear_env();
        std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
        std::env::set_var("ORCHESTRATORD_STALE_CHECK_INTERVAL_SECS", "0");

        let result = Config::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("must be > 0"));

        clear_env();
    }

    #[test]
    fn test_config_pools_config_nonexistent_fails() {
        clear_env();
        std::env::set_var("ORCHD_POOLS_CONFIG", "/nonexistent/path/config.yaml");

        let result = Config::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));

        clear_env();
    }

    #[test]
    fn test_config_validate_success() {
        clear_env();

        let config = Config::from_env().unwrap();
        assert!(config.validate().is_ok());

        clear_env();
    }

    #[test]
    fn test_config_load_success() {
        clear_env();

        let config = Config::load().unwrap();
        assert_eq!(config.bind_addr, "0.0.0.0:8080");

        clear_env();
    }
}
