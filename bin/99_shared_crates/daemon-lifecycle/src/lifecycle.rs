//! High-level daemon lifecycle operations
//!
//! TEAM-276: Combines spawn + health polling + shutdown for HTTP-based daemons
//! TEAM-316: Extended HttpDaemonConfig with lifecycle-specific fields
//!
//! Provides complete start/stop patterns that combine lower-level utilities.

use anyhow::Result;
use std::path::PathBuf;
use tokio::process::Child;

use crate::health::{poll_until_healthy, HealthPollConfig};
use crate::manager::DaemonManager;
use crate::shutdown::{graceful_shutdown, ShutdownConfig};

// TEAM-316: Use HttpDaemonConfig from daemon-contract as base
// Note: daemon-lifecycle extends this with binary_path, args, and health polling config
pub use daemon_contract::HttpDaemonConfig as HttpDaemonConfigBase;

/// Extended HTTP daemon config with lifecycle-specific fields
///
/// TEAM-316: Extends daemon-contract::HttpDaemonConfig with lifecycle management fields
#[derive(Clone)]
pub struct HttpDaemonConfig {
    /// Base configuration from contract
    pub base: HttpDaemonConfigBase,

    /// Path to daemon binary
    pub binary_path: PathBuf,

    /// Command-line arguments for daemon
    pub args: Vec<String>,

    /// Maximum health check attempts (default: 10)
    pub max_health_attempts: Option<usize>,

    /// Initial health check delay in ms (default: 200)
    pub health_initial_delay_ms: Option<u64>,
}

impl HttpDaemonConfig {
    /// Create a new HTTP daemon config
    pub fn new(
        daemon_name: impl Into<String>,
        binary_path: PathBuf,
        health_url: impl Into<String>,
    ) -> Self {
        let daemon_name = daemon_name.into();
        let health_url_str = health_url.into();
        
        let base = HttpDaemonConfigBase {
            daemon_name,
            health_url: health_url_str.clone(),
            shutdown_endpoint: Some(format!("{}/v1/shutdown", health_url_str)),
            job_id: None,
        };

        Self {
            base,
            binary_path,
            args: Vec::new(),
            max_health_attempts: None,
            health_initial_delay_ms: None,
        }
    }

    /// Set command-line arguments
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set job_id for narration
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.base.job_id = Some(job_id.into());
        self
    }

    /// Set custom shutdown endpoint
    pub fn with_shutdown_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.base.shutdown_endpoint = Some(endpoint.into());
        self
    }

    /// Set max health check attempts
    pub fn with_max_health_attempts(mut self, attempts: usize) -> Self {
        self.max_health_attempts = Some(attempts);
        self
    }

    /// Set initial health check delay
    pub fn with_health_initial_delay_ms(mut self, delay_ms: u64) -> Self {
        self.health_initial_delay_ms = Some(delay_ms);
        self
    }
}

/// Start an HTTP-based daemon (spawn + health polling)
///
/// TEAM-276: High-level function that combines spawn and health polling
///
/// Steps:
/// 1. Spawn the daemon process
/// 2. Poll health endpoint until ready
/// 3. Return the spawned Child process
///
/// # Arguments
/// * `config` - HTTP daemon configuration
///
/// # Returns
/// * `Ok(Child)` - Daemon started and healthy
/// * `Err` - Failed to start or become healthy
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{start_http_daemon, HttpDaemonConfig};
/// use std::path::PathBuf;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = HttpDaemonConfig::new(
///     "queen-rbee",
///     PathBuf::from("target/release/queen-rbee"),
///     "http://localhost:8500",
/// )
/// .with_job_id("job-123");
///
/// let child = start_http_daemon(config).await?;
/// // Daemon is now running and healthy
/// std::mem::forget(child); // Keep daemon alive
/// # Ok(())
/// # }
/// ```
pub async fn start_http_daemon(config: HttpDaemonConfig) -> Result<Child> {
    // Step 1: Spawn the daemon
    let manager = DaemonManager::new(config.binary_path.clone(), config.args.clone());
    let child = manager.spawn().await?;

    // Step 2: Poll until healthy
    let mut health_config =
        HealthPollConfig::new(&config.base.health_url).with_daemon_name(&config.base.daemon_name);

    if let Some(attempts) = config.max_health_attempts {
        health_config = health_config.with_max_attempts(attempts);
    }

    if let Some(job_id) = config.base.job_id.as_deref() {
        health_config = health_config.with_job_id(job_id);
    }

    poll_until_healthy(health_config).await?;

    Ok(child)
}

/// Stop an HTTP-based daemon gracefully
///
/// TEAM-276: High-level function for graceful daemon shutdown
///
/// Steps:
/// 1. Check if daemon is running
/// 2. Send shutdown request to HTTP endpoint
/// 3. Handle expected connection errors
///
/// # Arguments
/// * `config` - HTTP daemon configuration (uses health_url and shutdown_endpoint)
///
/// # Returns
/// * `Ok(())` - Daemon stopped successfully
/// * `Err` - Unexpected error during shutdown
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{stop_http_daemon, HttpDaemonConfig};
/// use std::path::PathBuf;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = HttpDaemonConfig::new(
///     "queen-rbee",
///     PathBuf::from("target/release/queen-rbee"),
///     "http://localhost:8500",
/// )
/// .with_job_id("job-123");
///
/// stop_http_daemon(config).await?;
/// # Ok(())
/// # }
/// ```
pub async fn stop_http_daemon(config: HttpDaemonConfig) -> Result<()> {
    let shutdown_endpoint = config.base.shutdown_endpoint
        .unwrap_or_else(|| format!("{}/v1/shutdown", config.base.health_url));
    
    let shutdown_config =
        ShutdownConfig::new(config.base.daemon_name, config.base.health_url, shutdown_endpoint);

    let shutdown_config = if let Some(job_id) = config.base.job_id {
        shutdown_config.with_job_id(job_id)
    } else {
        shutdown_config
    };

    graceful_shutdown(shutdown_config).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_daemon_config_builder() {
        let config = HttpDaemonConfig::new(
            "test-daemon",
            PathBuf::from("/usr/bin/test"),
            "http://localhost:8080",
        )
        .with_args(vec!["--config".to_string(), "test.toml".to_string()])
        .with_job_id("job-123")
        .with_max_health_attempts(5);

        assert_eq!(config.base.daemon_name, "test-daemon");
        assert_eq!(config.base.health_url, "http://localhost:8080");
        assert_eq!(config.base.shutdown_endpoint, Some("http://localhost:8080/v1/shutdown".to_string()));
        assert_eq!(config.args.len(), 2);
        assert_eq!(config.base.job_id, Some("job-123".to_string()));
        assert_eq!(config.max_health_attempts, Some(5));
    }

    #[test]
    fn test_http_daemon_config_custom_shutdown() {
        let config = HttpDaemonConfig::new(
            "test-daemon",
            PathBuf::from("/usr/bin/test"),
            "http://localhost:8080",
        )
        .with_shutdown_endpoint("http://localhost:8080/api/shutdown");

        assert_eq!(config.base.shutdown_endpoint, Some("http://localhost:8080/api/shutdown".to_string()));
    }
}
