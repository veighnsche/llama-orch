//! Daemon lifecycle configuration types
//!
//! TEAM-315: Extracted from daemon-lifecycle
//! TEAM-316: Added full HttpDaemonConfig with lifecycle fields (moved from daemon-lifecycle)

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for HTTP-based daemons
///
/// TEAM-316: Complete config with both contract fields and lifecycle management fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpDaemonConfig {
    /// Daemon name for narration (e.g., "queen-rbee", "rbee-hive")
    pub daemon_name: String,

    /// Health check URL (e.g., "http://localhost:7833/health")
    pub health_url: String,

    /// Optional shutdown endpoint (e.g., "/v1/shutdown")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shutdown_endpoint: Option<String>,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,

    // TEAM-316: Lifecycle management fields (moved from daemon-lifecycle)
    /// Path to daemon binary
    pub binary_path: PathBuf,

    /// Command-line arguments for daemon
    #[serde(default)]
    pub args: Vec<String>,

    /// Maximum health check attempts (default: 10)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_health_attempts: Option<usize>,

    /// Initial health check delay in ms (default: 200)
    #[serde(skip_serializing_if = "Option::is_none")]
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

        Self {
            daemon_name,
            health_url: health_url_str.clone(),
            shutdown_endpoint: Some(format!("{}/v1/shutdown", health_url_str)),
            job_id: None,
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
        self.job_id = Some(job_id.into());
        self
    }

    /// Set custom shutdown endpoint
    pub fn with_shutdown_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.shutdown_endpoint = Some(endpoint.into());
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

        assert_eq!(config.daemon_name, "test-daemon");
        assert_eq!(config.health_url, "http://localhost:8080");
        assert_eq!(config.shutdown_endpoint, Some("http://localhost:8080/v1/shutdown".to_string()));
        assert_eq!(config.args.len(), 2);
        assert_eq!(config.job_id, Some("job-123".to_string()));
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

        assert_eq!(config.shutdown_endpoint, Some("http://localhost:8080/api/shutdown".to_string()));
    }

    #[test]
    fn test_http_daemon_config_serialization() {
        let config = HttpDaemonConfig::new(
            "queen-rbee",
            PathBuf::from("/usr/bin/queen-rbee"),
            "http://localhost:7833/health",
        )
        .with_job_id("job-123");

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: HttpDaemonConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.daemon_name, deserialized.daemon_name);
        assert_eq!(config.health_url, deserialized.health_url);
        assert_eq!(config.shutdown_endpoint, deserialized.shutdown_endpoint);
    }
}
