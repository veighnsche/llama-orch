//! Daemon lifecycle configuration types
//!
//! TEAM-315: Extracted from daemon-lifecycle

use serde::{Deserialize, Serialize};

/// Configuration for HTTP-based daemons
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_daemon_config_serialization() {
        let config = HttpDaemonConfig {
            daemon_name: "queen-rbee".to_string(),
            health_url: "http://localhost:7833/health".to_string(),
            shutdown_endpoint: Some("/v1/shutdown".to_string()),
            job_id: Some("job-123".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: HttpDaemonConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.daemon_name, deserialized.daemon_name);
        assert_eq!(config.health_url, deserialized.health_url);
        assert_eq!(config.shutdown_endpoint, deserialized.shutdown_endpoint);
    }
}
