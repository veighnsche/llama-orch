//! Uninstall operation configuration types
//!
//! TEAM-329: Extracted from types/install.rs (PARITY - each operation gets its own types file)

use serde::{Deserialize, Serialize};

/// Configuration for daemon uninstallation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UninstallConfig {
    /// Name of the daemon (e.g., "queen-rbee", "rbee-hive")
    pub daemon_name: String,

    /// Installation path
    pub install_path: String,

    /// Optional health check URL to verify daemon is not running
    /// TEAM-316: Added for daemon-lifecycle compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_url: Option<String>,

    /// Optional timeout for health check (default: 2 seconds)
    /// TEAM-316: Added for daemon-lifecycle compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_timeout_secs: Option<u64>,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

impl UninstallConfig {
    /// Create a new uninstall config
    pub fn new(daemon_name: impl Into<String>, install_path: impl Into<String>) -> Self {
        Self {
            daemon_name: daemon_name.into(),
            install_path: install_path.into(),
            health_url: None,
            health_timeout_secs: None,
            job_id: None,
        }
    }

    /// Set health check URL
    pub fn with_health_url(mut self, url: impl Into<String>) -> Self {
        self.health_url = Some(url.into());
        self
    }

    /// Set health check timeout
    pub fn with_health_timeout_secs(mut self, secs: u64) -> Self {
        self.health_timeout_secs = Some(secs);
        self
    }

    /// Set job_id for narration
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uninstall_config_serialization() {
        let config = UninstallConfig {
            daemon_name: "rbee-hive".to_string(),
            install_path: "/usr/local/bin/rbee-hive".to_string(),
            health_url: Some("http://localhost:7835/health".to_string()),
            health_timeout_secs: Some(2),
            job_id: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: UninstallConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.daemon_name, deserialized.daemon_name);
        assert_eq!(config.install_path, deserialized.install_path);
        assert_eq!(config.health_url, deserialized.health_url);
        assert_eq!(config.health_timeout_secs, deserialized.health_timeout_secs);
    }
}
