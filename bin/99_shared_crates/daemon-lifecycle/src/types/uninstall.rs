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
