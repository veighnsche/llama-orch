//! Daemon installation types
//!
//! TEAM-315: Extracted from daemon-lifecycle

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Configuration for daemon installation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallConfig {
    /// Name of the daemon binary (e.g., "rbee-hive", "vllm-worker")
    pub binary_name: String,

    /// Optional path to binary (auto-detects if None)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<String>,

    /// Optional target installation path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_path: Option<String>,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

/// Result of daemon installation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallResult {
    /// Path to the installed binary
    pub binary_path: String,

    /// Installation timestamp
    #[serde(with = "systemtime_serde")]
    pub install_time: SystemTime,
}

/// Configuration for daemon uninstallation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UninstallConfig {
    /// Name of the daemon (e.g., "queen-rbee", "rbee-hive")
    pub daemon_name: String,

    /// Installation path
    pub install_path: String,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

// Helper module for SystemTime serialization
mod systemtime_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap();
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + std::time::Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_install_config_serialization() {
        let config = InstallConfig {
            binary_name: "rbee-hive".to_string(),
            binary_path: Some("/usr/local/bin/rbee-hive".to_string()),
            target_path: None,
            job_id: Some("job-123".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: InstallConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.binary_name, deserialized.binary_name);
        assert_eq!(config.binary_path, deserialized.binary_path);
    }

    #[test]
    fn test_install_result_serialization() {
        let result = InstallResult {
            binary_path: "/usr/local/bin/rbee-hive".to_string(),
            install_time: SystemTime::now(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: InstallResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.binary_path, deserialized.binary_path);
    }

    #[test]
    fn test_uninstall_config_serialization() {
        let config = UninstallConfig {
            daemon_name: "rbee-hive".to_string(),
            install_path: "/usr/local/bin/rbee-hive".to_string(),
            job_id: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: UninstallConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.daemon_name, deserialized.daemon_name);
        assert_eq!(config.install_path, deserialized.install_path);
    }
}
