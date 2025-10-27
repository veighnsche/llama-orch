//! Daemon installation types
//!
//! TEAM-315: Extracted from daemon-lifecycle
//! TEAM-329: Moved from daemon-contract to daemon-lifecycle/types (inline)

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

impl InstallConfig {
    /// Create a new install config
    pub fn new(binary_name: impl Into<String>) -> Self {
        Self {
            binary_name: binary_name.into(),
            binary_path: None,
            target_path: None,
            job_id: None,
        }
    }

    /// Set explicit binary path
    pub fn with_binary_path(mut self, path: impl Into<String>) -> Self {
        self.binary_path = Some(path.into());
        self
    }

    /// Set target installation path
    pub fn with_target_path(mut self, path: impl Into<String>) -> Self {
        self.target_path = Some(path.into());
        self
    }

    /// Set job_id for narration
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }
}

/// Result of daemon installation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallResult {
    /// Path to the installed binary
    pub binary_path: String,

    /// Installation timestamp
    #[serde(
        serialize_with = "crate::utils::serde::serialize_systemtime",
        deserialize_with = "crate::utils::serde::deserialize_systemtime"
    )]
    pub install_time: SystemTime,
    
    /// Whether the binary was found in target directory (vs provided path)
    /// TEAM-316: Added for daemon-lifecycle compatibility
    pub found_in_target: bool,
}

// TEAM-329: UninstallConfig moved to types/uninstall.rs (PARITY)
// TEAM-329: systemtime_serde moved to utils/serde.rs (serialization helpers are utilities)

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
            found_in_target: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: InstallResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.binary_path, deserialized.binary_path);
        assert_eq!(result.found_in_target, deserialized.found_in_target);
    }

    // TEAM-329: test_uninstall_config_serialization moved to types/uninstall.rs
}
