//! Daemon shutdown configuration types
//!
//! TEAM-315: Extracted from daemon-lifecycle
//! TEAM-329: Moved from daemon-contract to daemon-lifecycle/types (inline)

use serde::{Deserialize, Serialize};

/// Configuration for graceful shutdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownConfig {
    /// Daemon name for narration
    pub daemon_name: String,

    /// Process ID to shut down
    pub pid: u32,

    /// Timeout for graceful shutdown (seconds)
    #[serde(default = "default_graceful_timeout")]
    pub graceful_timeout_secs: u64,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

fn default_graceful_timeout() -> u64 {
    5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shutdown_config_serialization() {
        let config = ShutdownConfig {
            daemon_name: "rbee-hive".to_string(),
            pid: 12345,
            graceful_timeout_secs: 10,
            job_id: Some("job-123".to_string()),
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ShutdownConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.daemon_name, deserialized.daemon_name);
        assert_eq!(config.pid, deserialized.pid);
        assert_eq!(config.graceful_timeout_secs, deserialized.graceful_timeout_secs);
    }

    #[test]
    fn test_shutdown_config_default_timeout() {
        let config = ShutdownConfig {
            daemon_name: "rbee-hive".to_string(),
            pid: 12345,
            graceful_timeout_secs: default_graceful_timeout(),
            job_id: None,
        };
        assert_eq!(config.graceful_timeout_secs, 5);
    }
}
