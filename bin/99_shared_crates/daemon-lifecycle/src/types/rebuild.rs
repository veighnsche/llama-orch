//! Rebuild operation configuration types
//!
//! TEAM-329: Extracted from src/rebuild.rs

use serde::{Deserialize, Serialize};

/// Configuration for daemon rebuild
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebuildConfig {
    /// Name of the daemon binary (e.g., "queen-rbee", "rbee-hive")
    pub binary_name: String,

    /// Optional features to enable (e.g., "local-hive")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<String>>,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

impl RebuildConfig {
    /// Create a new rebuild config
    pub fn new(binary_name: impl Into<String>) -> Self {
        Self {
            binary_name: binary_name.into(),
            features: None,
            job_id: None,
        }
    }

    /// Set features to enable
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = Some(features);
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
    fn test_rebuild_config_builder() {
        let config = RebuildConfig::new("queen-rbee")
            .with_features(vec!["local-hive".to_string()])
            .with_job_id("job-123");

        assert_eq!(config.binary_name, "queen-rbee");
        assert_eq!(config.features, Some(vec!["local-hive".to_string()]));
        assert_eq!(config.job_id, Some("job-123".to_string()));
    }

    #[test]
    fn test_rebuild_config_serialization() {
        let config = RebuildConfig::new("rbee-hive")
            .with_job_id("job-456");

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RebuildConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.binary_name, deserialized.binary_name);
        assert_eq!(config.job_id, deserialized.job_id);
    }
}
