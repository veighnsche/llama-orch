//! Rebuild operation configuration types
//!
//! TEAM-329: Extracted from src/rebuild.rs

/// Configuration for daemon rebuild
#[derive(Debug, Clone)]
pub struct RebuildConfig {
    /// Name of the daemon binary (e.g., "queen-rbee", "rbee-hive")
    pub binary_name: String,

    /// Optional features to enable (e.g., "local-hive")
    pub features: Option<Vec<String>>,

    /// Optional job ID for narration routing
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
