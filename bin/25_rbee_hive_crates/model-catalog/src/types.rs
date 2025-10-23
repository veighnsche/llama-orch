// TEAM-267: Model catalog types
use rbee_hive_artifact_catalog::{Artifact, ArtifactStatus};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Model status (alias for ArtifactStatus)
pub type ModelStatus = ArtifactStatus;

/// Model entry in the catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Unique model ID (e.g., "meta-llama/Llama-2-7b")
    id: String,

    /// Human-readable name
    name: String,

    /// Filesystem path to model files
    path: PathBuf,

    /// Size in bytes
    size: u64,

    /// Current status
    status: ArtifactStatus,

    /// When the model was added
    #[serde(default = "chrono::Utc::now")]
    added_at: chrono::DateTime<chrono::Utc>,
}

impl ModelEntry {
    /// Create a new model entry
    pub fn new(id: String, name: String, path: PathBuf, size: u64) -> Self {
        Self {
            id,
            name,
            path,
            size,
            status: ArtifactStatus::Available,
            added_at: chrono::Utc::now(),
        }
    }

    /// Get the model name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get when the model was added
    pub fn added_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.added_at
    }
}

// Implement Artifact trait
impl Artifact for ModelEntry {
    fn id(&self) -> &str {
        &self.id
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn status(&self) -> &ArtifactStatus {
        &self.status
    }

    fn set_status(&mut self, status: ArtifactStatus) {
        self.status = status;
    }

    fn name(&self) -> &str {
        &self.name
    }
}
