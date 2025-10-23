// TEAM-273: Shared artifact types
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Artifact status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ArtifactStatus {
    /// Artifact is available and ready to use
    Available,

    /// Artifact is currently being downloaded/provisioned
    Downloading,

    /// Artifact download/provisioning failed
    Failed {
        /// Error message
        error: String,
    },
}

/// Core artifact trait
///
/// Implemented by ModelEntry, WorkerBinary, etc.
pub trait Artifact: Clone + Serialize + for<'de> Deserialize<'de> {
    /// Unique identifier for this artifact
    fn id(&self) -> &str;

    /// Filesystem path to the artifact
    fn path(&self) -> &Path;

    /// Size in bytes
    fn size(&self) -> u64;

    /// Current status
    fn status(&self) -> &ArtifactStatus;

    /// Set status (mutable)
    fn set_status(&mut self, status: ArtifactStatus);

    /// Human-readable name
    fn name(&self) -> &str {
        self.id()
    }
}

/// Metadata for filesystem-based catalogs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata<T> {
    /// The artifact itself
    pub artifact: T,

    /// When it was added to catalog
    pub added_at: chrono::DateTime<chrono::Utc>,

    /// Last accessed time (optional)
    pub last_accessed: Option<chrono::DateTime<chrono::Utc>>,
}

impl<T> ArtifactMetadata<T> {
    /// Create new metadata
    pub fn new(artifact: T) -> Self {
        Self { artifact, added_at: chrono::Utc::now(), last_accessed: None }
    }

    /// Update last accessed time
    pub fn touch(&mut self) {
        self.last_accessed = Some(chrono::Utc::now());
    }
}
