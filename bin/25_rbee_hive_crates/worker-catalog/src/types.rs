// TEAM-273: Worker catalog types
use rbee_hive_artifact_catalog::{Artifact, ArtifactStatus};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Worker status (alias for ArtifactStatus)
pub type WorkerStatus = ArtifactStatus;

/// Worker type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerType {
    /// CPU-based LLM worker
    CpuLlm,
    /// CUDA-based LLM worker
    CudaLlm,
    /// Metal-based LLM worker (macOS)
    MetalLlm,
}

impl WorkerType {
    /// Get the binary name for this worker type
    ///
    /// TEAM-NARRATION-FIX: Updated to match actual binary names in Cargo.toml
    pub fn binary_name(&self) -> &str {
        match self {
            WorkerType::CpuLlm => "llm-worker-rbee-cpu",
            WorkerType::CudaLlm => "llm-worker-rbee-cuda",
            WorkerType::MetalLlm => "llm-worker-rbee-metal",
        }
    }

    /// Get the crate name (for building)
    ///
    /// TEAM-NARRATION-FIX: All workers are in the same crate
    pub fn crate_name(&self) -> &str {
        "llm-worker-rbee"
    }

    /// Get the features needed to build this worker type
    ///
    /// TEAM-NARRATION-FIX: Feature flags for compile-time backend selection
    pub fn build_features(&self) -> Option<&str> {
        match self {
            WorkerType::CpuLlm => Some("cpu"),
            WorkerType::CudaLlm => Some("cuda"),
            WorkerType::MetalLlm => Some("metal"),
        }
    }
}

/// Platform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Platform {
    /// Linux
    Linux,
    /// macOS
    MacOS,
    /// Windows
    Windows,
}

impl Platform {
    /// Get the current platform
    pub fn current() -> Self {
        #[cfg(target_os = "linux")]
        return Platform::Linux;

        #[cfg(target_os = "macos")]
        return Platform::MacOS;

        #[cfg(target_os = "windows")]
        return Platform::Windows;
    }

    /// Get the file extension for this platform
    pub fn extension(&self) -> &str {
        match self {
            Platform::Linux | Platform::MacOS => "",
            Platform::Windows => ".exe",
        }
    }
}

/// Worker binary entry in the catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerBinary {
    /// Unique worker ID (e.g., "cpu-llm-worker-rbee-v0.1.0-linux")
    id: String,

    /// Worker type
    worker_type: WorkerType,

    /// Platform
    platform: Platform,

    /// Filesystem path to worker binary
    path: PathBuf,

    /// Size in bytes
    size: u64,

    /// Current status
    status: ArtifactStatus,

    /// Version
    version: String,

    /// When the worker was added
    #[serde(default = "chrono::Utc::now")]
    added_at: chrono::DateTime<chrono::Utc>,
}

impl WorkerBinary {
    /// Create a new worker binary entry
    pub fn new(
        id: String,
        worker_type: WorkerType,
        platform: Platform,
        path: PathBuf,
        size: u64,
        version: String,
    ) -> Self {
        Self {
            id,
            worker_type,
            platform,
            path,
            size,
            status: ArtifactStatus::Available,
            version,
            added_at: chrono::Utc::now(),
        }
    }

    /// Get the worker type
    pub fn worker_type(&self) -> &WorkerType {
        &self.worker_type
    }

    /// Get the platform
    pub fn platform(&self) -> &Platform {
        &self.platform
    }

    /// Get the version
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Get when the worker was added
    pub fn added_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.added_at
    }
}

// Implement Artifact trait
impl Artifact for WorkerBinary {
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
        &self.id
    }
}
