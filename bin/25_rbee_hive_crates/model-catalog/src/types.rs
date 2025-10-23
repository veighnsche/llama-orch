// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-267: Implemented model catalog types

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Model catalog entry
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelEntry {
    /// Unique model ID (e.g., "meta-llama/Llama-2-7b-chat-hf")
    pub id: String,

    /// Display name
    pub name: String,

    /// Local path where model files are stored
    pub path: PathBuf,

    /// Model size in bytes
    pub size_bytes: u64,

    /// When the model was added to catalog
    pub added_at: chrono::DateTime<chrono::Utc>,

    /// Model status
    pub status: ModelStatus,

    /// Optional metadata
    pub metadata: Option<ModelMetadata>,
}

/// Model status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    /// Model is fully downloaded and ready
    Ready,

    /// Model is currently being downloaded
    Downloading {
        /// Download progress (0.0 to 1.0)
        progress: f32,
    },

    /// Model download failed
    Failed {
        /// Error message describing the failure
        error: String,
    },
}

/// Optional model metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetadata {
    /// Model architecture (e.g., "llama", "mistral")
    pub architecture: Option<String>,

    /// Parameter count (e.g., "7B", "13B")
    pub parameters: Option<String>,

    /// Quantization (e.g., "Q4_K_M", "Q5_K_S")
    pub quantization: Option<String>,

    /// HuggingFace repo URL
    pub source_url: Option<String>,
}

impl ModelEntry {
    /// Create a new model entry with default values
    pub fn new(id: String, name: String, path: PathBuf, size_bytes: u64) -> Self {
        Self {
            id,
            name,
            path,
            size_bytes,
            added_at: chrono::Utc::now(),
            status: ModelStatus::Ready,
            metadata: None,
        }
    }

    /// Add metadata to the model entry
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Check if the model is ready for use
    pub fn is_ready(&self) -> bool {
        matches!(self.status, ModelStatus::Ready)
    }
}
