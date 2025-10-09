//! Error types for llm-worker-rbee
//!
//! Created by: TEAM-000

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlorchError {
    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Tensor error: {0}")]
    TensorError(String),

    #[error("GGUF parsing error: {0}")]
    GgufError(String),

    #[error("Checkpoint validation error: {0}")]
    CheckpointError(String),

    #[error("CUDA error: {0}")]
    #[cfg(feature = "cuda")]
    CudaError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
