//! Error types for llorch-cpud

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Tensor operation error: {0}")]
    TensorOp(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Worker error: {0}")]
    Worker(String),
}

impl From<worker_common::WorkerError> for Error {
    fn from(e: worker_common::WorkerError) -> Self {
        Error::Worker(e.to_string())
    }
}
