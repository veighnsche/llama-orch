//! worker-error-handler — Worker-level error handling and recovery

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("cuda error: {0}")]
    CudaError(String),
    #[error("inference error: {0}")]
    InferenceError(String),
}

pub type Result<T> = std::result::Result<T, WorkerError>;

pub struct ErrorHandler;

impl ErrorHandler {
    pub fn new() -> Self {
        Self
    }
    
    pub fn handle_cuda_error(&self, error: &str) -> Result<()> {
        tracing::error!(error = %error, "CUDA error");
        Err(WorkerError::CudaError(error.to_string()))
    }
}

impl Default for ErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}
