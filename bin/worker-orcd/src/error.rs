//! Worker error types

use crate::cuda::CudaError;

/// Worker-level errors
#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    
    #[error("Inference timeout")]
    Timeout,
    
    #[error("Worker unhealthy: {0}")]
    Unhealthy(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl WorkerError {
    /// Get stable error code for API responses
    pub fn code(&self) -> &'static str {
        match self {
            Self::Cuda(CudaError::OutOfMemory { .. }) => "VRAM_OOM",
            Self::Cuda(CudaError::ModelLoadFailed(_)) => "MODEL_LOAD_FAILED",
            Self::Cuda(CudaError::InferenceFailed(_)) => "INFERENCE_FAILED",
            Self::Cuda(CudaError::VramResidencyFailed) => "VRAM_RESIDENCY_FAILED",
            Self::Cuda(_) => "CUDA_ERROR",
            Self::InvalidRequest(_) => "INVALID_REQUEST",
            Self::Timeout => "INFERENCE_TIMEOUT",
            Self::Unhealthy(_) => "WORKER_UNHEALTHY",
            Self::Internal(_) => "INTERNAL",
        }
    }
    
    /// Check if error is retriable by orchestrator
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            Self::Cuda(CudaError::OutOfMemory { .. })
                | Self::Timeout
                | Self::Internal(_)
        )
    }
    
    /// Get HTTP status code
    pub fn status_code(&self) -> u16 {
        match self {
            Self::InvalidRequest(_) => 400,
            Self::Unhealthy(_) => 503,
            _ => 500,
        }
    }
}
