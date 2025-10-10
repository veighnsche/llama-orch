//! Error types for GPU detection

use thiserror::Error;

/// GPU detection errors
#[derive(Debug, Error)]
pub enum GpuError {
    /// No GPU detected
    #[error("No NVIDIA GPU detected. This application requires an NVIDIA GPU with CUDA support.")]
    NoGpuDetected,

    /// nvidia-smi not found in PATH
    #[error("nvidia-smi not found in PATH. Please install NVIDIA drivers.")]
    NvidiaSmiNotFound,

    /// Failed to parse nvidia-smi output
    #[error("Failed to parse nvidia-smi output: {0}")]
    NvidiaSmiParseFailed(String),

    /// Invalid device index
    #[error("Invalid GPU device index: {0} (available: 0-{1})")]
    InvalidDevice(u32, usize),

    /// CUDA runtime error (only with cuda-runtime feature)
    #[cfg(feature = "cuda-runtime")]
    #[error("CUDA runtime error: {0}")]
    CudaRuntimeError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Other errors (TEAM-052: for backend detection)
    #[error("{0}")]
    Other(String),
}

/// Result type for GPU operations
pub type Result<T> = std::result::Result<T, GpuError>;
