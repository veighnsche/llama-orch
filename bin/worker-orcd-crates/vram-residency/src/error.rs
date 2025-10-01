//! Error types for VRAM residency
//!
//! All error types used throughout the crate.

use thiserror::Error;

/// Result type alias
pub type Result<T> = std::result::Result<T, VramError>;

/// VRAM residency errors
#[derive(Debug, Error)]
pub enum VramError {
    /// Insufficient VRAM available
    #[error("insufficient VRAM: need {0} bytes, have {1} bytes")]
    InsufficientVram(usize, usize),
    
    /// Seal verification failed
    #[error("seal verification failed: digest mismatch")]
    SealVerificationFailed,
    
    /// Model not sealed
    #[error("model not sealed")]
    NotSealed,
    
    /// VRAM integrity violation
    #[error("VRAM integrity violation")]
    IntegrityViolation,
    
    /// Invalid input
    #[error("invalid input: {0}")]
    InvalidInput(String),
    
    /// CUDA allocation failed
    #[error("CUDA allocation failed: {0}")]
    CudaAllocationFailed(String),
    
    /// Policy violation
    #[error("VRAM-only policy violation: {0}")]
    PolicyViolation(String),
    
    /// Configuration error
    #[error("configuration error: {0}")]
    ConfigError(String),
}

impl VramError {
    /// Check if error is retriable
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            VramError::InsufficientVram(_, _) | VramError::CudaAllocationFailed(_)
        )
    }
    
    /// Check if error is fatal
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            VramError::SealVerificationFailed
                | VramError::IntegrityViolation
                | VramError::PolicyViolation(_)
        )
    }
}
