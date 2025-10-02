//! VRAM-only policy enforcement for workers.
//!
//! Ensures a single model stays resident in GPU VRAM without RAM fallback.

use thiserror::Error;

/// VRAM policy errors.
#[derive(Debug, Error)]
pub enum VramError {
    #[error("insufficient VRAM: need {needed} bytes, have {available} bytes")]
    InsufficientVram { needed: u64, available: u64 },

    #[error("VRAM-only policy violation: {0}")]
    PolicyViolation(String),

    #[error("CUDA error: {0}")]
    CudaError(String),
}

/// VRAM-only policy enforcer.
pub struct VramPolicy {
    gpu_device: u32,
}

impl VramPolicy {
    /// Create policy enforcer for a GPU device.
    pub fn new(gpu_device: u32) -> Result<Self, VramError> {
        tracing::info!(gpu_device, "Initializing VRAM policy");
        Ok(Self { gpu_device })
    }

    /// Enforce VRAM-only policy: disable UMA, zero-copy, pinned memory.
    pub fn enforce_vram_only(&self) -> Result<(), VramError> {
        tracing::info!(gpu_device = self.gpu_device, "Enforcing VRAM-only policy");

        // TODO: Actual CUDA FFI calls to:
        // - Disable unified memory (UMA)
        // - Disable zero-copy mode
        // - Disable pinned host memory
        // - Verify no BAR/Resizable-BAR modes

        Ok(())
    }

    /// Load model bytes to VRAM. Returns actual VRAM bytes used.
    pub fn load_model_to_vram(&self, _model_bytes: &[u8]) -> Result<u64, VramError> {
        tracing::info!(
            gpu_device = self.gpu_device,
            size = _model_bytes.len(),
            "Loading model to VRAM"
        );

        // TODO: Actual CUDA FFI calls to:
        // - cudaMalloc() for model size
        // - cudaMemcpy() to copy model bytes
        // - Return actual VRAM usage

        // Stub: just return input size
        Ok(_model_bytes.len() as u64)
    }

    /// Verify model is still resident in VRAM (health check).
    pub fn verify_vram_residency(&self) -> Result<(), VramError> {
        // TODO: Verify no RAM fallback has occurred

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enforce_vram_only_succeeds() {
        let policy = VramPolicy::new(0).unwrap();
        assert!(policy.enforce_vram_only().is_ok());
    }

    #[test]
    fn test_load_model_returns_size() {
        let policy = VramPolicy::new(0).unwrap();
        let model_bytes = vec![0u8; 1024];
        let vram_used = policy.load_model_to_vram(&model_bytes).unwrap();
        assert_eq!(vram_used, 1024);
    }
}
