//! Preflight checks for GPU-only enforcement and environment validations.

use anyhow::{anyhow, Result};

/// Return true if CUDA appears available (nvcc or nvidia-smi present).
pub fn cuda_available() -> bool {
    which::which("nvcc").is_ok() || which::which("nvidia-smi").is_ok()
}

/// Fail fast if GPU is not available or CUDA toolchain missing when GPU-only is required.
pub fn assert_gpu_only() -> Result<()> {
    if !cuda_available() {
        return Err(anyhow!(
            "GPU-only enforcement: CUDA toolkit or NVIDIA driver not detected (nvcc/nvidia-smi not found)"
        ));
    }
    Ok(())
}
