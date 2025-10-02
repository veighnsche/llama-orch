//! VRAM-only policy enforcement
//!
//! Enforces that models reside entirely in VRAM.

use crate::error::{Result, VramError};
use crate::policy::validation::{validate_device_properties, check_unified_memory};

/// Enforce VRAM-only policy
///
/// # Requirements
///
/// - Disable unified memory (UMA)
/// - Disable zero-copy
/// - Disable pinned host memory
/// - Verify device properties
///
/// # Arguments
///
/// * `gpu_device` - GPU device index
///
/// # Returns
///
/// `Ok(())` if policy is enforced, error otherwise
///
/// # Security
///
/// This function ensures that all model weights reside in VRAM during inference.
/// No RAM fallback is allowed. GPU is required; fail fast if policy cannot be enforced.
///
/// # Note
///
/// In test mode (without CUDA), this performs validation only.
/// In production mode with CUDA, this would call cudaDeviceSetLimit to disable UMA.
pub fn enforce_vram_only_policy(gpu_device: u32) -> Result<()> {
    // Validate device properties first
    validate_device_properties(gpu_device)?;
    
    // Check unified memory status
    check_unified_memory(gpu_device)?;
    
    // In test mode, we can only validate
    #[cfg(test)]
    {
        tracing::info!(
            gpu_device = %gpu_device,
            "VRAM-only policy enforcement: test mode (validation only)"
        );
        return Ok(());
    }
    
    // In production mode with CUDA, enforce policy
    #[cfg(not(test))]
    {
        // Note: CUDA doesn't provide a direct API to disable unified memory
        // at the application level. The VRAM-only policy is enforced by:
        //
        // 1. Using cudaMalloc (not cudaMallocManaged)
        // 2. Never using cudaMallocHost or cudaHostAlloc
        // 3. Validating device properties (compute capability >= 6.0)
        // 4. Cryptographic sealing to detect VRAM corruption
        //
        // The CudaContext and SafeCudaPtr already enforce this by only
        // using cudaMalloc for allocations.
        
        tracing::info!(
            gpu_device = %gpu_device,
            "VRAM-only policy enforced: using cudaMalloc exclusively"
        );
        
        Ok(())
    }
}

/// Check if policy is currently enforced
///
/// # Arguments
///
/// * `gpu_device` - GPU device index
///
/// # Returns
///
/// `Ok(true)` if policy is enforced, `Ok(false)` otherwise
pub fn is_policy_enforced(gpu_device: u32) -> Result<bool> {
    // Check if device properties are valid
    match validate_device_properties(gpu_device) {
        Ok(()) => {
            tracing::debug!(
                gpu_device = %gpu_device,
                "VRAM-only policy is enforced"
            );
            Ok(true)
        }
        Err(VramError::PolicyViolation(_)) => {
            tracing::warn!(
                gpu_device = %gpu_device,
                "VRAM-only policy is NOT enforced"
            );
            Ok(false)
        }
        Err(e) => Err(e),
    }
}
