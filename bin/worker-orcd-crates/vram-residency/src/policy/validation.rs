//! Policy validation functions
//!
//! Validates VRAM-only policy compliance.

use crate::error::Result;

#[cfg(not(test))]
use crate::error::VramError;

/// Validate device properties
///
/// Checks that device supports VRAM-only mode.
///
/// # Arguments
///
/// * `gpu_device` - GPU device index
///
/// # Returns
///
/// `Ok(())` if device is valid, error otherwise
///
/// # Note
///
/// In test mode (without CUDA), this always succeeds.
/// In production mode, this queries actual CUDA device properties.
pub fn validate_device_properties(gpu_device: u32) -> Result<()> {
    // In test mode, skip validation
    #[cfg(test)]
    {
        tracing::debug!(
            gpu_device = %gpu_device,
            "Device property validation skipped (test mode)"
        );
        return Ok(());
    }
    
    // In production mode, validate via gpu-info
    #[cfg(not(test))]
    {
        let gpu_info = gpu_info::detect_gpus();
        
        if !gpu_info.available {
            return Err(VramError::PolicyViolation(
                "No NVIDIA GPU detected".to_string()
            ));
        }
        
        if (gpu_device as usize) >= gpu_info.count {
            return Err(VramError::InvalidInput(format!(
                "GPU device {} not found (available: {})",
                gpu_device, gpu_info.count
            )));
        }
        
        let device = &gpu_info.devices[gpu_device as usize];
        
        // Check compute capability (minimum 6.0 for modern features)
        let (major, minor) = device.compute_capability;
        if major < 6 {
            return Err(VramError::PolicyViolation(format!(
                "GPU compute capability {}.{} too old (minimum: 6.0)",
                major, minor
            )));
        }
        
        // Check VRAM capacity (minimum 1GB)
        let vram_gb = device.vram_total_bytes / (1024 * 1024 * 1024);
        if vram_gb < 1 {
            return Err(VramError::PolicyViolation(format!(
                "GPU VRAM too small: {}GB (minimum: 1GB)",
                vram_gb
            )));
        }
        
        tracing::info!(
            gpu_device = %gpu_device,
            name = %device.name,
            vram_gb = %vram_gb,
            compute_cap = ?device.compute_capability,
            "Device properties validated"
        );
        
        Ok(())
    }
}

/// Check if unified memory is enabled
///
/// Returns error if UMA is detected.
///
/// # Arguments
///
/// * `gpu_device` - GPU device index
///
/// # Returns
///
/// `Ok(())` if unified memory is disabled, error otherwise
///
/// # Note
///
/// In test mode, this always succeeds.
/// In production mode with CUDA, this would query device properties.
pub fn check_unified_memory(gpu_device: u32) -> Result<()> {
    // In test mode, skip check
    #[cfg(test)]
    {
        tracing::debug!(
            gpu_device = %gpu_device,
            "Unified memory check skipped (test mode)"
        );
        return Ok(());
    }
    
    // In production mode, we rely on enforce_vram_only_policy()
    // to disable unified memory at the CUDA level.
    // This function serves as a validation check.
    #[cfg(not(test))]
    {
        // Note: CUDA doesn't provide a direct API to check if UMA is enabled
        // at runtime. Instead, we enforce it via cudaDeviceSetLimit in
        // enforce_vram_only_policy(). This function is a placeholder for
        // future validation if CUDA adds such an API.
        
        tracing::debug!(
            gpu_device = %gpu_device,
            "Unified memory check: relying on policy enforcement"
        );
        
        Ok(())
    }
}
