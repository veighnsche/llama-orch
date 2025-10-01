//! GPU device validation
//!
//! Validates GPU device indices.

use crate::error::{Result, VramError};

/// Validate GPU device index
///
/// # Arguments
///
/// * `gpu_device` - GPU device index
/// * `max_devices` - Maximum number of devices
///
/// # Returns
///
/// `Ok(())` if valid, error otherwise
pub fn validate_gpu_device(gpu_device: u32, max_devices: u32) -> Result<()> {
    // TODO: Integrate with input-validation crate
    // - Use validate_range()
    // - Check against actual GPU count
    
    if gpu_device >= max_devices {
        return Err(VramError::InvalidInput(format!(
            "GPU device {} out of range (max: {})",
            gpu_device, max_devices
        )));
    }
    
    Ok(())
}
