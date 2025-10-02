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
///
/// # Security
///
/// Prevents out-of-bounds device access and integer overflow.
pub fn validate_gpu_device(gpu_device: u32, max_devices: u32) -> Result<()> {
    // Check if max_devices is reasonable (prevent overflow)
    if max_devices == 0 {
        return Err(VramError::InvalidInput(
            "max_devices cannot be zero".to_string()
        ));
    }
    
    if max_devices > 16 {
        return Err(VramError::InvalidInput(format!(
            "max_devices too large: {} (max: 16)",
            max_devices
        )));
    }
    
    // Check device index is in range
    if gpu_device >= max_devices {
        return Err(VramError::InvalidInput(format!(
            "GPU device {} out of range (available: 0-{})",
            gpu_device,
            max_devices.saturating_sub(1)
        )));
    }
    
    tracing::debug!(
        gpu_device = %gpu_device,
        max_devices = %max_devices,
        "GPU device index validated"
    );
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_device_in_range() {
        assert!(validate_gpu_device(0, 1).is_ok());
        assert!(validate_gpu_device(0, 4).is_ok());
        assert!(validate_gpu_device(3, 4).is_ok());
        assert!(validate_gpu_device(7, 8).is_ok());
    }

    #[test]
    fn test_device_zero_with_one_device() {
        assert!(validate_gpu_device(0, 1).is_ok());
    }

    #[test]
    fn test_device_out_of_range_rejected() {
        assert!(validate_gpu_device(4, 4).is_err());
        assert!(validate_gpu_device(10, 4).is_err());
        assert!(validate_gpu_device(999, 8).is_err());
    }

    #[test]
    fn test_max_devices_zero_rejected() {
        let result = validate_gpu_device(0, 0);
        assert!(result.is_err());
        assert!(matches!(result, Err(VramError::InvalidInput(_))));
    }

    #[test]
    fn test_max_devices_too_large_rejected() {
        assert!(validate_gpu_device(0, 17).is_err());
        assert!(validate_gpu_device(0, 100).is_err());
        assert!(validate_gpu_device(0, 999).is_err());
    }

    #[test]
    fn test_boundary_device_equals_max_rejected() {
        assert!(validate_gpu_device(4, 4).is_err());
        assert!(validate_gpu_device(16, 16).is_err());
    }

    #[test]
    fn test_boundary_device_one_less_than_max_accepted() {
        assert!(validate_gpu_device(3, 4).is_ok());
        assert!(validate_gpu_device(15, 16).is_ok());
    }

    #[test]
    fn test_max_allowed_devices_16() {
        assert!(validate_gpu_device(15, 16).is_ok());
        assert!(validate_gpu_device(0, 16).is_ok());
    }
}
