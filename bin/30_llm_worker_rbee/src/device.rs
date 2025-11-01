// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Multi-backend device initialization

//! Device initialization and management
//!
//! Provides backend-specific device initialization with strict residency.
//! Each backend (CPU, CUDA, Metal) has its own initialization path.
//!
//! Created by: TEAM-007
//! Modified by: TEAM-018 (Removed Accelerate, added Metal)

use crate::narration::ACTION_DEVICE_INIT;
use candle_core::{Device, Result as CandleResult};
use observability_narration_core::n;

/// Initialize CPU device
#[cfg(feature = "cpu")]
pub fn init_cpu_device() -> CandleResult<Device> {
    tracing::info!("Initializing CPU device");

    n!(ACTION_DEVICE_INIT, "Initialized CPU device");

    Ok(Device::Cpu)
}

/// Initialize CUDA device
#[cfg(feature = "cuda")]
pub fn init_cuda_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing CUDA device {}", gpu_id);

    let device = Device::new_cuda(gpu_id)?;

    n!(ACTION_DEVICE_INIT, "Initialized CUDA device {}", gpu_id);

    Ok(device)
}

/// Initialize Apple Metal device (GPU)
/// Note: Metal is Apple's GPU API, equivalent to CUDA for NVIDIA
#[cfg(feature = "metal")]
pub fn init_metal_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing Apple Metal device (GPU) {}", gpu_id);

    let device = Device::new_metal(gpu_id)?;

    n!(ACTION_DEVICE_INIT, "Initialized Apple Metal device {}", gpu_id);

    Ok(device)
}

/// Verify device is available and working
/// Performs a simple smoke test: create tensor and verify operations
pub fn verify_device(device: &Device) -> CandleResult<()> {
    use candle_core::Tensor;

    // Simple smoke test: create tensor and verify
    let test = Tensor::zeros((2, 2), candle_core::DType::F32, device)?;
    let _sum = test.sum_all()?; // TEAM-010: Verify tensor operations work

    tracing::info!("Device verification passed: {:?}", device);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cpu")]
    fn test_cpu_device_init() {
        let device = init_cpu_device().unwrap();
        verify_device(&device).unwrap();
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_device_init() {
        // Only run if CUDA is available
        if let Ok(device) = init_cuda_device(0) {
            verify_device(&device).unwrap();
        }
    }

    #[test]
    #[cfg(feature = "metal")]
    fn test_metal_device_init() {
        // Only run if Metal is available
        if let Ok(device) = init_metal_device(0) {
            verify_device(&device).unwrap();
        }
    }
}
