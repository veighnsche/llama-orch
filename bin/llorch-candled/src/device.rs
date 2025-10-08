//! Device initialization and management
//!
//! Provides backend-specific device initialization with strict residency.
//! Each backend (CPU, CUDA, Accelerate) has its own initialization path.
//!
//! Created by: TEAM-007

use candle_core::{Device, Result as CandleResult};

/// Initialize CPU device
#[cfg(feature = "cpu")]
pub fn init_cpu_device() -> CandleResult<Device> {
    tracing::info!("Initializing CPU device");
    Ok(Device::Cpu)
}

/// Initialize CUDA device
#[cfg(feature = "cuda")]
pub fn init_cuda_device(gpu_id: usize) -> CandleResult<Device> {
    tracing::info!("Initializing CUDA device {}", gpu_id);
    Device::new_cuda(gpu_id)
}

/// Initialize Apple Accelerate device
/// Note: Accelerate is CPU-based with Apple framework optimizations, NOT Metal (GPU)
#[cfg(feature = "accelerate")]
pub fn init_accelerate_device() -> CandleResult<Device> {
    tracing::info!("Initializing Apple Accelerate device (CPU-optimized)");
    // Accelerate is CPU-based, not Metal
    Ok(Device::Cpu)
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
    #[cfg(feature = "accelerate")]
    fn test_accelerate_device_init() {
        let device = init_accelerate_device().unwrap();
        verify_device(&device).unwrap();
    }
}
