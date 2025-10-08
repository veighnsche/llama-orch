//! RMSNorm (Root Mean Square Normalization)
//!
//! Llama-2 uses RMSNorm instead of LayerNorm:
//! - Simpler: no mean subtraction
//! - No bias term
//! - Formula: x / sqrt(mean(x²) + eps) * weight
//!
//! Checkpoint 1 validation target
//!
//! Hybrid implementation using Candle:
//! - Uses candle_nn::ops::rms_norm for optimized math
//! - Automatic CUDA kernel selection when available
//! - Our architecture, Candle's math (TEAM_001_CANDLE_CATALOG_PLAN.md)
//!
//! Created by: TEAM-000
//! Modified by: TEAM-001

use candle_core::{Tensor, Result as CandleResult, Device, DType};
use candle_nn::ops::rms_norm as candle_rms_norm;

/// RMSNorm layer for Llama-2
///
/// Simpler than LayerNorm - no mean subtraction, no bias
/// Uses Candle's optimized rms_norm function (automatically uses CUDA when available)
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
    device: Device,
}

impl RMSNorm {
    /// Create new RMSNorm layer from Candle tensors
    ///
    /// Uses Candle's optimized implementation with automatic CUDA kernel selection
    pub fn new(weight: Tensor, eps: f64) -> CandleResult<Self> {
        let device = weight.device().clone();
        Ok(Self { weight, eps, device })
    }

    /// Create from raw f32 array (for testing and compatibility)
    ///
    /// Converts ndarray-style weights to Candle tensors
    pub fn from_array(weight: &[f32], eps: f64, device: &Device) -> CandleResult<Self> {
        let weight_tensor = Tensor::from_slice(weight, weight.len(), device)?;
        Self::new(weight_tensor, eps)
    }

    /// Forward pass using Candle's optimized RmsNorm
    ///
    /// Automatically uses CUDA kernel if available, CPU fallback otherwise
    /// Formula: x / sqrt(mean(x²) + eps) * weight
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Use Candle's optimized RmsNorm function
        // This automatically selects CUDA kernel when available
        candle_rms_norm(x, &self.weight, self.eps as f32)
    }

    /// Get the device this layer is on
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rms_norm_shape() -> CandleResult<()> {
        let device = Device::Cpu;
        let hidden_size = 4096;
        let seq_len = 2;

        // Create weight tensor
        let weight = vec![1.0f32; hidden_size];
        let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

        // Create input tensor [seq_len, hidden_size]
        let input = Tensor::randn(0f32, 1.0, (seq_len, hidden_size), &device)?;
        
        // Forward pass
        let output = norm.forward(&input)?;

        // Validate shape
        assert_eq!(output.shape().dims(), &[seq_len, hidden_size]);
        
        Ok(())
    }

    #[test]
    fn test_rms_norm_no_nan() -> CandleResult<()> {
        let device = Device::Cpu;
        let weight = vec![1.0f32; 128];
        let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

        let input = Tensor::randn(0f32, 1.0, (2, 128), &device)?;
        let output = norm.forward(&input)?;

        // Check no NaN values
        let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
        assert!(output_vec.iter().all(|&v| !v.is_nan()));
        
        Ok(())
    }
}
