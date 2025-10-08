//! TEAM-002 Critical Review: Edge Cases & Numerical Stability Tests
//!
//! Additional tests to validate TEAM-001's RMSNorm implementation
//! Testing edge cases, numerical stability, and extreme values
//!
//! Created by: TEAM-002

use llorch_candled::layers::RMSNorm;
use candle_core::{Tensor, Device};

#[test]
fn test_zero_input() -> candle_core::Result<()> {
    println!("\n🔍 TEAM-002: Testing zero input (edge case)");
    
    let device = Device::Cpu;
    let hidden_size = 4096;
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    // Zero input - epsilon should prevent division by zero
    let input = Tensor::zeros((2, hidden_size), candle_core::DType::F32, &device)?;
    let output = norm.forward(&input)?;
    
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    
    // Should not contain NaN (eps prevents division by zero)
    assert!(output_vec.iter().all(|&v| !v.is_nan()), "Zero input produced NaN");
    assert!(output_vec.iter().all(|&v| v.is_finite()), "Zero input produced Inf");
    
    println!("✅ Zero input handled correctly (no NaN/Inf)");
    Ok(())
}

#[test]
fn test_very_large_values() -> candle_core::Result<()> {
    println!("\n🔍 TEAM-002: Testing very large values");
    
    let device = Device::Cpu;
    let hidden_size = 128;
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    // Very large values (but not overflow)
    let input = Tensor::ones((2, hidden_size), candle_core::DType::F32, &device)?.affine(1e6, 0.0)?;
    let output = norm.forward(&input)?;
    
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    
    assert!(output_vec.iter().all(|&v| !v.is_nan()), "Large values produced NaN");
    assert!(output_vec.iter().all(|&v| v.is_finite()), "Large values produced Inf");
    
    println!("✅ Large values handled correctly");
    Ok(())
}

#[test]
fn test_very_small_values() -> candle_core::Result<()> {
    println!("\n🔍 TEAM-002: Testing very small values");
    
    let device = Device::Cpu;
    let hidden_size = 128;
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    // Very small values
    let input = Tensor::ones((2, hidden_size), candle_core::DType::F32, &device)?.affine(1e-10, 0.0)?;
    let output = norm.forward(&input)?;
    
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    
    assert!(output_vec.iter().all(|&v| !v.is_nan()), "Small values produced NaN");
    assert!(output_vec.iter().all(|&v| v.is_finite()), "Small values produced Inf");
    
    println!("✅ Small values handled correctly");
    Ok(())
}

#[test]
fn test_mixed_signs() -> candle_core::Result<()> {
    println!("\n🔍 TEAM-002: Testing mixed positive/negative values");
    
    let device = Device::Cpu;
    let hidden_size = 128;
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    // Alternating positive/negative
    let input_data: Vec<f32> = (0..256)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let input = Tensor::from_vec(input_data, (2, hidden_size), &device)?;
    
    let output = norm.forward(&input)?;
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    
    assert!(output_vec.iter().all(|&v| !v.is_nan()), "Mixed signs produced NaN");
    assert!(output_vec.iter().all(|&v| v.is_finite()), "Mixed signs produced Inf");
    
    println!("✅ Mixed signs handled correctly");
    Ok(())
}

#[test]
fn test_single_token() -> candle_core::Result<()> {
    println!("\n🔍 TEAM-002: Testing single token (batch_size=1)");
    
    let device = Device::Cpu;
    let hidden_size = 4096;
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    // Single token
    let input = Tensor::randn(0f32, 1.0, (1, hidden_size), &device)?;
    let output = norm.forward(&input)?;
    
    assert_eq!(output.shape().dims(), &[1, hidden_size]);
    
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    let mean_sq: f32 = output_vec.iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;
    let rms = mean_sq.sqrt();
    
    assert!((rms - 1.0).abs() < 0.01, "Single token RMS should be ~1.0, got {}", rms);
    
    println!("✅ Single token handled correctly");
    Ok(())
}

#[test]
fn test_large_batch() -> candle_core::Result<()> {
    println!("\n🔍 TEAM-002: Testing large batch");
    
    let device = Device::Cpu;
    let hidden_size = 128;
    let batch_size = 100;
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    let input = Tensor::randn(0f32, 1.0, (batch_size, hidden_size), &device)?;
    let output = norm.forward(&input)?;
    
    assert_eq!(output.shape().dims(), &[batch_size, hidden_size]);
    
    let output_vec = output.to_vec2::<f32>()?;
    
    // Each row should be normalized independently
    for (row_idx, row) in output_vec.iter().enumerate() {
        let mean_sq: f32 = row.iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;
        let rms = mean_sq.sqrt();
        assert!(
            (rms - 1.0).abs() < 0.01,
            "Row {} RMS should be ~1.0, got {}", row_idx, rms
        );
    }
    
    println!("✅ Large batch handled correctly");
    Ok(())
}

#[test]
fn test_epsilon_importance() -> candle_core::Result<()> {
    println!("\n🔍 TEAM-002: Testing epsilon importance");
    
    let device = Device::Cpu;
    let hidden_size = 128;
    let weight = vec![1.0f32; hidden_size];
    
    let norm_small_eps = RMSNorm::from_array(&weight, 1e-10, &device)?;
    let norm_large_eps = RMSNorm::from_array(&weight, 1e-3, &device)?;

    // Near-zero input to emphasize epsilon effect
    let input = Tensor::ones((2, hidden_size), candle_core::DType::F32, &device)?.affine(1e-8, 0.0)?;
    
    let out_small = norm_small_eps.forward(&input)?;
    let out_large = norm_large_eps.forward(&input)?;
    
    let vec_small = out_small.flatten_all()?.to_vec1::<f32>()?;
    let vec_large = out_large.flatten_all()?.to_vec1::<f32>()?;
    
    // Outputs should differ based on epsilon
    let max_diff = vec_small.iter()
        .zip(vec_large.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    assert!(max_diff > 1e-6, "Epsilon should affect output, max_diff={}", max_diff);
    
    println!("✅ Epsilon correctly affects output (max_diff={})", max_diff);
    Ok(())
}

#[test]
fn test_negative_weights() -> candle_core::Result<()> {
    println!("\n🔍 TEAM-002: Testing negative weights");
    
    let device = Device::Cpu;
    let hidden_size = 128;
    
    // Negative weights (valid in neural networks)
    let weight = vec![-1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    let input = Tensor::randn(0f32, 1.0, (2, hidden_size), &device)?;
    let output = norm.forward(&input)?;
    
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    
    assert!(output_vec.iter().all(|&v| !v.is_nan()), "Negative weights produced NaN");
    assert!(output_vec.iter().all(|&v| v.is_finite()), "Negative weights produced Inf");
    
    // With negative weight, RMS should be ~1.0 (absolute value)
    let mean_sq: f32 = output_vec.iter().map(|&x| x * x).sum::<f32>() / (2 * hidden_size) as f32;
    let rms = mean_sq.sqrt();
    assert!((rms - 1.0).abs() < 0.1, "RMS with negative weights should be ~1.0, got {}", rms);
    
    println!("✅ Negative weights handled correctly");
    Ok(())
}

#[test]
fn test_candle_formula_verification() -> candle_core::Result<()> {
    println!("\n🔍 TEAM-002: Verifying Candle's RMS formula matches spec");
    
    let device = Device::Cpu;
    let hidden_size = 8; // Small for manual verification
    let seq_len = 1;
    
    let weight = vec![2.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    // Simple input for manual calculation
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Tensor::from_vec(input_data.clone(), (seq_len, hidden_size), &device)?;
    
    // Manual calculation
    let sum_sq: f32 = input_data.iter().map(|&x| x * x).sum();
    let mean_sq = sum_sq / hidden_size as f32;
    let rms = (mean_sq + 1e-5).sqrt();
    let expected: Vec<f32> = input_data.iter().map(|&x| (x / rms) * 2.0).collect();
    
    println!("  Manual calculation:");
    println!("    sum_sq={}, mean_sq={}, rms={}", sum_sq, mean_sq, rms);
    println!("    expected output: {:?}", expected);
    
    // Candle output
    let output = norm.forward(&input)?;
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    
    println!("  Candle output: {:?}", output_vec);
    
    // Compare
    for (i, (&exp, &got)) in expected.iter().zip(output_vec.iter()).enumerate() {
        let diff = (exp - got).abs();
        assert!(diff < 1e-5, "Element {} differs: expected={}, got={}, diff={}", i, exp, got, diff);
    }
    
    println!("✅ Candle formula matches manual calculation");
    Ok(())
}

#[test]
fn test_determinism_across_runs() -> candle_core::Result<()> {
    println!("\n🔍 TEAM-002: Testing determinism across multiple runs");
    
    let device = Device::Cpu;
    let hidden_size = 4096;
    let weight = vec![1.0f32; hidden_size];
    
    // Fixed seed input
    let input_data: Vec<f32> = (0..8192)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    let input = Tensor::from_vec(input_data, (2, hidden_size), &device)?;
    
    // Run 10 times
    let mut outputs = Vec::new();
    for _ in 0..10 {
        let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;
        let output = norm.forward(&input)?;
        outputs.push(output.flatten_all()?.to_vec1::<f32>()?);
    }
    
    // All outputs must be bit-exact
    for run_idx in 1..10 {
        for (i, (&v0, &v)) in outputs[0].iter().zip(outputs[run_idx].iter()).enumerate() {
            assert_eq!(
                v0.to_bits(), v.to_bits(),
                "Run {} differs from run 0 at element {}: {} vs {}", 
                run_idx, i, v0, v
            );
        }
    }
    
    println!("✅ Determinism verified across 10 runs");
    Ok(())
}
