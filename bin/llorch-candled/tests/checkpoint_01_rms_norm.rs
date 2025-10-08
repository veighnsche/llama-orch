//! Checkpoint 1: RMSNorm Validation
//!
//! Tests RMSNorm implementation using Candle's optimized functions.
//! Following TEAM_001_CANDLE_CATALOG_PLAN.md: Use Candle for math, our architecture.
//!
//! Created by: TEAM-001

use llorch_candled::layers::RMSNorm;
use candle_core::{Tensor, Device, DType};
use approx::assert_abs_diff_eq;

/// Generate deterministic test input for RMSNorm validation
/// Pattern: sequential values scaled to realistic magnitude
fn generate_test_input(device: &Device) -> candle_core::Result<Tensor> {
    // Simple deterministic input: 2 tokens, 4096 dimensions (Llama-2 hidden size)
    let seq_len = 2;
    let hidden_size = 4096;
    
    let data: Vec<f32> = (0..seq_len * hidden_size)
        .map(|i| {
            let idx = i as f32;
            (idx * 0.001).sin() * 0.5 // Range: [-0.5, 0.5]
        })
        .collect();
    
    Tensor::from_vec(data, (seq_len, hidden_size), device)
}

#[test]
fn test_rms_norm_shape() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let hidden_size = 4096;
    let seq_len = 2;

    // Create RMSNorm with ones weight (standard initialization)
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    // Generate test input
    let input = generate_test_input(&device)?;
    
    // Forward pass
    let output = norm.forward(&input)?;

    // Validate shape matches input
    assert_eq!(output.shape().dims(), &[seq_len, hidden_size]);
    
    Ok(())
}

#[test]
fn test_rms_norm_no_nan() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let hidden_size = 4096;
    
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    let input = generate_test_input(&device)?;
    let output = norm.forward(&input)?;

    // Check no NaN values
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
    assert!(output_vec.iter().all(|&v| !v.is_nan()), "Output contains NaN values");
    assert!(output_vec.iter().all(|&v| v.is_finite()), "Output contains infinite values");
    
    Ok(())
}

#[test]
fn test_rms_norm_determinism() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 1: RMSNorm Determinism Test                 ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 4096;
    
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    let input = generate_test_input(&device)?;

    // Run 3 times - must be bit-exact
    let out1 = norm.forward(&input)?;
    let out2 = norm.forward(&input)?;
    let out3 = norm.forward(&input)?;

    let vec1 = out1.flatten_all()?.to_vec1::<f32>()?;
    let vec2 = out2.flatten_all()?.to_vec1::<f32>()?;
    let vec3 = out3.flatten_all()?.to_vec1::<f32>()?;

    // Bit-exact comparison
    for (i, ((v1, v2), v3)) in vec1.iter().zip(vec2.iter()).zip(vec3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "Run 1 vs 2 differ at element {}", i);
        assert_eq!(v2.to_bits(), v3.to_bits(), "Run 2 vs 3 differ at element {}", i);
    }

    let sample: Vec<f32> = vec1.iter().take(5).copied().collect();
    println!("RMSNorm output (first 5): {:?}", sample);
    println!("✅ RMSNorm implementation is deterministic");
    
    Ok(())
}

#[test]
fn test_rms_norm_normalization_properties() -> candle_core::Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 1: RMSNorm Mathematical Properties          ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 128; // Smaller for easier verification
    let seq_len = 2;
    
    // Create RMSNorm with ones weight
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    // Create simple test input
    let input_data: Vec<f32> = (0..seq_len * hidden_size)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let input = Tensor::from_vec(input_data.clone(), (seq_len, hidden_size), &device)?;
    
    // Forward pass
    let output = norm.forward(&input)?;
    let output_vec = output.to_vec2::<f32>()?;

    println!("\n📊 Input Statistics:");
    for row_idx in 0..seq_len {
        let row = &input_data[row_idx * hidden_size..(row_idx + 1) * hidden_size];
        let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
        let mean_sq: f32 = row.iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;
        let rms = (mean_sq + 1e-5).sqrt();
        println!("  Row {}: mean={:.6}, mean_sq={:.6}, rms={:.6}", row_idx, mean, mean_sq, rms);
    }

    println!("\n📊 Output Statistics:");
    for row_idx in 0..seq_len {
        let row = &output_vec[row_idx];
        let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
        let mean_sq: f32 = row.iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;
        let rms = mean_sq.sqrt();
        
        println!("  Row {}: mean={:.6}, mean_sq={:.6}, rms={:.6}", row_idx, mean, mean_sq, rms);
        
        // After RMSNorm with weight=1, RMS should be approximately 1
        // (within tolerance due to epsilon)
        assert!(
            (rms - 1.0).abs() < 0.01,
            "Row {} RMS should be ~1.0, got {}", row_idx, rms
        );
        
        // Check reasonable range
        let min = row.iter().copied().fold(f32::INFINITY, f32::min);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!("       Range: [{:.6}, {:.6}]", min, max);
        assert!(min > -10.0 && max < 10.0, "Values should be in reasonable range");
    }

    println!("\n✅ RMSNorm mathematical properties verified");
    
    Ok(())
}

#[test]
fn test_rms_norm_with_scale() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let hidden_size = 128;
    
    // Create RMSNorm with scale weight (2.0)
    let weight = vec![2.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    let input_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.1).collect();
    let input = Tensor::from_vec(input_data, (1, hidden_size), &device)?;
    
    let output = norm.forward(&input)?;
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;

    // With weight=2.0, RMS should be approximately 2.0
    let mean_sq: f32 = output_vec.iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;
    let rms = mean_sq.sqrt();
    
    assert!(
        (rms - 2.0).abs() < 0.1,
        "RMS with weight=2.0 should be ~2.0, got {}", rms
    );
    
    Ok(())
}

#[test]
fn test_rms_norm_batch() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let hidden_size = 128;
    let batch_size = 4;
    
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device)?;

    // Create batch input with different patterns per row
    let input_data: Vec<f32> = (0..batch_size * hidden_size)
        .map(|i| {
            let row = i / hidden_size;
            let col = i % hidden_size;
            ((row + 1) as f32) * (col as f32) * 0.01
        })
        .collect();
    let input = Tensor::from_vec(input_data, (batch_size, hidden_size), &device)?;
    
    let output = norm.forward(&input)?;
    let output_vec = output.to_vec2::<f32>()?;

    // Each row should be normalized independently
    for row_idx in 0..batch_size {
        let row = &output_vec[row_idx];
        let mean_sq: f32 = row.iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;
        let rms = mean_sq.sqrt();
        
        assert!(
            (rms - 1.0).abs() < 0.01,
            "Row {} RMS should be ~1.0, got {}", row_idx, rms
        );
    }
    
    Ok(())
}

#[test]
fn test_rms_norm_complete_validation() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Checkpoint 1: Complete RMSNorm Validation              ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let device = Device::Cpu;
    let hidden_size = 4096;
    let seq_len = 2;
    
    let weight = vec![1.0f32; hidden_size];
    let norm = RMSNorm::from_array(&weight, 1e-5, &device).unwrap();

    let input = generate_test_input(&device).unwrap();
    let output = norm.forward(&input).unwrap();

    println!("\n📊 Test Configuration:");
    println!("  Input shape: {:?}", input.shape().dims());
    println!("  Output shape: {:?}", output.shape().dims());
    println!("  Hidden size: {}", hidden_size);
    println!("  Sequence length: {}", seq_len);
    println!("  Epsilon: 1e-5");
    println!("  Weight: ones (standard initialization)");

    let output_vec = output.to_vec2::<f32>().unwrap();
    
    println!("\n📊 Output Analysis:");
    for row_idx in 0..seq_len {
        let row = &output_vec[row_idx];
        let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
        let mean_sq: f32 = row.iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;
        let rms = mean_sq.sqrt();
        let min = row.iter().copied().fold(f32::INFINITY, f32::min);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        println!("  Row {}: mean={:.6}, rms={:.6}, range=[{:.6}, {:.6}]", 
                 row_idx, mean, rms, min, max);
        
        // Validation checks
        assert!(mean.is_finite(), "Mean should be finite");
        assert!(rms.is_finite(), "RMS should be finite");
        assert!(min > -10.0 && max < 10.0, "Values should be in reasonable range");
        assert!((rms - 1.0).abs() < 0.01, "RMS should be ~1.0");
    }

    let sample: Vec<f32> = output_vec[0].iter().take(10).copied().collect();
    println!("\n📊 Output Sample (first 10 values):");
    println!("  {:?}", sample);

    println!("\n✅ Validation Checks:");
    println!("  ✅ Shape correct: {:?}", output.shape().dims());
    println!("  ✅ No NaN/Inf values");
    println!("  ✅ Values in reasonable range [-10, 10]");
    println!("  ✅ RMS ≈ 1.0 per row (normalized)");
    println!("  ✅ Deterministic across runs");

    println!("\n📝 Next Steps:");
    println!("  1. Checkpoint 1 PASSED ✅");
    println!("  2. Ready for Checkpoint 1B (RoPE)");
    println!("  3. Using Candle's optimized rms_norm function");
    println!("  4. CUDA acceleration available with --features cuda");
}
