//! Standalone test for LayerNorm (Checkpoint 1)
//!
//! Run with: cargo run --example test_layer_norm

use ndarray::{Array, Array1, Array2, Axis};

// Copy of LayerNorm implementation
pub struct LayerNorm {
    weight: Array1<f32>,
    bias: Array1<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Array1<f32>, bias: Array1<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let shape = x.shape();
        let _dim = shape[1];
        
        // 1. Compute mean across last dimension (axis=1)
        let mean = x.mean_axis(Axis(1)).unwrap();
        
        // 2. Center the input: x - mean
        let x_centered = x - &mean.insert_axis(Axis(1));
        
        // 3. Compute biased variance: mean((x - mean)^2)
        let variance = (&x_centered * &x_centered).mean_axis(Axis(1)).unwrap();
        
        // 4. Normalize: (x - mean) / sqrt(variance + eps)
        let std = (&variance + self.eps).mapv(f32::sqrt);
        let normalized = &x_centered / &std.insert_axis(Axis(1));
        
        // 5. Apply scale and bias: normalized * weight + bias
        let output = &normalized * &self.weight.view().insert_axis(Axis(0)) 
                   + &self.bias.view().insert_axis(Axis(0));
        
        output
    }
}

fn main() {
    println!("=== Checkpoint 1: LayerNorm Tests ===\n");

    // Test 1: Shape preservation
    println!("Test 1: Shape preservation");
    let dim = 1024;
    let weight = Array1::ones(dim);
    let bias = Array1::zeros(dim);
    let ln = LayerNorm::new(weight, bias, 1e-5);
    let input = Array2::zeros((2, dim));
    let output = ln.forward(&input);
    assert_eq!(output.shape(), input.shape());
    println!("✅ PASS: Output shape matches input shape\n");

    // Test 2: Mean and variance
    println!("Test 2: Mean and variance normalization");
    let dim = 4;
    let weight = Array1::ones(dim);
    let bias = Array1::zeros(dim);
    let ln = LayerNorm::new(weight, bias, 1e-5);
    let input = Array::from_shape_vec((1, dim), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = ln.forward(&input);
    
    let output_mean = output.mean_axis(Axis(1)).unwrap()[[0]];
    let output_centered = &output - output_mean;
    let output_var = (&output_centered * &output_centered).mean_axis(Axis(1)).unwrap()[[0]];
    
    println!("  Input: {:?}", input.row(0));
    println!("  Output: {:?}", output.row(0));
    println!("  Output mean: {:.6} (should be ~0)", output_mean);
    println!("  Output variance: {:.6} (should be ~1)", output_var);
    
    assert!((output_mean.abs()) < 1e-5, "Mean should be ~0, got {}", output_mean);
    assert!((output_var - 1.0).abs() < 1e-4, "Variance should be ~1, got {}", output_var);
    println!("✅ PASS: Mean ≈ 0, Variance ≈ 1\n");

    // Test 3: Scale and bias
    println!("Test 3: Scale and bias application");
    let dim = 4;
    let weight = Array1::from_vec(vec![2.0, 2.0, 2.0, 2.0]);
    let bias = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
    let ln = LayerNorm::new(weight, bias, 1e-5);
    let input = Array::from_shape_vec((1, dim), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = ln.forward(&input);
    
    let output_mean = output.mean_axis(Axis(1)).unwrap()[[0]];
    println!("  Input: {:?}", input.row(0));
    println!("  Output: {:?}", output.row(0));
    println!("  Output mean: {:.6} (should be ~1.0 due to bias)", output_mean);
    
    assert!((output_mean - 1.0).abs() < 1e-4, "Mean should be ~1.0, got {}", output_mean);
    println!("✅ PASS: Scale and bias applied correctly\n");

    // Test 4: Batch processing
    println!("Test 4: Batch processing");
    let dim = 4;
    let batch_size = 3;
    let weight = Array1::ones(dim);
    let bias = Array1::zeros(dim);
    let ln = LayerNorm::new(weight, bias, 1e-5);
    
    let input = Array2::from_shape_fn((batch_size, dim), |(i, j)| {
        (i * dim + j) as f32
    });
    let output = ln.forward(&input);
    
    println!("  Batch size: {}", batch_size);
    for i in 0..batch_size {
        let row = output.row(i);
        let row_mean = row.mean().unwrap();
        let row_var = row.mapv(|x| (x - row_mean).powi(2)).mean().unwrap();
        println!("  Row {}: mean={:.6}, var={:.6}", i, row_mean, row_var);
        
        assert!((row_mean.abs()) < 1e-5, "Row {} mean should be ~0", i);
        assert!((row_var - 1.0).abs() < 1e-4, "Row {} variance should be ~1", i);
    }
    println!("✅ PASS: Each batch normalized independently\n");

    println!("=== ✅ All Checkpoint 1 Tests PASSED ===");
    println!("\n✅ CHECKPOINT 1: LayerNorm implementation is CORRECT!");
    println!("   Ready to proceed to Checkpoint 2 (QKV Projection)");
}
