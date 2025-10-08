//! Candle QKV Projection Reference Implementation
//!
//! This test validates our QKV projection against Candle's Linear layer.
//! We use the same input, weights, and bias to ensure bit-exact comparison.

use candle_core::{Device, IndexOp, Module, Tensor};
use candle_nn::Linear;
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    
    // Configuration: GPT-2 Medium
    let dim = 1024;
    let n_heads = 16;
    let head_dim = dim / n_heads; // 64
    let qkv_dim = 3 * dim; // 3072
    
    println!("=== CANDLE QKV PROJECTION TEST ===");
    println!("Config: dim={}, n_heads={}, head_dim={}", dim, n_heads, head_dim);
    
    // 1. Generate deterministic input [2, 1024]
    let input_data: Vec<f32> = (0..2 * dim)
        .map(|i| ((i as f32) * 0.001).sin() * 0.5)
        .collect();
    let input = Tensor::from_vec(input_data, (2, dim), &device)?;
    
    println!("\nInput shape: {:?}", input.shape());
    let input_sample = input.flatten_all()?.to_vec1::<f32>()?;
    println!("Input sample (first 5): {:?}", &input_sample[..5]);
    
    // 2. Create deterministic weights and bias
    // Weight: Candle Linear expects [out_features, in_features] = [3072, 1024]
    // Generate data in row-major order for [3072, 1024]
    let weight_data: Vec<f32> = (0..qkv_dim * dim)
        .map(|i| {
            let row = i / dim;  // out_feature index (0..3072)
            let col = i % dim;  // in_feature index (0..1024)
            ((row + col) as f32 * 0.01).sin() * 0.1
        })
        .collect();
    let weight = Tensor::from_vec(weight_data, (qkv_dim, dim), &device)?;
    
    // Bias: [3*dim] = [3072]
    let bias_data: Vec<f32> = (0..qkv_dim)
        .map(|i| ((i as f32) * 0.01).cos() * 0.1)
        .collect();
    let bias = Tensor::from_vec(bias_data, qkv_dim, &device)?;
    
    println!("\nWeight shape: {:?}", weight.shape());
    println!("Bias shape: {:?}", bias.shape());
    
    // 3. Create Linear layer (Candle's equivalent of c_attn)
    let linear = Linear::new(weight, Some(bias));
    
    // 4. Forward pass: [2, 1024] @ [1024, 3072] + [3072] → [2, 3072]
    let qkv_combined = linear.forward(&input)?;
    
    println!("\nQKV combined shape: {:?}", qkv_combined.shape());
    let qkv_sample = qkv_combined.flatten_all()?.to_vec1::<f32>()?;
    println!("QKV combined sample (first 5): {:?}", &qkv_sample[..5]);
    
    // 5. Reshape to [2, 3, 16, 64]
    let qkv_reshaped = qkv_combined.reshape((2, 3, n_heads, head_dim))?;
    println!("\nQKV reshaped: {:?}", qkv_reshaped.shape());
    
    // 6. Split into Q, K, V
    // Q = qkv[:, 0, :, :]  → [2, 16, 64]
    // K = qkv[:, 1, :, :]  → [2, 16, 64]
    // V = qkv[:, 2, :, :]  → [2, 16, 64]
    let q = qkv_reshaped.i((.., 0, .., ..))?;
    let k = qkv_reshaped.i((.., 1, .., ..))?;
    let v = qkv_reshaped.i((.., 2, .., ..))?;
    
    println!("\nQ shape: {:?}", q.shape());
    println!("K shape: {:?}", k.shape());
    println!("V shape: {:?}", v.shape());
    
    // 7. Get sample values
    let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
    let k_vec = k.flatten_all()?.to_vec1::<f32>()?;
    let v_vec = v.flatten_all()?.to_vec1::<f32>()?;
    
    println!("\nQ sample (first 10): {:?}", &q_vec[..10]);
    println!("K sample (first 10): {:?}", &k_vec[..10]);
    println!("V sample (first 10): {:?}", &v_vec[..10]);
    
    // 8. Write outputs to files for comparison
    write_output("checkpoint_02_q_candle.txt", &q_vec)?;
    write_output("checkpoint_02_k_candle.txt", &k_vec)?;
    write_output("checkpoint_02_v_candle.txt", &v_vec)?;
    
    println!("\n✅ Candle QKV projection complete");
    println!("Output files written to:");
    println!("  - checkpoint_02_q_candle.txt");
    println!("  - checkpoint_02_k_candle.txt");
    println!("  - checkpoint_02_v_candle.txt");
    
    Ok(())
}

fn write_output(filename: &str, data: &[f32]) -> anyhow::Result<()> {
    let mut file = std::fs::File::create(filename)?;
    for (i, &val) in data.iter().enumerate() {
        writeln!(file, "{}", val)?;
        if i >= 99 {
            break; // Only write first 100 values
        }
    }
    Ok(())
}
