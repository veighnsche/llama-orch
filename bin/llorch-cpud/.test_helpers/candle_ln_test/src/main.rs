use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Module};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Generate IDENTICAL input as Rust test
    // Pattern: (i * 1024 + j) * 0.001).sin() * 0.5
    let mut input_data = vec![0.0f32; 2 * 1024];
    for i in 0..2 {
        for j in 0..1024 {
            let idx = (i * 1024 + j) as f32;
            input_data[i * 1024 + j] = (idx * 0.001).sin() * 0.5;
        }
    }

    eprintln!("Input shape: [2, 1024]");
    eprintln!("Input sample (first 5): {:?}", &input_data[..5]);

    let input = Tensor::from_vec(input_data, (2, 1024), &device)?;

    // LayerNorm with same params as llorch-cpud
    // weight=ones, bias=zeros, eps=1e-5
    let weight = Tensor::ones((1024,), DType::F32, &device)?;
    let bias = Tensor::zeros((1024,), DType::F32, &device)?;
    let ln = LayerNorm::new(weight, bias, 1e-5);

    // Run forward
    let output = ln.forward(&input)?;
    let output_vec = output.flatten_all()?.to_vec1::<f32>()?;

    eprintln!("\nOutput shape: [2, 1024]");
    eprintln!("Output sample (first 10): {:?}", &output_vec[..10]);

    // Print for comparison
    println!("\n=== CANDLE LAYERNORM OUTPUT ===");
    println!("Shape: [2, 1024]");
    println!("First 10: {:?}", &output_vec[..10]);

    // Calculate stats
    let mean: f32 = output_vec.iter().sum::<f32>() / output_vec.len() as f32;
    let variance: f32 = output_vec
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>()
        / output_vec.len() as f32;
    let std = variance.sqrt();
    let min = output_vec.iter().copied().fold(f32::INFINITY, f32::min);
    let max = output_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!("Mean: {:.6}, Std: {:.6}", mean, std);
    println!("Min: {:.6}, Max: {:.6}", min, max);

    // Save to file for comparison
    let output_str = output_vec
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write("/tmp/llorch_test_output_candle.txt", output_str)?;

    eprintln!("\n✅ SUCCESS: Output saved to /tmp/llorch_test_output_candle.txt");

    Ok(())
}
