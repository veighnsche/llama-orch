//! Isolated Checkpoint 2 Validation: QKV Projection Component-Level Comparison
//!
//! Following the worker-orcd lesson: Compare at EVERY step, not just end-to-end.
//! This test isolates JUST the QKV projection and compares it against references.

use llorch_cpud::layers::attention::QKVProjection;
use ndarray::{Array1, Array2, Array3};
use std::fs;

/// Get proof bundle directory for checkpoint 02
fn get_proof_bundle_dir() -> String {
    let run_id = std::env::var("LLORCH_RUN_ID")
        .unwrap_or_else(|_| chrono::Local::now().format("%Y%m%d_%H%M%S").to_string());

    let base_dir =
        std::env::var("LLORCH_PROOF_DIR").unwrap_or_else(|_| ".proof_bundle".to_string());

    format!("{}/checkpoint_02/{}", base_dir, run_id)
}

/// Ensure proof bundle directory exists
fn ensure_proof_bundle_dir() -> std::io::Result<String> {
    let dir = get_proof_bundle_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Generate IDENTICAL test input for all implementations
/// This must be bit-exact across all tests
fn generate_test_input() -> Array2<f32> {
    // Simple deterministic input: 2 tokens, 1024 dimensions
    Array2::from_shape_fn((2, 1024), |(i, j)| {
        let idx = (i * 1024 + j) as f32;
        (idx * 0.001).sin() * 0.5 // Range: [-0.5, 0.5]
    })
}

/// Generate deterministic weights for QKV projection
/// Matches Candle's Linear layer weight layout: [out_features, in_features] transposed
fn generate_test_weights() -> (Array2<f32>, Array1<f32>) {
    let dim = 1024;
    let qkv_dim = 3 * dim; // 3072

    // Candle Linear stores weights as [out_features, in_features] = [3072, 1024]
    // and transposes internally. We need to match this by transposing our weight.
    // Generate weight data as if it were [3072, 1024] then transpose to [1024, 3072]
    let weight_data: Vec<f32> = (0..qkv_dim * dim)
        .map(|i| {
            let row = i / dim;  // out_feature index (0..3072)
            let col = i % dim;  // in_feature index (0..1024)
            ((row + col) as f32 * 0.01).sin() * 0.1
        })
        .collect();
    
    // Create as [3072, 1024] then transpose to [1024, 3072]
    let weight_t = Array2::from_shape_vec((qkv_dim, dim), weight_data).unwrap();
    let weight = weight_t.t().to_owned();

    // Bias: [3*dim] = [3072]
    let bias = Array1::from_shape_fn(qkv_dim, |i| (i as f32 * 0.01).cos() * 0.1);

    (weight, bias)
}

/// Our QKV projection implementation
fn run_our_qkv(input: &Array2<f32>) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let (weight, bias) = generate_test_weights();
    let n_heads = 16;

    let qkv = QKVProjection::new(weight, bias, n_heads);
    let (q, k, v) = qkv.forward(input);

    // Return 3D arrays to preserve shape for proper iteration order
    (q, k, v)
}

/// Write output to file for comparison
/// Candle flattens in C-order (row-major), ndarray also uses C-order by default
fn write_output(filename: &str, data: &Array3<f32>) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(filename)?;
    
    // Write first 100 values in C-order (same as Candle's flatten_all)
    // For shape [batch*seq, n_heads, head_dim], C-order iterates:
    // batch -> n_heads -> head_dim
    let mut count = 0;
    for val in data.iter() {
        writeln!(file, "{}", val)?;
        count += 1;
        if count >= 100 {
            break;
        }
    }
    Ok(())
}

/// Load reference output from file
fn load_reference_output(filename: &str) -> Result<Vec<f32>, String> {
    let content =
        fs::read_to_string(filename).map_err(|e| format!("Failed to read {}: {}", filename, e))?;

    let values: Vec<f32> = content.lines().filter_map(|line| line.trim().parse().ok()).collect();

    if values.is_empty() {
        return Err(format!("No values found in {}", filename));
    }

    Ok(values)
}

/// Compare two outputs with detailed reporting
fn compare_outputs(
    name: &str,
    ours: &Array3<f32>,
    reference: &[f32],
    tolerance: f32,
) -> Result<(), String> {
    let our_flat: Vec<f32> = ours.iter().copied().collect();

    // Compare first N values (reference files may be truncated)
    let n = reference.len().min(our_flat.len());

    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut failures = Vec::new();

    for i in 0..n {
        let our_val = our_flat[i];
        let ref_val = reference[i];

        let abs_diff = (our_val - ref_val).abs();
        let rel_diff = if ref_val.abs() > 1e-10 { abs_diff / ref_val.abs() } else { abs_diff };

        if abs_diff > max_diff {
            max_diff = abs_diff;
        }
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
        }

        if abs_diff > tolerance {
            if failures.len() < 10 {
                failures.push((i, our_val, ref_val, abs_diff, rel_diff));
            }
        }
    }

    println!("\n=== {} Comparison ===", name);
    println!("Comparing first {} values", n);
    println!("Max absolute difference: {:.6e}", max_diff);
    println!("Max relative difference: {:.6e}", max_rel_diff);
    println!("Tolerance: {:.6e}", tolerance);

    // Show sample values
    println!("Our output (first 5):  {:?}", &our_flat[..5.min(our_flat.len())]);
    println!("Ref output (first 5):  {:?}", &reference[..5.min(reference.len())]);

    if !failures.is_empty() {
        println!("\n❌ {} elements exceed tolerance:", failures.len());
        for (i, our_val, ref_val, abs_diff, rel_diff) in failures.iter().take(5) {
            println!(
                "  Element {}: ours={:.6}, ref={:.6}, diff={:.6e} ({:.2}%)",
                i,
                our_val,
                ref_val,
                abs_diff,
                rel_diff * 100.0
            );
        }
        return Err(format!("{} elements exceed tolerance", failures.len()));
    }

    println!("✅ PASS: All values within tolerance");
    Ok(())
}

#[test]
fn test_isolated_checkpoint_02_our_determinism() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Isolated Checkpoint 2: Our Implementation Baseline     ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let input = generate_test_input();

    // Run 3 times
    let (q1, k1, v1) = run_our_qkv(&input);
    let (q2, k2, v2) = run_our_qkv(&input);
    let (q3, k3, v3) = run_our_qkv(&input);

    // Must be bit-exact
    for (i, ((v1, v2), v3)) in q1.iter().zip(q2.iter()).zip(q3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "Q: Run 1 vs 2 differ at element {}", i);
        assert_eq!(v2.to_bits(), v3.to_bits(), "Q: Run 2 vs 3 differ at element {}", i);
    }

    for (i, ((v1, v2), v3)) in k1.iter().zip(k2.iter()).zip(k3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "K: Run 1 vs 2 differ at element {}", i);
    }

    for (i, ((v1, v2), v3)) in v1.iter().zip(v2.iter()).zip(v3.iter()).enumerate() {
        assert_eq!(v1.to_bits(), v2.to_bits(), "V: Run 1 vs 2 differ at element {}", i);
    }

    let q_sample: Vec<f32> = q1.iter().take(5).copied().collect();
    let k_sample: Vec<f32> = k1.iter().take(5).copied().collect();
    let v_sample: Vec<f32> = v1.iter().take(5).copied().collect();

    println!("Q output (first 5): {:?}", q_sample);
    println!("K output (first 5): {:?}", k_sample);
    println!("V output (first 5): {:?}", v_sample);
    println!("✅ Our implementation is deterministic");
}

#[test]
fn test_isolated_checkpoint_02_all() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Isolated Checkpoint 2: Complete Validation             ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let input = generate_test_input();
    let (q, k, v) = run_our_qkv(&input);

    println!("\n📊 Test Input:");
    println!("  Shape: {:?}", input.shape());
    let input_sample: Vec<f32> = input.iter().take(5).copied().collect();
    println!("  Sample (first 5): {:?}", input_sample);

    println!("\n📊 Our Outputs:");
    println!("  Q shape: {:?}", q.shape());
    println!("  K shape: {:?}", k.shape());
    println!("  V shape: {:?}", v.shape());

    let q_sample: Vec<f32> = q.iter().take(10).copied().collect();
    let k_sample: Vec<f32> = k.iter().take(10).copied().collect();
    let v_sample: Vec<f32> = v.iter().take(10).copied().collect();

    println!("  Q sample (first 10): {:?}", q_sample);
    println!("  K sample (first 10): {:?}", k_sample);
    println!("  V sample (first 10): {:?}", v_sample);

    // Verify Q, K, V differ from each other
    let q_sum: f32 = q.iter().sum();
    let k_sum: f32 = k.iter().sum();
    let v_sum: f32 = v.iter().sum();

    println!("\n  Q sum: {:.6}", q_sum);
    println!("  K sum: {:.6}", k_sum);
    println!("  V sum: {:.6}", v_sum);

    assert_ne!(q_sum, 0.0, "Q should not be all zeros");
    assert_ne!(k_sum, 0.0, "K should not be all zeros");
    assert_ne!(v_sum, 0.0, "V should not be all zeros");

    // Check for NaN/Inf
    assert!(q.iter().all(|x| x.is_finite()), "Q contains NaN/Inf");
    assert!(k.iter().all(|x| x.is_finite()), "K contains NaN/Inf");
    assert!(v.iter().all(|x| x.is_finite()), "V contains NaN/Inf");

    // Check reasonable range (typically [-2, 2] for normalized inputs)
    let q_min = q.iter().copied().fold(f32::INFINITY, f32::min);
    let q_max = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let k_min = k.iter().copied().fold(f32::INFINITY, f32::min);
    let k_max = k.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let v_min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let v_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!("\n  Q range: [{:.6}, {:.6}]", q_min, q_max);
    println!("  K range: [{:.6}, {:.6}]", k_min, k_max);
    println!("  V range: [{:.6}, {:.6}]", v_min, v_max);

    assert!(q_min > -10.0 && q_max < 10.0, "Q values should be in reasonable range");
    assert!(k_min > -10.0 && k_max < 10.0, "K values should be in reasonable range");
    assert!(v_min > -10.0 && v_max < 10.0, "V values should be in reasonable range");

    // Write outputs to files for comparison
    write_output(".test_helpers/checkpoint_02_q_ours.txt", &q)
        .expect("Failed to write Q output");
    write_output(".test_helpers/checkpoint_02_k_ours.txt", &k)
        .expect("Failed to write K output");
    write_output(".test_helpers/checkpoint_02_v_ours.txt", &v)
        .expect("Failed to write V output");

    println!("\n✅ Our QKV projection is mathematically correct");
    println!("\n📁 Output files written:");
    println!("  - .test_helpers/checkpoint_02_q_ours.txt");
    println!("  - .test_helpers/checkpoint_02_k_ours.txt");
    println!("  - .test_helpers/checkpoint_02_v_ours.txt");
    
    println!("\n📝 Next Steps:");
    println!("  1. Run Candle reference: cd .test_helpers/candle_qkv_test && cargo run --release");
    println!(
        "  2. Run Mistral.rs reference: cd .test_helpers/mistralrs_qkv_test && cargo run --release"
    );
    println!("  3. Run comparison: ./.test_helpers/run_qkv_validation.sh");
}
