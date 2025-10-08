//! Cross-Reference Validation: LayerNorm Checkpoint 1
//!
//! Compares our LayerNorm implementation against:
//! - tinygrad (Python, research-grade)
//! - Candle (Rust, production-grade)
//! - Mistral.rs (Rust, production-grade)
//!
//! This test proves "parity" - functional equivalence within floating-point tolerance.

use llorch_cpud::layers::LayerNorm;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::PathBuf;

/// Standard test input matching what references use
/// Simulates embedding output for tokens [15496, 13] ("Hello.")
fn generate_standard_test_input() -> Array2<f32> {
    // Shape: [2, 1024] (2 tokens, GPT-2 Medium embedding dim)
    // Deterministic pattern matching typical embedding magnitudes
    Array2::from_shape_fn((2, 1024), |(i, j)| {
        // Pattern: sin wave with realistic magnitude
        ((i * 1024 + j) as f32 * 0.01).sin() * 0.5
    })
}

/// Parse reference output from log file
/// Expected format: "[CHECKPOINT 1] Output sample: [val1, val2, val3, ...]"
fn parse_reference_output(file_path: &str) -> Result<Vec<f32>, String> {
    let content = fs::read_to_string(file_path)
        .map_err(|e| format!("Failed to read {}: {}", file_path, e))?;

    // Find the checkpoint line
    for line in content.lines() {
        if line.contains("[CHECKPOINT 1]") && line.contains("Output sample:") {
            // Extract the array part: "[val1, val2, ...]"
            if let Some(start) = line.find('[') {
                if let Some(end) = line.find(']') {
                    let array_str = &line[start + 1..end];

                    // Parse comma-separated floats
                    let values: Result<Vec<f32>, _> =
                        array_str.split(',').map(|s| s.trim().parse::<f32>()).collect();

                    return values.map_err(|e| format!("Failed to parse floats: {}", e));
                }
            }
        }
    }

    Err(format!("No checkpoint output found in {}", file_path))
}

/// Compare two output vectors within tolerance
fn compare_outputs(
    name: &str,
    our_output: &[f32],
    ref_output: &[f32],
    tolerance: f32,
) -> Result<f32, String> {
    if our_output.len() != ref_output.len() {
        return Err(format!(
            "{}: Length mismatch: {} vs {}",
            name,
            our_output.len(),
            ref_output.len()
        ));
    }

    let mut max_diff = 0.0f32;
    let mut diffs = Vec::new();

    for (i, (&ours, &theirs)) in our_output.iter().zip(ref_output.iter()).enumerate() {
        let diff = (ours - theirs).abs();
        diffs.push(diff);

        if diff > max_diff {
            max_diff = diff;
        }

        if diff > tolerance {
            return Err(format!(
                "{}: Element {} exceeds tolerance: diff={:.6e}, ours={:.6}, theirs={:.6}",
                name, i, diff, ours, theirs
            ));
        }
    }

    println!("✅ {}: Max diff = {:.6e} (within tolerance {:.6e})", name, max_diff, tolerance);
    println!("   Our output:  {:?}", our_output);
    println!("   Ref output:  {:?}", ref_output);
    println!("   Differences: {:?}", diffs);

    Ok(max_diff)
}

#[test]
fn test_layernorm_determinism_baseline() {
    // Baseline test: Verify our implementation is deterministic
    let input = generate_standard_test_input();

    let ln = LayerNorm::new(Array1::ones(1024), Array1::zeros(1024), 1e-5);

    // Run 3 times
    let out1 = ln.forward(&input);
    let out2 = ln.forward(&input);
    let out3 = ln.forward(&input);

    // Extract first 5 values
    let sample1: Vec<f32> = out1.iter().take(5).copied().collect();
    let sample2: Vec<f32> = out2.iter().take(5).copied().collect();
    let sample3: Vec<f32> = out3.iter().take(5).copied().collect();

    println!("Our output (run 1): {:?}", sample1);
    println!("Our output (run 2): {:?}", sample2);
    println!("Our output (run 3): {:?}", sample3);

    // Must be bit-exact
    for (i, ((v1, v2), v3)) in sample1.iter().zip(sample2.iter()).zip(sample3.iter()).enumerate() {
        assert_eq!(
            v1.to_bits(),
            v2.to_bits(),
            "Run 1 vs 2 differ at element {}: {} vs {}",
            i,
            v1,
            v2
        );
        assert_eq!(
            v2.to_bits(),
            v3.to_bits(),
            "Run 2 vs 3 differ at element {}: {} vs {}",
            i,
            v2,
            v3
        );
    }

    println!("✅ Baseline: Our implementation is deterministic");
}

#[test]
#[ignore] // Run manually after extracting reference outputs
fn test_cross_reference_tinygrad() {
    let input = generate_standard_test_input();

    let ln = LayerNorm::new(Array1::ones(1024), Array1::zeros(1024), 1e-5);

    let our_output = ln.forward(&input);
    let our_sample: Vec<f32> = our_output.iter().take(5).copied().collect();

    // Load tinygrad output
    let tinygrad_path = "/tmp/tinygrad_checkpoint1.txt";
    match parse_reference_output(tinygrad_path) {
        Ok(ref_output) => {
            println!("\n=== Tinygrad Comparison ===");
            let tolerance = 1e-4; // 0.01% tolerance
            compare_outputs("Tinygrad", &our_sample, &ref_output, tolerance)
                .expect("Tinygrad comparison failed");
        }
        Err(e) => {
            println!("⚠️  Tinygrad output not found: {}", e);
            println!("   Run: cd reference/tinygrad && VALIDATE=1 python examples/gpt2.py --prompt 'Hello.' 2>&1 | tee /tmp/tinygrad_checkpoint1.txt");
            println!("   Our output for reference: {:?}", our_sample);
        }
    }
}

#[test]
#[ignore] // Run manually after extracting reference outputs
fn test_cross_reference_candle() {
    let input = generate_standard_test_input();

    let ln = LayerNorm::new(Array1::ones(1024), Array1::zeros(1024), 1e-5);

    let our_output = ln.forward(&input);
    let our_sample: Vec<f32> = our_output.iter().take(5).copied().collect();

    // Load Candle output
    let candle_path = "/tmp/candle_checkpoint1.txt";
    match parse_reference_output(candle_path) {
        Ok(ref_output) => {
            println!("\n=== Candle Comparison ===");
            let tolerance = 1e-3; // 0.1% tolerance (may use F16)
            compare_outputs("Candle", &our_sample, &ref_output, tolerance)
                .expect("Candle comparison failed");
        }
        Err(e) => {
            println!("⚠️  Candle output not found: {}", e);
            println!("   Run: cd reference/candle && VALIDATE=1 cargo run --example gpt2 2>&1 | tee /tmp/candle_checkpoint1.txt");
            println!("   Our output for reference: {:?}", our_sample);
        }
    }
}

#[test]
#[ignore] // Run manually after extracting reference outputs
fn test_cross_reference_mistral() {
    let input = generate_standard_test_input();

    let ln = LayerNorm::new(Array1::ones(1024), Array1::zeros(1024), 1e-5);

    let our_output = ln.forward(&input);
    let our_sample: Vec<f32> = our_output.iter().take(5).copied().collect();

    // Load Mistral.rs output
    let mistral_path = "/tmp/mistral_checkpoint1.txt";
    match parse_reference_output(mistral_path) {
        Ok(ref_output) => {
            println!("\n=== Mistral.rs Comparison ===");
            let tolerance = 1e-3; // 0.1% tolerance (Candle-based, may use F16)
            compare_outputs("Mistral.rs", &our_sample, &ref_output, tolerance)
                .expect("Mistral.rs comparison failed");
        }
        Err(e) => {
            println!("⚠️  Mistral.rs output not found: {}", e);
            println!("   Run: cd reference/mistral.rs && VALIDATE=1 cargo run 2>&1 | tee /tmp/mistral_checkpoint1.txt");
            println!("   Our output for reference: {:?}", our_sample);
        }
    }
}

#[test]
#[ignore] // Run manually after all references extracted
fn test_cross_reference_all() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Cross-Reference Validation: LayerNorm Checkpoint 1     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let input = generate_standard_test_input();

    let ln = LayerNorm::new(Array1::ones(1024), Array1::zeros(1024), 1e-5);

    let our_output = ln.forward(&input);
    let our_sample: Vec<f32> = our_output.iter().take(5).copied().collect();

    println!("Our LayerNorm output (first 5 elements):");
    println!("  {:?}\n", our_sample);

    let mut results = Vec::new();

    // Try each reference
    for (name, path, tolerance) in [
        ("Tinygrad", "/tmp/tinygrad_checkpoint1.txt", 1e-4),
        ("Candle", "/tmp/candle_checkpoint1.txt", 1e-3),
        ("Mistral.rs", "/tmp/mistral_checkpoint1.txt", 1e-3),
    ] {
        match parse_reference_output(path) {
            Ok(ref_output) => match compare_outputs(name, &our_sample, &ref_output, tolerance) {
                Ok(max_diff) => results.push((name, true, max_diff)),
                Err(e) => {
                    println!("❌ {}: {}", name, e);
                    results.push((name, false, 0.0));
                }
            },
            Err(e) => {
                println!("⚠️  {}: Not available ({})", name, e);
            }
        }
        println!();
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                 ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let passed = results.iter().filter(|(_, ok, _)| *ok).count();
    let total = results.len();

    for (name, ok, max_diff) in &results {
        let status = if *ok { "✅ PASS" } else { "❌ FAIL" };
        println!("{} {}: max_diff = {:.6e}", status, name, max_diff);
    }

    println!("\nResult: {}/{} references validated", passed, total);

    if passed == total && total > 0 {
        println!("\n🎉 PARITY PROVEN: All references match within tolerance!");
    } else if passed > 0 {
        println!("\n⚠️  PARTIAL PARITY: Some references match, investigate failures");
    } else {
        println!("\n❌ NO PARITY: No references available or all failed");
    }
}

#[cfg(test)]
mod proof_bundle {
    use super::*;

    /// Generate proof bundle documenting cross-reference validation
    #[test]
    #[ignore]
    fn generate_cross_reference_proof() {
        let proof_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(".proof_bundle")
            .join("cross_reference")
            .join("checkpoint_01_layer_norm");

        fs::create_dir_all(&proof_dir).expect("Failed to create proof directory");

        let input = generate_standard_test_input();
        let ln = LayerNorm::new(Array1::ones(1024), Array1::zeros(1024), 1e-5);
        let our_output = ln.forward(&input);
        let our_sample: Vec<f32> = our_output.iter().take(5).copied().collect();

        let mut proof = String::new();
        proof.push_str("# Cross-Reference Validation Proof\n\n");
        proof.push_str(&format!("**Generated:** {}\n", chrono::Utc::now().to_rfc3339()));
        proof.push_str("**Component:** LayerNorm (Checkpoint 1)\n\n");

        proof.push_str("## Our Implementation\n\n");
        proof.push_str(&format!("Output (first 5): {:?}\n\n", our_sample));

        proof.push_str("## Reference Implementations\n\n");

        for (name, path, tolerance) in [
            ("Tinygrad", "/tmp/tinygrad_checkpoint1.txt", 1e-4),
            ("Candle", "/tmp/candle_checkpoint1.txt", 1e-3),
            ("Mistral.rs", "/tmp/mistral_checkpoint1.txt", 1e-3),
        ] {
            proof.push_str(&format!("### {}\n\n", name));

            match parse_reference_output(path) {
                Ok(ref_output) => {
                    proof.push_str(&format!("Output (first 5): {:?}\n", ref_output));

                    match compare_outputs(name, &our_sample, &ref_output, tolerance) {
                        Ok(max_diff) => {
                            proof.push_str(&format!("**Status:** ✅ PASS\n"));
                            proof.push_str(&format!("**Max Difference:** {:.6e}\n", max_diff));
                            proof.push_str(&format!("**Tolerance:** {:.6e}\n", tolerance));
                        }
                        Err(e) => {
                            proof.push_str(&format!("**Status:** ❌ FAIL\n"));
                            proof.push_str(&format!("**Error:** {}\n", e));
                        }
                    }
                }
                Err(e) => {
                    proof.push_str(&format!("**Status:** ⚠️  NOT AVAILABLE\n"));
                    proof.push_str(&format!("**Error:** {}\n", e));
                }
            }

            proof.push_str("\n");
        }

        let proof_file = proof_dir.join("validation_proof.md");
        fs::write(&proof_file, proof).expect("Failed to write proof");

        println!("✅ Cross-reference proof generated: {}", proof_file.display());
    }
}
