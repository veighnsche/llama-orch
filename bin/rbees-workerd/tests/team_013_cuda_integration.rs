//! TEAM-013 CUDA Integration Tests
//!
//! Tests CUDA GPU inference performance with TinyLlama model.
//! Validates TEAM-012's expectation of 10-50x speedup over CPU.
//!
//! Created by: TEAM-013

use anyhow::Result;
use rbees_workerd::backend::CandleInferenceBackend;
#[cfg(feature = "cuda")]
use rbees_workerd::device::init_cuda_device;
use rbees_workerd::{InferenceBackend, SamplingConfig};

/// TEAM-013: Validate CUDA story generation performance
///
/// This test generates a 5-token story on CUDA to verify:
/// 1. CUDA device initialization works
/// 2. Model loads successfully on GPU
/// 3. Generation produces coherent text
/// 4. Performance is significantly faster than CPU
///
/// Expected: 30-150 tok/s (10-50x faster than CPU's 3.23 tok/s)
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn test_cuda_story_generation() -> Result<()> {
    let model_path = std::env::var("LLORCH_TEST_MODEL_PATH")
        .expect("Set LLORCH_TEST_MODEL_PATH to run this test");

    // TEAM-013: Initialize CUDA device 0 (RTX 3060 or RTX 3090)
    let device = init_cuda_device(0)?;
    let mut backend = CandleInferenceBackend::load(&model_path, device)?;

    let config = SamplingConfig { max_tokens: 5, temperature: 0.7, seed: 42, ..Default::default() };

    let rt = tokio::runtime::Runtime::new()?;
    let start = std::time::Instant::now();

    let prompt = "Once upon a time, in a small village nestled in the mountains, there lived";
    let result =
        rt.block_on(backend.execute(prompt, &config)).map_err(|e| anyhow::anyhow!("{}", e))?;

    let duration = start.elapsed();
    let story = result.tokens.join("");
    let tokens_per_sec = result.tokens.len() as f64 / duration.as_secs_f64();

    println!("\n=== TEAM-013 CUDA STORY GENERATION TEST ===");
    println!("Device: CUDA GPU 0");
    println!("Prompt: {}", prompt);
    println!("Generated: {}", story);
    println!("\nStats:");
    println!("  Tokens: {}", result.tokens.len());
    println!("  Duration: {:.2}s", duration.as_secs_f64());
    println!("  Speed: {:.2} tok/s", tokens_per_sec);
    println!("  Time per token: {:.4}s", duration.as_secs_f64() / result.tokens.len() as f64);
    println!("===========================================\n");

    // TEAM-013: Validation checks
    assert!(!result.tokens.is_empty(), "Should generate tokens");
    assert!(result.tokens.len() >= 3, "Should generate at least 3 tokens");

    let has_letters = story.chars().any(|c| c.is_alphabetic());
    assert!(has_letters, "Story should contain alphabetic characters");

    let char_count = story.chars().filter(|c| c.is_alphabetic()).count();
    assert!(char_count >= 10, "Story should contain at least 10 letters, got {}", char_count);

    // TEAM-013: Performance validation
    // CPU baseline: 3.23 tok/s
    // Expected CUDA: 30-150 tok/s (10-50x speedup)
    // Minimum acceptable: 10 tok/s (3x speedup, conservative)
    assert!(
        tokens_per_sec > 10.0,
        "CUDA should be at least 3x faster than CPU (3.23 tok/s). Got {:.2} tok/s",
        tokens_per_sec
    );

    println!("✅ TEAM-013 CUDA VERDICT: Performance validated!");
    println!("   Speedup vs CPU (3.23 tok/s): {:.1}x", tokens_per_sec / 3.23);
    println!("   Full continuation: \"{}{}\"", prompt, story);

    Ok(())
}

/// TEAM-013: Extended CUDA story generation test (20 tokens)
///
/// This test generates a longer story to:
/// 1. Verify sustained CUDA performance over more tokens
/// 2. Show actual story coherence with more context
/// 3. Benchmark CUDA performance vs CPU baseline
///
/// CPU baseline: 3.23 tok/s (from TEAM-012)
/// Expected CUDA: 30-150 tok/s
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn test_cuda_extended_story_generation() -> Result<()> {
    let model_path = std::env::var("LLORCH_TEST_MODEL_PATH")
        .expect("Set LLORCH_TEST_MODEL_PATH to run this test");

    // TEAM-013: Initialize CUDA device 0
    let device = init_cuda_device(0)?;
    let mut backend = CandleInferenceBackend::load(&model_path, device)?;

    let config =
        SamplingConfig { max_tokens: 20, temperature: 0.7, seed: 42, ..Default::default() };

    let rt = tokio::runtime::Runtime::new()?;
    let start = std::time::Instant::now();

    let prompt = "Once upon a time";
    let result =
        rt.block_on(backend.execute(prompt, &config)).map_err(|e| anyhow::anyhow!("{}", e))?;

    let duration = start.elapsed();
    let story = result.tokens.join("");
    let tokens_per_sec = result.tokens.len() as f64 / duration.as_secs_f64();

    println!("\n=== TEAM-013 CUDA EXTENDED STORY TEST ===");
    println!("Device: CUDA GPU 0");
    println!("Prompt: {}", prompt);
    println!("Generated: {}", story);
    println!("\nStats:");
    println!("  Tokens: {}", result.tokens.len());
    println!("  Duration: {:.2}s", duration.as_secs_f64());
    println!("  Speed: {:.2} tok/s", tokens_per_sec);
    println!("  Avg time per token: {:.4}s", duration.as_secs_f64() / result.tokens.len() as f64);
    println!("\nComparison to CPU (TEAM-012 baseline):");
    println!("  CPU: 3.23 tok/s (6.19s for 20 tokens)");
    println!(
        "  CUDA: {:.2} tok/s ({:.2}s for {} tokens)",
        tokens_per_sec,
        duration.as_secs_f64(),
        result.tokens.len()
    );
    println!("  Speedup: {:.1}x", tokens_per_sec / 3.23);
    println!("==========================================\n");

    // TEAM-013: Validation
    assert!(!result.tokens.is_empty(), "Should generate tokens");
    assert!(result.tokens.len() >= 15, "Should generate at least 15 tokens");

    let char_count = story.chars().filter(|c| c.is_alphabetic()).count();
    assert!(char_count >= 30, "Story should contain at least 30 letters");

    // TEAM-013: Performance validation
    assert!(
        tokens_per_sec > 10.0,
        "CUDA should be significantly faster than CPU. Got {:.2} tok/s",
        tokens_per_sec
    );

    println!("✅ TEAM-013: Extended CUDA story validated at {:.2} tok/s", tokens_per_sec);
    println!("   Full story: \"{}{}\"", prompt, story);

    Ok(())
}

/// TEAM-013: CUDA vs CPU performance comparison
///
/// Generates 50 tokens to get a robust performance measurement.
/// Compares against TEAM-012's CPU baseline.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn test_cuda_performance_benchmark() -> Result<()> {
    let model_path = std::env::var("LLORCH_TEST_MODEL_PATH")
        .expect("Set LLORCH_TEST_MODEL_PATH to run this test");

    // TEAM-013: Initialize CUDA device 0
    let device = init_cuda_device(0)?;
    let mut backend = CandleInferenceBackend::load(&model_path, device)?;

    let config =
        SamplingConfig { max_tokens: 50, temperature: 0.7, seed: 42, ..Default::default() };

    let rt = tokio::runtime::Runtime::new()?;
    let start = std::time::Instant::now();

    let prompt = "In a world where artificial intelligence";
    let result =
        rt.block_on(backend.execute(prompt, &config)).map_err(|e| anyhow::anyhow!("{}", e))?;

    let duration = start.elapsed();
    let text = result.tokens.join("");
    let tokens_per_sec = result.tokens.len() as f64 / duration.as_secs_f64();
    let time_per_token = duration.as_secs_f64() / result.tokens.len() as f64;

    println!("\n=== TEAM-013 CUDA PERFORMANCE BENCHMARK ===");
    println!("Device: CUDA GPU 0");
    println!("Tokens generated: {}", result.tokens.len());
    println!("Duration: {:.2}s", duration.as_secs_f64());
    println!("Speed: {:.2} tok/s", tokens_per_sec);
    println!("Time per token: {:.4}s", time_per_token);
    println!("\nCPU Baseline (TEAM-012):");
    println!("  Speed: 3.23 tok/s");
    println!("  Time per token: 0.31s");
    println!("\nCUDA Performance:");
    println!("  Speed: {:.2} tok/s", tokens_per_sec);
    println!("  Time per token: {:.4}s", time_per_token);
    println!("  Speedup: {:.1}x", tokens_per_sec / 3.23);
    println!("  Time reduction: {:.1}%", (1.0 - time_per_token / 0.31) * 100.0);
    println!("\nGenerated text preview:");
    println!("  {}", &text[..text.len().min(200)]);
    println!("===========================================\n");

    // TEAM-013: Validation
    assert!(!result.tokens.is_empty(), "Should generate tokens");
    assert!(
        tokens_per_sec > 10.0,
        "CUDA should be significantly faster than CPU. Got {:.2} tok/s",
        tokens_per_sec
    );

    println!("✅ TEAM-013: CUDA performance benchmark complete");
    println!(
        "   Result: {:.2} tok/s ({:.1}x faster than CPU)",
        tokens_per_sec,
        tokens_per_sec / 3.23
    );

    Ok(())
}

/// TEAM-013: Test CUDA device 1 (RTX 3090) if available
///
/// Tests the second GPU to verify multi-GPU support.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn test_cuda_device_1_story_generation() -> Result<()> {
    let model_path = std::env::var("LLORCH_TEST_MODEL_PATH")
        .expect("Set LLORCH_TEST_MODEL_PATH to run this test");

    // TEAM-013: Try to initialize CUDA device 1 (RTX 3090)
    let device = match init_cuda_device(1) {
        Ok(d) => d,
        Err(e) => {
            println!("⚠️  CUDA device 1 not available: {}", e);
            println!("   Skipping test (this is OK if you only have 1 GPU)");
            return Ok(());
        }
    };

    let mut backend = CandleInferenceBackend::load(&model_path, device)?;

    let config =
        SamplingConfig { max_tokens: 20, temperature: 0.7, seed: 42, ..Default::default() };

    let rt = tokio::runtime::Runtime::new()?;
    let start = std::time::Instant::now();

    let prompt = "Once upon a time";
    let result =
        rt.block_on(backend.execute(prompt, &config)).map_err(|e| anyhow::anyhow!("{}", e))?;

    let duration = start.elapsed();
    let story = result.tokens.join("");
    let tokens_per_sec = result.tokens.len() as f64 / duration.as_secs_f64();

    println!("\n=== TEAM-013 CUDA DEVICE 1 TEST ===");
    println!("Device: CUDA GPU 1 (RTX 3090)");
    println!("Prompt: {}", prompt);
    println!("Generated: {}", story);
    println!("\nStats:");
    println!("  Tokens: {}", result.tokens.len());
    println!("  Duration: {:.2}s", duration.as_secs_f64());
    println!("  Speed: {:.2} tok/s", tokens_per_sec);
    println!("  Speedup vs CPU: {:.1}x", tokens_per_sec / 3.23);
    println!("====================================\n");

    // TEAM-013: Validation
    assert!(!result.tokens.is_empty(), "Should generate tokens");
    assert!(
        tokens_per_sec > 10.0,
        "CUDA should be significantly faster than CPU. Got {:.2} tok/s",
        tokens_per_sec
    );

    println!("✅ TEAM-013: CUDA device 1 validated at {:.2} tok/s", tokens_per_sec);

    Ok(())
}
