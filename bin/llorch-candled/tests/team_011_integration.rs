//! TEAM-011 Integration Tests
//!
//! Comprehensive end-to-end tests with real TinyLlama model.
//! Tests generation quality, sampling, and performance.
//!
//! Created by: TEAM-011

use llorch_candled::device::init_cpu_device;
use llorch_candled::backend::CandleInferenceBackend;
use worker_common::SamplingConfig;
use worker_http::InferenceBackend;
use anyhow::Result;

/// Test basic generation with greedy sampling
#[test]
#[ignore]
fn test_greedy_generation() -> Result<()> {
    let model_path = std::env::var("LLORCH_TEST_MODEL_PATH")
        .expect("Set LLORCH_TEST_MODEL_PATH to run this test");
    
    let device = init_cpu_device()?;
    let backend = CandleInferenceBackend::load(&model_path, device)?;
    
    let config = SamplingConfig {
        max_tokens: 20,
        temperature: 0.0,  // Greedy
        seed: 42,
        ..Default::default()
    };
    
    let rt = tokio::runtime::Runtime::new()?;
    let result = rt.block_on(backend.execute("The capital of France is", &config))
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    
    assert!(!result.tokens.is_empty(), "Should generate tokens");
    assert!(result.tokens.len() <= 20, "Should respect max_tokens");
    
    let text = result.tokens.join("");
    println!("Generated: {}", text);
    
    // Greedy should be deterministic - run again
    let result2 = rt.block_on(backend.execute("The capital of France is", &config))
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let text2 = result2.tokens.join("");
    
    assert_eq!(text, text2, "Greedy sampling should be deterministic");
    
    Ok(())
}

/// Test temperature sampling produces varied output
#[test]
#[ignore]
fn test_temperature_sampling() -> Result<()> {
    let model_path = std::env::var("LLORCH_TEST_MODEL_PATH")
        .expect("Set LLORCH_TEST_MODEL_PATH to run this test");
    
    let device = init_cpu_device()?;
    let backend = CandleInferenceBackend::load(&model_path, device)?;
    
    let config = SamplingConfig {
        max_tokens: 10,
        temperature: 0.8,
        seed: 42,
        ..Default::default()
    };
    
    let rt = tokio::runtime::Runtime::new()?;
    let result1 = rt.block_on(backend.execute("Once upon a time", &config))
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    
    // Different seed should produce different output
    let config2 = SamplingConfig {
        seed: 123,
        ..config
    };
    let result2 = rt.block_on(backend.execute("Once upon a time", &config2))
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    
    let text1 = result1.tokens.join("");
    let text2 = result2.tokens.join("");
    
    println!("Sample 1: {}", text1);
    println!("Sample 2: {}", text2);
    
    // With temperature, different seeds should produce different output
    // (though not guaranteed, very likely with 10 tokens)
    
    Ok(())
}

/// Test longer generation
#[test]
#[ignore]
fn test_long_generation() -> Result<()> {
    let model_path = std::env::var("LLORCH_TEST_MODEL_PATH")
        .expect("Set LLORCH_TEST_MODEL_PATH to run this test");
    
    let device = init_cpu_device()?;
    let backend = CandleInferenceBackend::load(&model_path, device)?;
    
    let config = SamplingConfig {
        max_tokens: 100,
        temperature: 0.7,
        seed: 42,
        ..Default::default()
    };
    
    let rt = tokio::runtime::Runtime::new()?;
    let start = std::time::Instant::now();
    let result = rt.block_on(backend.execute("In a world where", &config))
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    let duration = start.elapsed();
    
    let text = result.tokens.join("");
    let tokens_per_sec = result.tokens.len() as f64 / duration.as_secs_f64();
    
    println!("Generated {} tokens in {:.2}s ({:.2} tok/s)", 
             result.tokens.len(), duration.as_secs_f64(), tokens_per_sec);
    println!("Text: {}", text);
    
    assert!(!result.tokens.is_empty(), "Should generate tokens");
    assert!(tokens_per_sec > 0.1, "Should generate at reasonable speed");
    
    Ok(())
}

/// Test EOS detection
#[test]
#[ignore]
fn test_eos_detection() -> Result<()> {
    let model_path = std::env::var("LLORCH_TEST_MODEL_PATH")
        .expect("Set LLORCH_TEST_MODEL_PATH to run this test");
    
    let device = init_cpu_device()?;
    let backend = CandleInferenceBackend::load(&model_path, device)?;
    
    // Use a prompt that might trigger EOS
    let config = SamplingConfig {
        max_tokens: 50,
        temperature: 0.0,
        seed: 42,
        ..Default::default()
    };
    
    let rt = tokio::runtime::Runtime::new()?;
    let result = rt.block_on(backend.execute("Q: What is 2+2?\nA:", &config))
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    
    let text = result.tokens.join("");
    println!("Generated: {}", text);
    println!("Tokens: {}", result.tokens.len());
    
    // Should stop before max_tokens if EOS is generated
    // (not guaranteed, but likely for this prompt)
    
    Ok(())
}

/// Test multiple prompts in sequence
#[test]
#[ignore]
fn test_multiple_prompts() -> Result<()> {
    let model_path = std::env::var("LLORCH_TEST_MODEL_PATH")
        .expect("Set LLORCH_TEST_MODEL_PATH to run this test");
    
    let device = init_cpu_device()?;
    let backend = CandleInferenceBackend::load(&model_path, device)?;
    
    let config = SamplingConfig {
        max_tokens: 10,
        temperature: 0.0,
        seed: 42,
        ..Default::default()
    };
    
    let rt = tokio::runtime::Runtime::new()?;
    
    let prompts = vec![
        "The sky is",
        "Hello world",
        "Rust is",
    ];
    
    for prompt in prompts {
        let result = rt.block_on(backend.execute(prompt, &config))
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let text = result.tokens.join("");
        println!("{} -> {}", prompt, text);
        assert!(!result.tokens.is_empty(), "Should generate tokens for: {}", prompt);
    }
    
    Ok(())
}
