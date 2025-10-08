//! Checkpoint 0: Foundation Setup
//!
//! Validates that the foundational infrastructure is correctly set up.
//!
//! Created by: TEAM-000

use llorch_candled::backend::CandleInferenceBackend;
use worker_common::SamplingConfig;
use worker_http::InferenceBackend;

#[tokio::test]
async fn test_backend_stub_works() {
    // Create backend
    let backend = CandleInferenceBackend::load("test.gguf").unwrap();
    
    // Test execute returns stub data
    let config = SamplingConfig::default();
    let result = backend.execute("Hello", &config).await.unwrap();
    
    // Validate stub response
    assert_eq!(result.tokens.len(), 3);
    assert_eq!(result.tokens[0], "STUB");
    assert_eq!(result.tokens[1], "LLAMA2");
    assert_eq!(result.tokens[2], "RESPONSE");
}

#[test]
fn test_backend_is_healthy() {
    let backend = CandleInferenceBackend::load("test.gguf").unwrap();
    assert!(backend.is_healthy());
}

#[test]
fn test_backend_memory_type() {
    let backend = CandleInferenceBackend::load("test.gguf").unwrap();
    
    #[cfg(feature = "cuda")]
    assert_eq!(backend.memory_architecture(), "cuda");
    
    #[cfg(not(feature = "cuda"))]
    assert_eq!(backend.memory_architecture(), "cpu");
}

#[test]
fn test_backend_worker_type() {
    let backend = CandleInferenceBackend::load("test.gguf").unwrap();
    
    #[cfg(feature = "cuda")]
    assert_eq!(backend.worker_type(), "candle-cuda");
    
    #[cfg(not(feature = "cuda"))]
    assert_eq!(backend.worker_type(), "candle-cpu");
}

#[test]
fn test_backend_capabilities() {
    let backend = CandleInferenceBackend::load("test.gguf").unwrap();
    let caps = backend.capabilities();
    
    assert!(caps.contains(&"text-gen"));
    assert!(caps.contains(&"llama-2"));
}

#[test]
fn test_backend_vram_usage() {
    let backend = CandleInferenceBackend::load("test.gguf").unwrap();
    
    // For stub, VRAM should be 0
    // After real implementation, this will vary
    assert_eq!(backend.vram_usage(), 0);
}
