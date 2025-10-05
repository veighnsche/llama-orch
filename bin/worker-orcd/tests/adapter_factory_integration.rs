//! Adapter Factory Integration Tests
//!
//! Tests for the adapter factory pattern with automatic architecture detection.
//!
//! Spec: FT-036

use worker_orcd::models::{AdapterFactory, AdapterForwardConfig, Architecture, ModelType};

/// Test factory with Qwen model
#[test]
fn test_factory_qwen() {
    let adapter = AdapterFactory::from_gguf("qwen-2.5-0.5b.gguf").unwrap();
    
    assert_eq!(adapter.model_type(), ModelType::Qwen2_5);
    assert_eq!(adapter.vocab_size().unwrap(), 151936);
    assert_eq!(adapter.hidden_dim().unwrap(), 896);
    assert_eq!(adapter.num_layers().unwrap(), 24);
    
    // Test generation
    let input_ids = vec![1, 2, 3];
    let config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: 3,
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };
    
    let output = adapter.generate(&input_ids, 10, &config).unwrap();
    assert_eq!(output.len(), input_ids.len() + 10);
}

/// Test factory with Phi-3 model
#[test]
fn test_factory_phi3() {
    let adapter = AdapterFactory::from_gguf("phi-3-mini.gguf").unwrap();
    
    assert_eq!(adapter.model_type(), ModelType::Phi3);
    assert_eq!(adapter.vocab_size().unwrap(), 32064);
    assert_eq!(adapter.hidden_dim().unwrap(), 3072);
    assert_eq!(adapter.num_layers().unwrap(), 32);
}

/// Test factory with GPT-2 model
#[test]
fn test_factory_gpt2() {
    let adapter = AdapterFactory::from_gguf("gpt2-small.gguf").unwrap();
    
    assert_eq!(adapter.model_type(), ModelType::GPT2);
    assert_eq!(adapter.vocab_size().unwrap(), 50257);
    assert_eq!(adapter.hidden_dim().unwrap(), 768);
    assert_eq!(adapter.num_layers().unwrap(), 12);
}

/// Test factory with explicit architecture
#[test]
fn test_factory_explicit_architecture() {
    let llama_adapter = AdapterFactory::from_gguf_with_arch("qwen-model.gguf", Architecture::Llama).unwrap();
    assert_eq!(llama_adapter.model_type(), ModelType::Qwen2_5);
    
    let gpt_adapter = AdapterFactory::from_gguf_with_arch("gpt2-model.gguf", Architecture::GPT).unwrap();
    assert_eq!(gpt_adapter.model_type(), ModelType::GPT2);
}

/// Test factory with architecture string
#[test]
fn test_factory_architecture_string() {
    let adapter = AdapterFactory::from_gguf_with_arch_str("qwen-model.gguf", "llama").unwrap();
    assert_eq!(adapter.model_type(), ModelType::Qwen2_5);
    
    let adapter = AdapterFactory::from_gguf_with_arch_str("gpt2-model.gguf", "gpt").unwrap();
    assert_eq!(adapter.model_type(), ModelType::GPT2);
}

/// Test factory error handling
#[test]
fn test_factory_error_handling() {
    // Unknown architecture
    let result = AdapterFactory::from_gguf("unknown-model.gguf");
    assert!(result.is_err());
    
    // Unsupported variant
    let result = AdapterFactory::from_gguf("llama-2-7b.gguf");
    assert!(result.is_err());
}

/// Test default factory for testing
#[test]
fn test_factory_default() {
    let adapter = AdapterFactory::default_for_testing().unwrap();
    assert_eq!(adapter.model_type(), ModelType::Qwen2_5);
    assert!(adapter.vocab_size().is_ok());
}

/// Test polymorphic model handling
#[test]
fn test_polymorphic_handling() {
    let models = vec![
        AdapterFactory::from_gguf("qwen-2.5-0.5b.gguf").unwrap(),
        AdapterFactory::from_gguf("phi-3-mini.gguf").unwrap(),
        AdapterFactory::from_gguf("gpt2-small.gguf").unwrap(),
    ];
    
    for adapter in &models {
        // All adapters support the same interface
        assert!(adapter.vocab_size().is_ok());
        assert!(adapter.hidden_dim().is_ok());
        assert!(adapter.num_layers().is_ok());
        assert!(adapter.vram_usage().is_ok());
    }
}

/// Test adapter switching
#[test]
fn test_adapter_switching() {
    let input_ids = vec![1, 2, 3];
    let config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: 3,
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };
    
    // Create different adapters
    let qwen = AdapterFactory::from_gguf("qwen-2.5-0.5b.gguf").unwrap();
    let phi3 = AdapterFactory::from_gguf("phi-3-mini.gguf").unwrap();
    let gpt2 = AdapterFactory::from_gguf("gpt2-small.gguf").unwrap();
    
    // All should generate with same interface
    let qwen_output = qwen.generate(&input_ids, 5, &config).unwrap();
    let phi3_output = phi3.generate(&input_ids, 5, &config).unwrap();
    let gpt2_output = gpt2.generate(&input_ids, 5, &config).unwrap();
    
    assert_eq!(qwen_output.len(), input_ids.len() + 5);
    assert_eq!(phi3_output.len(), input_ids.len() + 5);
    assert_eq!(gpt2_output.len(), input_ids.len() + 5);
}

// ---
// Built by Foundation-Alpha 🏗️
