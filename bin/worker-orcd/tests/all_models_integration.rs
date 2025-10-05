//! All Models Integration Test - FT-041
//!
//! Comprehensive integration test covering all supported models.
//! Tests that both Llama and GPT architectures work correctly.

use worker_orcd::models::{AdapterFactory, AdapterForwardConfig, ModelType};

// FT-041: All Models Integration Test
// Tests that all supported models work correctly with the adapter pattern

#[test]
fn test_all_models_load() {
    let models = vec![
        ("qwen-2.5-0.5b.gguf", ModelType::Qwen2_5),
        ("phi-3-mini.gguf", ModelType::Phi3),
        ("gpt2-small.gguf", ModelType::GPT2),
    ];
    
    for (path, expected_type) in models {
        let adapter = AdapterFactory::from_gguf(path).unwrap();
        assert_eq!(adapter.model_type(), expected_type);
        assert!(adapter.vocab_size().is_ok());
        assert!(adapter.vram_usage().is_ok());
    }
}

#[test]
fn test_all_models_generate() {
    let models = vec![
        AdapterFactory::from_gguf("qwen-2.5-0.5b.gguf").unwrap(),
        AdapterFactory::from_gguf("phi-3-mini.gguf").unwrap(),
        AdapterFactory::from_gguf("gpt2-small.gguf").unwrap(),
    ];
    
    let input_ids = vec![1, 2, 3];
    let config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: 3,
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };
    
    for adapter in &models {
        let output = adapter.generate(&input_ids, 10, &config).unwrap();
        assert_eq!(output.len(), input_ids.len() + 10);
    }
}

#[test]
fn test_all_models_deterministic() {
    let models = vec![
        AdapterFactory::from_gguf("qwen-2.5-0.5b.gguf").unwrap(),
        AdapterFactory::from_gguf("phi-3-mini.gguf").unwrap(),
        AdapterFactory::from_gguf("gpt2-small.gguf").unwrap(),
    ];
    
    let input_ids = vec![1, 2, 3];
    let config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: 3,
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };
    
    for adapter in &models {
        let output1 = adapter.generate(&input_ids, 5, &config).unwrap();
        let output2 = adapter.generate(&input_ids, 5, &config).unwrap();
        assert_eq!(output1, output2, "Model {:?} not deterministic", adapter.model_type());
    }
}

#[test]
fn test_all_models_vram_usage() {
    let models = vec![
        "qwen-2.5-0.5b.gguf",
        "phi-3-mini.gguf",
        "gpt2-small.gguf",
    ];
    
    for path in models {
        let adapter = AdapterFactory::from_gguf(path).unwrap();
        let vram = adapter.vram_usage().unwrap();
        
        // Just verify VRAM is reported and reasonable (> 0, < 100GB)
        assert!(vram > 0, "VRAM usage should be > 0");
        assert!(vram < 100_000_000_000, "VRAM usage should be < 100GB");
    }
}

#[test]
fn test_all_models_temperature_sweep() {
    let adapter = AdapterFactory::from_gguf("qwen-2.5-0.5b.gguf").unwrap();
    let input_ids = vec![1, 2, 3];
    
    for temp in [0.1, 0.5, 1.0, 1.5, 2.0] {
        let config = AdapterForwardConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len: 3,
            cache_len: 0,
            temperature: temp,
            seed: 42,
        };
        
        let result = adapter.generate(&input_ids, 5, &config);
        assert!(result.is_ok(), "Failed with temperature {}", temp);
    }
}

#[test]
fn test_all_models_long_generation() {
    let models = vec![
        AdapterFactory::from_gguf("qwen-2.5-0.5b.gguf").unwrap(),
        AdapterFactory::from_gguf("gpt2-small.gguf").unwrap(),
    ];
    
    let input_ids = vec![1, 2, 3];
    let config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: 3,
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };
    
    for adapter in &models {
        let output = adapter.generate(&input_ids, 100, &config).unwrap();
        assert_eq!(output.len(), input_ids.len() + 100);
    }
}

// Built by Foundation-Alpha 🏗️
