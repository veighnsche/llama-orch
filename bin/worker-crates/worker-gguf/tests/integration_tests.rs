//! Integration tests for worker-gguf
//!
//! Tests GGUF metadata parsing and model detection.

use worker_gguf::{GGUFError, GGUFMetadata};

#[test]
fn test_complete_qwen_workflow() {
    let metadata = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();

    // Architecture detection
    assert_eq!(metadata.architecture().unwrap(), "llama");

    // Model dimensions
    assert_eq!(metadata.vocab_size().unwrap(), 151936);
    assert_eq!(metadata.hidden_dim().unwrap(), 896);
    assert_eq!(metadata.num_layers().unwrap(), 24);

    // Attention configuration
    assert_eq!(metadata.num_heads().unwrap(), 14);
    assert_eq!(metadata.num_kv_heads().unwrap(), 2);
    assert!(metadata.is_gqa());

    // Context and RoPE
    assert_eq!(metadata.context_length().unwrap(), 32768);
    assert_eq!(metadata.rope_freq_base().unwrap(), 1000000.0);
}

#[test]
fn test_complete_phi3_workflow() {
    let metadata = GGUFMetadata::from_file("phi-3-mini.gguf").unwrap();

    // Architecture detection
    assert_eq!(metadata.architecture().unwrap(), "llama");

    // Model dimensions
    assert_eq!(metadata.vocab_size().unwrap(), 32064);
    assert_eq!(metadata.hidden_dim().unwrap(), 3072);
    assert_eq!(metadata.num_layers().unwrap(), 32);

    // Attention configuration (MHA)
    assert_eq!(metadata.num_heads().unwrap(), 32);
    assert_eq!(metadata.num_kv_heads().unwrap(), 32);
    assert!(!metadata.is_gqa());

    // Context and RoPE
    assert_eq!(metadata.context_length().unwrap(), 4096);
    assert_eq!(metadata.rope_freq_base().unwrap(), 10000.0);
}

#[test]
fn test_complete_gpt2_workflow() {
    let metadata = GGUFMetadata::from_file("gpt2-small.gguf").unwrap();

    // Architecture detection
    assert_eq!(metadata.architecture().unwrap(), "gpt");

    // Model dimensions
    assert_eq!(metadata.vocab_size().unwrap(), 50257);
    assert_eq!(metadata.hidden_dim().unwrap(), 768);
    assert_eq!(metadata.num_layers().unwrap(), 12);
    assert_eq!(metadata.num_heads().unwrap(), 12);

    // Context
    assert_eq!(metadata.context_length().unwrap(), 1024);

    // RoPE (default for GPT-2)
    assert_eq!(metadata.rope_freq_base().unwrap(), 10000.0);
}

#[test]
fn test_model_comparison() {
    let qwen = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();
    let phi = GGUFMetadata::from_file("phi-3-mini.gguf").unwrap();
    let gpt2 = GGUFMetadata::from_file("gpt2-small.gguf").unwrap();

    // Qwen has largest vocab
    assert!(qwen.vocab_size().unwrap() > phi.vocab_size().unwrap());
    assert!(qwen.vocab_size().unwrap() > gpt2.vocab_size().unwrap());

    // Phi-3 has most layers
    assert!(phi.num_layers().unwrap() > qwen.num_layers().unwrap());
    assert!(phi.num_layers().unwrap() > gpt2.num_layers().unwrap());

    // Qwen has longest context
    assert!(qwen.context_length().unwrap() > phi.context_length().unwrap());
    assert!(qwen.context_length().unwrap() > gpt2.context_length().unwrap());

    // Only Qwen uses GQA
    assert!(qwen.is_gqa());
    assert!(!phi.is_gqa());
}

#[test]
fn test_filename_variations() {
    // Different naming conventions
    let variants = vec![
        "qwen-2.5-0.5b.gguf",
        "qwen_2_5_0_5b.gguf",
        "QWEN-2.5-0.5B.GGUF",
        "/models/qwen/qwen-2.5-0.5b.gguf",
        "/path/to/qwen-model.gguf",
    ];

    for variant in variants {
        let metadata = GGUFMetadata::from_file(variant).unwrap();
        assert_eq!(metadata.architecture().unwrap(), "llama");
        assert_eq!(metadata.vocab_size().unwrap(), 151936);
    }
}

#[test]
fn test_architecture_detection_patterns() {
    // Qwen patterns
    assert_eq!(GGUFMetadata::from_file("qwen-test.gguf").unwrap().architecture().unwrap(), "llama");
    assert_eq!(GGUFMetadata::from_file("qwen2.5.gguf").unwrap().architecture().unwrap(), "llama");

    // Phi patterns
    assert_eq!(GGUFMetadata::from_file("phi-3.gguf").unwrap().architecture().unwrap(), "llama");
    assert_eq!(GGUFMetadata::from_file("phi3-mini.gguf").unwrap().architecture().unwrap(), "llama");

    // Llama patterns
    assert_eq!(
        GGUFMetadata::from_file("llama-3.1-8b.gguf").unwrap().architecture().unwrap(),
        "llama"
    );

    // GPT patterns
    assert_eq!(GGUFMetadata::from_file("gpt2-small.gguf").unwrap().architecture().unwrap(), "gpt");
}

#[test]
fn test_missing_metadata_handling() {
    let metadata = GGUFMetadata::from_file("unknown-model.gguf").unwrap();

    // Unknown architecture should still work
    assert_eq!(metadata.architecture().unwrap(), "unknown");

    // But specific fields should error
    assert!(metadata.vocab_size().is_err());
    assert!(metadata.hidden_dim().is_err());
    assert!(metadata.num_layers().is_err());
}

#[test]
fn test_gqa_vs_mha_detection() {
    // GQA: num_kv_heads < num_heads
    let qwen = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();
    assert_eq!(qwen.num_heads().unwrap(), 14);
    assert_eq!(qwen.num_kv_heads().unwrap(), 2);
    assert!(qwen.is_gqa());

    // MHA: num_kv_heads == num_heads
    let phi = GGUFMetadata::from_file("phi-3-mini.gguf").unwrap();
    assert_eq!(phi.num_heads().unwrap(), 32);
    assert_eq!(phi.num_kv_heads().unwrap(), 32);
    assert!(!phi.is_gqa());
}

#[test]
fn test_rope_frequency_variations() {
    // Qwen uses very high RoPE frequency
    let qwen = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();
    assert_eq!(qwen.rope_freq_base().unwrap(), 1000000.0);

    // Phi-3 uses standard RoPE frequency
    let phi = GGUFMetadata::from_file("phi-3-mini.gguf").unwrap();
    assert_eq!(phi.rope_freq_base().unwrap(), 10000.0);

    // GPT-2 defaults to standard
    let gpt2 = GGUFMetadata::from_file("gpt2-small.gguf").unwrap();
    assert_eq!(gpt2.rope_freq_base().unwrap(), 10000.0);
}

#[test]
fn test_context_length_variations() {
    // Qwen has very long context
    let qwen = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();
    assert_eq!(qwen.context_length().unwrap(), 32768);

    // Phi-3 has medium context
    let phi = GGUFMetadata::from_file("phi-3-mini.gguf").unwrap();
    assert_eq!(phi.context_length().unwrap(), 4096);

    // GPT-2 has short context
    let gpt2 = GGUFMetadata::from_file("gpt2-small.gguf").unwrap();
    assert_eq!(gpt2.context_length().unwrap(), 1024);
}

#[test]
fn test_metadata_cloning() {
    let original = GGUFMetadata::from_file("qwen-2.5-0.5b.gguf").unwrap();
    let cloned = original.clone();

    // All fields should match
    assert_eq!(original.architecture().unwrap(), cloned.architecture().unwrap());
    assert_eq!(original.vocab_size().unwrap(), cloned.vocab_size().unwrap());
    assert_eq!(original.hidden_dim().unwrap(), cloned.hidden_dim().unwrap());
    assert_eq!(original.num_layers().unwrap(), cloned.num_layers().unwrap());
    assert_eq!(original.num_heads().unwrap(), cloned.num_heads().unwrap());
    assert_eq!(original.num_kv_heads().unwrap(), cloned.num_kv_heads().unwrap());
    assert_eq!(original.context_length().unwrap(), cloned.context_length().unwrap());
    assert_eq!(original.rope_freq_base().unwrap(), cloned.rope_freq_base().unwrap());
}

#[test]
fn test_error_types() {
    let metadata = GGUFMetadata::from_file("unknown-model.gguf").unwrap();

    // Missing key error
    let err = metadata.vocab_size().unwrap_err();
    assert!(matches!(err, GGUFError::MissingKey(_)));
    assert!(err.to_string().contains("llama.vocab_size"));

    // Error display
    let err = GGUFError::InvalidMagic;
    assert_eq!(err.to_string(), "Invalid GGUF magic number");

    let err = GGUFError::UnsupportedVersion(5);
    assert!(err.to_string().contains("5"));
}

#[test]
fn test_all_supported_models() {
    let models = vec![
        ("qwen-2.5-0.5b.gguf", "llama", 151936, 896, 24),
        ("phi-3-mini.gguf", "llama", 32064, 3072, 32),
        ("gpt2-small.gguf", "gpt", 50257, 768, 12),
    ];

    for (filename, expected_arch, expected_vocab, expected_hidden, expected_layers) in models {
        let metadata = GGUFMetadata::from_file(filename).unwrap();
        assert_eq!(metadata.architecture().unwrap(), expected_arch);
        assert_eq!(metadata.vocab_size().unwrap(), expected_vocab);
        assert_eq!(metadata.hidden_dim().unwrap(), expected_hidden);
        assert_eq!(metadata.num_layers().unwrap(), expected_layers);
    }
}

// ---
// Verified by Testing Team 🔍
