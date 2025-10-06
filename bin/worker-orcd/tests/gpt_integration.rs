// GPT Integration Tests - GT-XXX
//
// Integration tests for GPT-2/GPT-3 models.
// Tests LayerNorm, GELU, and MHA kernels.
//
// Spec: M0-W-1220

use worker_orcd::models::gpt::{GPTConfig, GPTForward, GPTForwardConfig, GPTWeightLoader};

mod common;


/// Integration test: GPT-2 model loading
#[test]
fn test_gpt2_model_loading() {
    common::init_test_env();
    announce_stub_mode!("test_gpt2_model_loading");
    let config = GPTConfig::gpt2_small();
    let model = GPTWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();

    assert_eq!(model.config.vocab_size, 50257);
    assert_eq!(model.config.hidden_dim, 768);
    assert_eq!(model.config.num_layers, 12);
    assert_eq!(model.config.num_heads, 12);
    assert!(model.total_vram_bytes > 0);
}

/// Integration test: GPT-2 configuration presets
#[test]
fn test_gpt2_config_presets() {
    let small = GPTConfig::gpt2_small();
    let medium = GPTConfig::gpt2_medium();
    let large = GPTConfig::gpt2_large();
    let xl = GPTConfig::gpt2_xl();

    // Verify sizes increase
    assert!(small.hidden_dim < medium.hidden_dim);
    assert!(medium.hidden_dim < large.hidden_dim);
    assert!(large.hidden_dim < xl.hidden_dim);

    // All should validate
    assert!(small.validate().is_ok());
    assert!(medium.validate().is_ok());
    assert!(large.validate().is_ok());
    assert!(xl.validate().is_ok());
}

/// Integration test: GPT forward pass (stub)
#[test]
fn test_gpt_forward_pass() {
    common::init_test_env();
    announce_stub_mode!("test_gpt_forward_pass");
    let config = GPTConfig::gpt2_small();
    let model = GPTWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();

    let input_ids = vec![1, 2, 3, 4, 5];
    let fwd_config = GPTForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: input_ids.len(),
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };

    let result = GPTForward::prefill(&model, &input_ids, &fwd_config);
    assert!(result.is_ok());
}

/// Integration test: GPT generation (stub)
#[test]
fn test_gpt_generation() {
    common::init_test_env();
    announce_stub_mode!("test_gpt_generation");
    let config = GPTConfig::gpt2_small();
    let model = GPTWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();

    let input_ids = vec![1, 2, 3];
    let fwd_config = GPTForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: input_ids.len(),
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };

    let output = GPTForward::generate(&model, &input_ids, 10, &fwd_config).unwrap();
    assert_eq!(output.len(), input_ids.len() + 10);
}

/// Integration test: VRAM calculation accuracy
#[test]
fn test_gpt_vram_calculation() {
    let small = GPTConfig::gpt2_small();
    let medium = GPTConfig::gpt2_medium();
    let large = GPTConfig::gpt2_large();

    let small_vram = GPTWeightLoader::calculate_vram_usage(&small);
    let medium_vram = GPTWeightLoader::calculate_vram_usage(&medium);
    let large_vram = GPTWeightLoader::calculate_vram_usage(&large);

    // Larger models should use more VRAM
    assert!(small_vram < medium_vram);
    assert!(medium_vram < large_vram);

    // Log VRAM usage
    eprintln!("GPT-2 Small VRAM: {} MB", small_vram / (1024 * 1024));
    eprintln!("GPT-2 Medium VRAM: {} MB", medium_vram / (1024 * 1024));
    eprintln!("GPT-2 Large VRAM: {} MB", large_vram / (1024 * 1024));
}

/// Integration test: GPT decode (stub)
#[test]
fn test_gpt_decode() {
    common::init_test_env();
    announce_stub_mode!("test_gpt_decode");
    let config = GPTConfig::gpt2_small();
    let model = GPTWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();

    let input_id = 42;
    let fwd_config = GPTForwardConfig {
        is_prefill: false,
        batch_size: 1,
        seq_len: 1,
        cache_len: 10,
        temperature: 1.0,
        seed: 42,
    };

    let result = GPTForward::decode(&model, input_id, &fwd_config);
    assert!(result.is_ok());
}

/// Integration test: GPT temperature sweep
#[test]
fn test_gpt_temperature_sweep() {
    common::init_test_env();
    announce_stub_mode!("test_gpt_temperature_sweep");
    let config = GPTConfig::gpt2_small();
    let model = GPTWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();

    let input_ids = vec![1, 2, 3];

    for temp in [0.1, 0.5, 0.7, 1.0, 1.5, 2.0] {
        let fwd_config = GPTForwardConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len: input_ids.len(),
            cache_len: 0,
            temperature: temp,
            seed: 42,
        };

        let result = GPTForward::generate(&model, &input_ids, 5, &fwd_config);
        assert!(result.is_ok(), "Failed with temperature {}", temp);
    }
}

/// Integration test: GPT configuration validation
#[test]
fn test_gpt_config_validation() {
    // Valid config
    let valid = GPTConfig::gpt2_small();
    assert!(valid.validate().is_ok());

    // Invalid: vocab_size = 0
    let invalid1 = GPTConfig { vocab_size: 0, ..GPTConfig::gpt2_small() };
    assert!(invalid1.validate().is_err());

    // Invalid: hidden_dim not divisible by num_heads
    let invalid2 = GPTConfig { hidden_dim: 777, num_heads: 12, ..GPTConfig::gpt2_small() };
    assert!(invalid2.validate().is_err());
}

// TODO(GPT-Gamma): Add tests for actual kernel implementations
// These tests will be uncommented and implemented when kernels are ready:

/// Integration test: LayerNorm kernel
#[test]
#[ignore] // Ignored until LayerNorm kernel is implemented
fn test_gpt_layernorm_kernel() {
    // TODO(GPT-Gamma): Test LayerNorm kernel
    // - Forward pass with LayerNorm
    // - Verify output shape
    // - Verify numerical accuracy
    // - Compare with reference implementation
}

/// Integration test: GELU activation kernel
#[test]
#[ignore] // Ignored until GELU kernel is implemented
fn test_gpt_gelu_kernel() {
    // TODO(GPT-Gamma): Test GELU kernel
    // - Apply GELU to test tensor
    // - Verify output shape
    // - Verify numerical accuracy
    // - Compare with reference implementation
}

/// Integration test: MHA (Multi-Head Attention) kernel
#[test]
#[ignore] // Ignored until MHA kernel is implemented
fn test_gpt_mha_kernel() {
    // TODO(GPT-Gamma): Test MHA kernel
    // - Forward pass with MHA
    // - Verify attention scores
    // - Verify output shape
    // - Test with various sequence lengths
    // - Compare with reference implementation
}

/// Integration test: Absolute positional embeddings
#[test]
#[ignore] // Ignored until positional embedding is implemented
fn test_gpt_positional_embeddings() {
    // TODO(GPT-Gamma): Test positional embeddings
    // - Verify embedding lookup
    // - Test with various positions
    // - Verify addition to token embeddings
}

/// Integration test: Full GPT-2 pipeline with real model
#[test]
#[ignore] // Ignored until full pipeline is implemented
fn test_gpt2_full_pipeline() {
    // TODO(GPT-Gamma): Test full pipeline
    // 1. Load real GPT-2 model
    // 2. Create tokenizer
    // 3. Encode prompt
    // 4. Generate tokens
    // 5. Decode output
    // 6. Verify coherent output
}

// ---
// Built by Foundation-Alpha 🏗️
