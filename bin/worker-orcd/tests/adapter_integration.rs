// LlamaInferenceAdapter Integration Tests - LT-033
//
// Tests unified adapter pattern for Llama-family models.
// Validates consistent interface across Qwen and Phi-3.
//
// Spec: FT-071 (adapter pattern)

use worker_orcd::models::{
    phi3::{Phi3Config, Phi3WeightLoader},
    qwen::{QwenConfig, QwenWeightLoader},
    AdapterForwardConfig, LlamaInferenceAdapter, ModelType,
};

#[test]
fn test_adapter_unified_interface_qwen() {
    let config = QwenConfig::qwen2_5_0_5b();
    let model = QwenWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
    let adapter = LlamaInferenceAdapter::new_qwen(model);

    // Test unified interface
    assert_eq!(adapter.model_type(), ModelType::Qwen2_5);
    assert!(adapter.vocab_size().is_ok());
    assert!(adapter.hidden_dim().is_ok());
    assert!(adapter.num_layers().is_ok());
    assert!(adapter.vram_usage().is_ok());
}

#[test]
fn test_adapter_unified_interface_phi3() {
    let config = Phi3Config::phi3_mini_4k();
    let model = Phi3WeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
    let adapter = LlamaInferenceAdapter::new_phi3(model);

    // Test unified interface
    assert_eq!(adapter.model_type(), ModelType::Phi3);
    assert!(adapter.vocab_size().is_ok());
    assert!(adapter.hidden_dim().is_ok());
    assert!(adapter.num_layers().is_ok());
    assert!(adapter.vram_usage().is_ok());
}

#[test]
fn test_adapter_generation_qwen() {
    let config = QwenConfig::qwen2_5_0_5b();
    let model = QwenWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
    let adapter = LlamaInferenceAdapter::new_qwen(model);

    let input_ids = vec![1, 2, 3];
    let fwd_config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: 3,
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };

    let result = adapter.generate(&input_ids, 10, &fwd_config);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), input_ids.len() + 10);
}

#[test]
fn test_adapter_generation_phi3() {
    let config = Phi3Config::phi3_mini_4k();
    let model = Phi3WeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
    let adapter = LlamaInferenceAdapter::new_phi3(model);

    let input_ids = vec![1, 2, 3];
    let fwd_config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: 3,
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };

    let result = adapter.generate(&input_ids, 10, &fwd_config);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), input_ids.len() + 10);
}

#[test]
fn test_adapter_consistent_interface() {
    // Test that both models expose same interface
    let qwen_config = QwenConfig::qwen2_5_0_5b();
    let qwen_model = QwenWeightLoader::load_to_vram("dummy.gguf", &qwen_config).unwrap();
    let qwen_adapter = LlamaInferenceAdapter::new_qwen(qwen_model);

    let phi3_config = Phi3Config::phi3_mini_4k();
    let phi3_model = Phi3WeightLoader::load_to_vram("dummy.gguf", &phi3_config).unwrap();
    let phi3_adapter = LlamaInferenceAdapter::new_phi3(phi3_model);

    // Both should support same operations
    assert!(qwen_adapter.vocab_size().is_ok());
    assert!(phi3_adapter.vocab_size().is_ok());

    assert!(qwen_adapter.hidden_dim().is_ok());
    assert!(phi3_adapter.hidden_dim().is_ok());

    assert!(qwen_adapter.num_layers().is_ok());
    assert!(phi3_adapter.num_layers().is_ok());

    assert!(qwen_adapter.vram_usage().is_ok());
    assert!(phi3_adapter.vram_usage().is_ok());
}

#[test]
fn test_adapter_model_differences() {
    let qwen_config = QwenConfig::qwen2_5_0_5b();
    let qwen_model = QwenWeightLoader::load_to_vram("dummy.gguf", &qwen_config).unwrap();
    let qwen_adapter = LlamaInferenceAdapter::new_qwen(qwen_model);

    let phi3_config = Phi3Config::phi3_mini_4k();
    let phi3_model = Phi3WeightLoader::load_to_vram("dummy.gguf", &phi3_config).unwrap();
    let phi3_adapter = LlamaInferenceAdapter::new_phi3(phi3_model);

    // Verify models have different characteristics
    assert_ne!(qwen_adapter.vocab_size().unwrap(), phi3_adapter.vocab_size().unwrap());
    assert_ne!(qwen_adapter.hidden_dim().unwrap(), phi3_adapter.hidden_dim().unwrap());
    assert_ne!(qwen_adapter.num_layers().unwrap(), phi3_adapter.num_layers().unwrap());

    eprintln!("Qwen vocab: {}", qwen_adapter.vocab_size().unwrap());
    eprintln!("Phi-3 vocab: {}", phi3_adapter.vocab_size().unwrap());
}

#[test]
fn test_adapter_prefill_decode_cycle() {
    let config = QwenConfig::qwen2_5_0_5b();
    let model = QwenWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
    let adapter = LlamaInferenceAdapter::new_qwen(model);

    let input_ids = vec![1, 2, 3];

    // Prefill
    let prefill_config = AdapterForwardConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: 3,
        cache_len: 0,
        temperature: 1.0,
        seed: 42,
    };

    let prefill_result = adapter.prefill(&input_ids, &prefill_config);
    assert!(prefill_result.is_ok());

    // Decode
    let decode_config = AdapterForwardConfig {
        is_prefill: false,
        batch_size: 1,
        seq_len: 1,
        cache_len: 3,
        temperature: 1.0,
        seed: 42,
    };

    let decode_result = adapter.decode(42, &decode_config);
    assert!(decode_result.is_ok());
}

#[test]
fn test_adapter_temperature_control() {
    let config = QwenConfig::qwen2_5_0_5b();
    let model = QwenWeightLoader::load_to_vram("dummy.gguf", &config).unwrap();
    let adapter = LlamaInferenceAdapter::new_qwen(model);

    let input_ids = vec![1, 2, 3];

    // Test different temperatures
    let temps = vec![0.1, 0.7, 1.0, 1.5];

    for temp in temps {
        let fwd_config = AdapterForwardConfig {
            is_prefill: true,
            batch_size: 1,
            seq_len: 3,
            cache_len: 0,
            temperature: temp,
            seed: 42,
        };

        let result = adapter.generate(&input_ids, 5, &fwd_config);
        assert!(result.is_ok(), "Failed with temperature {}", temp);
    }
}
