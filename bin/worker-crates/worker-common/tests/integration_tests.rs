//! Integration tests for worker-common
//!
//! Tests cross-module behavior and realistic usage patterns.

use worker_common::{InferenceResult, SamplingConfig, StopReason, WorkerError};

#[test]
fn test_inference_result_with_sampling_config() {
    // Simulate a complete inference flow
    let config = SamplingConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
        min_p: 0.05,
        stop_sequences: vec![],
        stop_strings: vec!["\n\n".to_string()],
        seed: 42,
        max_tokens: 100,
    };

    let tokens = vec!["Hello".to_string(), " world".to_string()];
    let token_ids = vec![100, 200];
    let result = InferenceResult::max_tokens(tokens, token_ids, config.seed, 1500);

    assert_eq!(result.seed, 42);
    assert_eq!(result.token_count(), 2);
    assert!(result.is_success());
}

#[test]
fn test_error_handling_with_partial_results() {
    // Simulate error during inference with partial generation
    let partial_tokens = vec!["Partial".to_string(), " output".to_string()];
    let partial_ids = vec![100, 200];
    let result = InferenceResult::error(partial_tokens.clone(), partial_ids.clone(), 42, 500);

    assert_eq!(result.token_count(), 2);
    assert!(!result.is_success());
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(result.tokens, partial_tokens);
}

#[test]
fn test_stop_sequence_matching_workflow() {
    // Simulate stop sequence detection
    let config = SamplingConfig {
        temperature: 0.7,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        min_p: 0.0,
        stop_sequences: vec![vec![10, 10]], // Tokenized "\n\n"
        stop_strings: vec!["\n\n".to_string()],
        seed: 42,
        max_tokens: 100,
    };

    assert!(config.has_stop_sequences());

    let tokens = vec!["Line 1".to_string(), "\n\n".to_string()];
    let token_ids = vec![100, 10, 10];
    let result =
        InferenceResult::stop_sequence(tokens, token_ids, config.seed, 800, "\n\n".to_string());

    assert!(result.is_success());
    assert_eq!(result.stop_reason, StopReason::StopSequence);
    assert_eq!(result.stop_sequence_matched, Some("\n\n".to_string()));
}

#[test]
fn test_greedy_sampling_workflow() {
    // Simulate greedy decoding (temperature = 0.0)
    let config = SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        min_p: 0.0,
        stop_sequences: vec![],
        stop_strings: vec![],
        seed: 42,
        max_tokens: 50,
    };

    assert!(config.is_greedy());
    assert_eq!(config.sampling_mode(), "greedy");
    assert!(!config.has_advanced_sampling());

    let tokens = vec!["The".to_string(), " answer".to_string(), " is".to_string()];
    let token_ids = vec![100, 200, 300];
    let result = InferenceResult::max_tokens(tokens, token_ids, config.seed, 300);

    assert_eq!(result.token_count(), 3);
    assert!(result.is_success());
}

#[test]
fn test_advanced_sampling_workflow() {
    // Simulate advanced sampling with all parameters
    let config = SamplingConfig {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40,
        repetition_penalty: 1.2,
        min_p: 0.02,
        stop_sequences: vec![],
        stop_strings: vec!["###".to_string(), "END".to_string()],
        seed: 12345,
        max_tokens: 200,
    };

    assert!(config.has_advanced_sampling());
    assert!(config.has_stop_sequences());
    assert!(!config.is_greedy());

    let mode = config.sampling_mode();
    assert!(mode.contains("top_p"));
    assert!(mode.contains("top_k"));
    assert!(mode.contains("rep_penalty"));
    assert!(mode.contains("min_p"));
}

#[test]
fn test_cancellation_workflow() {
    // Simulate user cancellation mid-generation
    let config = SamplingConfig::default();

    let partial_tokens = vec!["Hello".to_string(), " this".to_string(), " is".to_string()];
    let partial_ids = vec![100, 200, 300];
    let result =
        InferenceResult::cancelled(partial_tokens.clone(), partial_ids.clone(), config.seed, 450);

    assert_eq!(result.token_count(), 3);
    assert!(!result.is_success());
    assert_eq!(result.stop_reason, StopReason::Cancelled);
    assert_eq!(result.tokens, partial_tokens);
}

#[test]
fn test_error_types_with_retriability() {
    // Test that retriable errors are properly classified
    let cuda_error = WorkerError::Cuda("OOM".to_string());
    assert!(cuda_error.is_retriable());
    assert_eq!(cuda_error.code(), "CUDA_ERROR");

    let timeout_error = WorkerError::Timeout;
    assert!(timeout_error.is_retriable());
    assert_eq!(timeout_error.code(), "INFERENCE_TIMEOUT");

    let invalid_request = WorkerError::InvalidRequest("bad param".to_string());
    assert!(!invalid_request.is_retriable());
    assert_eq!(invalid_request.code(), "INVALID_REQUEST");

    let unhealthy = WorkerError::Unhealthy("model not loaded".to_string());
    assert!(!unhealthy.is_retriable());
    assert_eq!(unhealthy.code(), "WORKER_UNHEALTHY");
}

#[test]
fn test_sampling_config_validation_workflow() {
    // Test that conflicting configs are detected
    let bad_config = SamplingConfig {
        temperature: 0.3,
        top_p: 0.3,
        top_k: 5,
        repetition_penalty: 1.0,
        min_p: 0.0,
        stop_sequences: vec![],
        stop_strings: vec![],
        seed: 42,
        max_tokens: 100,
    };

    let validation_result = bad_config.validate_consistency();
    assert!(validation_result.is_err());

    let good_config = SamplingConfig::default();
    assert!(good_config.validate_consistency().is_ok());
}

#[test]
fn test_inference_result_stop_reason_descriptions() {
    // Test that all stop reasons have meaningful descriptions
    let max_tokens = InferenceResult::max_tokens(vec![], vec![], 42, 0);
    assert!(max_tokens.stop_reason_description().contains("max_tokens"));

    let mut eos = InferenceResult::max_tokens(vec![], vec![], 42, 0);
    eos.stop_reason = StopReason::Eos;
    assert!(eos.stop_reason_description().contains("End of sequence"));

    let stop_seq = InferenceResult::stop_sequence(vec![], vec![], 42, 0, "###".to_string());
    assert!(stop_seq.stop_reason_description().contains("###"));

    let cancelled = InferenceResult::cancelled(vec![], vec![], 42, 0);
    assert!(cancelled.stop_reason_description().contains("cancelled"));

    let error = InferenceResult::error(vec![], vec![], 42, 0);
    assert!(error.stop_reason_description().contains("error"));
}

#[test]
fn test_realistic_inference_pipeline() {
    // Simulate a complete inference pipeline from config to result

    // 1. Create sampling config
    let config = SamplingConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
        min_p: 0.0,
        stop_sequences: vec![vec![13, 13]], // "\n\n" tokenized
        stop_strings: vec!["\n\n".to_string()],
        seed: 42,
        max_tokens: 100,
    };

    // 2. Validate config
    assert!(config.validate_consistency().is_ok());
    assert!(config.has_advanced_sampling());
    assert!(config.has_stop_sequences());

    // 3. Simulate generation hitting stop sequence
    let tokens = vec![
        "The".to_string(),
        " quick".to_string(),
        " brown".to_string(),
        " fox".to_string(),
        "\n\n".to_string(),
    ];
    let token_ids = vec![100, 200, 300, 400, 13, 13];
    let result = InferenceResult::stop_sequence(
        tokens.clone(),
        token_ids.clone(),
        config.seed,
        1200,
        "\n\n".to_string(),
    );

    // 4. Verify result
    assert!(result.is_success());
    assert_eq!(result.token_count(), 5);
    assert_eq!(result.stop_reason, StopReason::StopSequence);
    assert_eq!(result.stop_sequence_matched, Some("\n\n".to_string()));
    assert_eq!(result.seed, 42);
    assert_eq!(result.decode_time_ms, 1200);

    // 5. Verify description is useful
    let desc = result.stop_reason_description();
    assert!(desc.contains("stop sequence"));
    assert!(desc.contains("\\n\\n"));
}

#[test]
fn test_error_recovery_workflow() {
    // Simulate error with partial results that could be retried

    // 1. Initial attempt fails with CUDA error
    let cuda_error = WorkerError::Cuda("device reset".to_string());
    assert!(cuda_error.is_retriable());

    // 2. Partial results are preserved
    let partial_tokens = vec!["Hello".to_string()];
    let partial_ids = vec![100];
    let error_result = InferenceResult::error(partial_tokens, partial_ids, 42, 100);

    assert!(!error_result.is_success());
    assert_eq!(error_result.token_count(), 1);

    // 3. On retry, we can use the same seed for reproducibility
    let retry_seed = error_result.seed;
    assert_eq!(retry_seed, 42);
}

#[test]
fn test_unicode_handling_across_modules() {
    // Test that unicode tokens work correctly
    let tokens =
        vec!["Hello".to_string(), " 世界".to_string(), "🌍".to_string(), "مرحبا".to_string()];
    let token_ids = vec![100, 200, 300, 400];

    let result = InferenceResult::max_tokens(tokens.clone(), token_ids.clone(), 42, 500);

    assert_eq!(result.token_count(), 4);
    assert_eq!(result.tokens, tokens);
    assert_eq!(result.token_ids, token_ids);
}

#[test]
fn test_large_generation_workflow() {
    // Test handling of large token sequences
    let large_tokens: Vec<String> = (0..2000).map(|i| format!("token{}", i)).collect();
    let large_ids: Vec<u32> = (0..2000).collect();

    let result = InferenceResult::max_tokens(large_tokens.clone(), large_ids.clone(), 42, 10000);

    assert_eq!(result.token_count(), 2000);
    assert_eq!(result.tokens.len(), 2000);
    assert_eq!(result.token_ids.len(), 2000);
    assert!(result.is_success());
}

#[test]
fn test_default_config_is_valid() {
    // Ensure default config is always valid
    let config = SamplingConfig::default();

    assert!(config.validate_consistency().is_ok());
    assert!(!config.is_greedy());
    assert!(!config.has_advanced_sampling());
    assert!(!config.has_stop_sequences());
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.max_tokens, 100);
}

// ---
// Verified by Testing Team 🔍
