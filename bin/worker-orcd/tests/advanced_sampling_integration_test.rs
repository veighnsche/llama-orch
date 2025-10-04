//! Integration tests for advanced sampling pipeline
//!
//! These tests validate the complete flow from HTTP request to inference result,
//! ensuring all layers work together correctly.
//!
//! # Spec References
//! - M0-W-1421: Advanced sampling parameters
//! - M0-W-1422: Stop sequences
//! - M0-W-1300: HTTP API extension

use worker_orcd::{InferenceExecutor, InferenceResult, SamplingConfig};
use worker_orcd::http::sse::StopReason;
use worker_orcd::http::validation::ExecuteRequest;

// ============================================================================
// Configuration Integration Tests
// ============================================================================

#[test]
fn test_request_to_config_to_executor() {
    let req = ExecuteRequest {
        job_id: "test-1".to_string(),
        prompt: "Hello world".to_string(),
        max_tokens: 10,
        temperature: 0.7,
        seed: Some(42),
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
        stop: vec!["\n\n".to_string()],
        min_p: 0.05,
    };
    
    // Validate request
    assert!(req.validate().is_ok());
    
    // Convert to config
    let config = SamplingConfig::from_request(&req);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_p, 0.9);
    assert_eq!(config.top_k, 50);
    assert_eq!(config.seed, 42);
    
    // Create executor
    let executor = InferenceExecutor::new(config);
    assert_eq!(executor.token_count(), 0);
    assert!(!executor.should_stop());
}

#[test]
fn test_backward_compatible_request_flow() {
    // Old Sprint 3 format
    let json = r#"{
        "job_id": "test-old",
        "prompt": "Hello",
        "max_tokens": 100,
        "temperature": 0.7,
        "seed": 42
    }"#;
    
    let req: ExecuteRequest = serde_json::from_str(json).unwrap();
    assert!(req.validate().is_ok());
    
    let config = SamplingConfig::from_request(&req);
    
    // Verify defaults applied
    assert_eq!(config.top_p, 1.0);
    assert_eq!(config.top_k, 0);
    assert_eq!(config.repetition_penalty, 1.0);
    assert_eq!(config.min_p, 0.0);
    assert!(!config.has_advanced_sampling());
    assert!(!config.has_stop_sequences());
}

#[test]
fn test_advanced_request_flow() {
    // New Sprint 4 format
    let json = r#"{
        "job_id": "test-new",
        "prompt": "Hello",
        "max_tokens": 100,
        "temperature": 0.7,
        "seed": 42,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "stop": ["\\n\\n", "END"],
        "min_p": 0.05
    }"#;
    
    let req: ExecuteRequest = serde_json::from_str(json).unwrap();
    assert!(req.validate().is_ok());
    
    let config = SamplingConfig::from_request(&req);
    
    // Verify all parameters
    assert_eq!(config.top_p, 0.9);
    assert_eq!(config.top_k, 50);
    assert_eq!(config.repetition_penalty, 1.1);
    assert_eq!(config.min_p, 0.05);
    assert_eq!(config.stop_strings.len(), 2);
    assert!(config.has_advanced_sampling());
    assert!(config.has_stop_sequences());
}

// ============================================================================
// Stop Reason Integration Tests
// ============================================================================

#[test]
fn test_max_tokens_stop_reason_flow() {
    let req = ExecuteRequest {
        job_id: "test-max-tokens".to_string(),
        prompt: "Count to 10".to_string(),
        max_tokens: 3,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    };
    
    let config = SamplingConfig::from_request(&req);
    let mut executor = InferenceExecutor::new(config);
    
    // Generate 3 tokens
    assert!(executor.add_token("1".to_string(), 100));
    assert!(executor.add_token("2".to_string(), 101));
    assert!(!executor.add_token("3".to_string(), 102));  // Hits max_tokens
    
    assert!(executor.should_stop());
    
    let result = executor.finalize();
    assert_eq!(result.stop_reason, StopReason::MaxTokens);
    assert_eq!(result.token_count(), 3);
    assert!(result.stop_sequence_matched.is_none());
    assert!(result.is_success());
}

#[test]
fn test_stop_sequence_flow() {
    let req = ExecuteRequest {
        job_id: "test-stop-seq".to_string(),
        prompt: "Write JSON".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec!["}".to_string()],
        min_p: 0.0,
    };
    
    let mut config = SamplingConfig::from_request(&req);
    // Simulate tokenized stop sequence
    config.stop_sequences = vec![vec![125]];  // Token ID for '}'
    
    let mut executor = InferenceExecutor::new(config);
    
    // Generate tokens until stop sequence
    assert!(executor.add_token("{".to_string(), 123));
    assert!(executor.add_token("\"name\"".to_string(), 124));
    assert!(!executor.add_token("}".to_string(), 125));  // Matches stop
    
    assert!(executor.should_stop());
    
    let result = executor.finalize();
    assert_eq!(result.stop_reason, StopReason::StopSequence);
    assert_eq!(result.stop_sequence_matched, Some("}".to_string()));
    assert!(result.is_success());
}

#[test]
fn test_cancellation_flow() {
    let req = ExecuteRequest {
        job_id: "test-cancel".to_string(),
        prompt: "Long generation".to_string(),
        max_tokens: 1000,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    };
    
    let config = SamplingConfig::from_request(&req);
    let mut executor = InferenceExecutor::new(config);
    
    // Generate some tokens
    executor.add_token("Token".to_string(), 100);
    executor.add_token("Token".to_string(), 101);
    
    // Client cancels
    executor.cancel();
    
    assert!(executor.should_stop());
    
    let result = executor.finalize();
    assert_eq!(result.stop_reason, StopReason::Cancelled);
    assert_eq!(result.token_count(), 2);
    assert!(!result.is_success());
}

#[test]
fn test_error_flow() {
    let req = ExecuteRequest {
        job_id: "test-error".to_string(),
        prompt: "Test".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    };
    
    let config = SamplingConfig::from_request(&req);
    let mut executor = InferenceExecutor::new(config);
    
    // Generate some tokens
    executor.add_token("Token".to_string(), 100);
    
    // Error occurs
    executor.error();
    
    assert!(executor.should_stop());
    
    let result = executor.finalize();
    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(!result.is_success());
}

// ============================================================================
// Multi-Parameter Integration Tests
// ============================================================================

#[test]
fn test_all_parameters_enabled() {
    let req = ExecuteRequest {
        job_id: "test-all-params".to_string(),
        prompt: "Test all parameters".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        seed: Some(42),
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
        stop: vec!["\n\n".to_string(), "END".to_string()],
        min_p: 0.05,
    };
    
    // Validate
    assert!(req.validate().is_ok());
    
    // Convert to config
    let config = SamplingConfig::from_request(&req);
    assert!(config.has_advanced_sampling());
    assert!(config.has_stop_sequences());
    
    // Check sampling mode description
    let mode = config.sampling_mode();
    assert!(mode.contains("top_p"));
    assert!(mode.contains("top_k"));
    assert!(mode.contains("rep_penalty"));
    assert!(mode.contains("min_p"));
    
    // Create executor
    let executor = InferenceExecutor::new(config);
    assert_eq!(executor.config().top_p, 0.9);
    assert_eq!(executor.config().top_k, 50);
}

#[test]
fn test_greedy_mode_detection() {
    let req = ExecuteRequest {
        job_id: "test-greedy".to_string(),
        prompt: "Test greedy".to_string(),
        max_tokens: 100,
        temperature: 0.0,  // Greedy
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    };
    
    let config = SamplingConfig::from_request(&req);
    assert!(config.is_greedy());
    assert_eq!(config.sampling_mode(), "greedy");
}

#[test]
fn test_consistency_validation_in_pipeline() {
    let req = ExecuteRequest {
        job_id: "test-conflict".to_string(),
        prompt: "Test".to_string(),
        max_tokens: 100,
        temperature: 0.3,  // Low temp
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.6,  // High min_p - conflicts with low temp
    };
    
    let config = SamplingConfig::from_request(&req);
    let validation = config.validate_consistency();
    
    assert!(validation.is_err());
    assert!(validation.unwrap_err().contains("Conflicting"));
}

// ============================================================================
// Result Construction Tests
// ============================================================================

#[test]
fn test_result_from_max_tokens() {
    let result = InferenceResult::max_tokens(
        vec!["Hello".to_string(), " world".to_string()],
        vec![100, 200],
        42,
        1000,
    );
    
    assert_eq!(result.stop_reason, StopReason::MaxTokens);
    assert!(result.stop_sequence_matched.is_none());
    assert_eq!(result.token_count(), 2);
    assert!(result.is_success());
    
    let desc = result.stop_reason_description();
    assert!(desc.contains("max_tokens"));
}

#[test]
fn test_result_from_stop_sequence() {
    let result = InferenceResult::stop_sequence(
        vec!["Line 1".to_string(), "\n\n".to_string()],
        vec![100, 200],
        42,
        500,
        "\n\n".to_string(),
    );
    
    assert_eq!(result.stop_reason, StopReason::StopSequence);
    assert_eq!(result.stop_sequence_matched, Some("\n\n".to_string()));
    assert!(result.is_success());
    
    let desc = result.stop_reason_description();
    assert!(desc.contains("stop sequence"));
    assert!(desc.contains("\\n\\n"));
}

#[test]
fn test_result_serialization_to_sse() {
    use worker_orcd::http::sse::InferenceEvent;
    
    // Create result
    let result = InferenceResult::stop_sequence(
        vec!["Test".to_string()],
        vec![100],
        42,
        250,
        "END".to_string(),
    );
    
    // Convert to SSE event
    let event = InferenceEvent::End {
        tokens_out: result.token_count(),
        decode_time_ms: result.decode_time_ms,
        stop_reason: result.stop_reason.clone(),
        stop_sequence_matched: result.stop_sequence_matched.clone(),
    };
    
    // Serialize
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"stop_reason\":\"stop_sequence\""));
    assert!(json.contains("\"stop_sequence_matched\":\"END\""));
}

// ============================================================================
// End-to-End Pipeline Tests
// ============================================================================

#[test]
fn test_complete_pipeline_max_tokens() {
    // 1. HTTP Request
    let json = r#"{
        "job_id": "e2e-max-tokens",
        "prompt": "Count to 100",
        "max_tokens": 5,
        "temperature": 0.7,
        "seed": 42
    }"#;
    
    let req: ExecuteRequest = serde_json::from_str(json).unwrap();
    
    // 2. Validation
    assert!(req.validate_all().is_ok());
    
    // 3. Configuration
    let config = SamplingConfig::from_request(&req);
    assert_eq!(config.max_tokens, 5);
    
    // 4. Execution
    let mut executor = InferenceExecutor::new(config);
    for i in 0..5 {
        let should_continue = executor.add_token(i.to_string(), 100 + i);
        if i < 4 {
            assert!(should_continue);
        } else {
            assert!(!should_continue);  // Last token hits max
        }
    }
    
    // 5. Result
    let result = executor.finalize();
    assert_eq!(result.stop_reason, StopReason::MaxTokens);
    assert_eq!(result.token_count(), 5);
    
    // 6. SSE Response
    use worker_orcd::http::sse::InferenceEvent;
    let event = InferenceEvent::End {
        tokens_out: result.token_count(),
        decode_time_ms: result.decode_time_ms,
        stop_reason: result.stop_reason,
        stop_sequence_matched: result.stop_sequence_matched,
    };
    
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("max_tokens"));
}

#[test]
fn test_complete_pipeline_stop_sequence() {
    // 1. HTTP Request
    let json = r#"{
        "job_id": "e2e-stop-seq",
        "prompt": "Generate JSON",
        "max_tokens": 100,
        "temperature": 0.7,
        "seed": 42,
        "stop": ["}"]
    }"#;
    
    let req: ExecuteRequest = serde_json::from_str(json).unwrap();
    
    // 2. Validation
    assert!(req.validate_all().is_ok());
    
    // 3. Configuration
    let mut config = SamplingConfig::from_request(&req);
    config.stop_sequences = vec![vec![125]];  // Token ID for '}'
    
    // 4. Execution
    let mut executor = InferenceExecutor::new(config);
    executor.add_token("{".to_string(), 123);
    executor.add_token("\"key\"".to_string(), 124);
    let should_continue = executor.add_token("}".to_string(), 125);
    
    assert!(!should_continue);
    assert!(executor.should_stop());
    
    // 5. Result
    let result = executor.finalize();
    assert_eq!(result.stop_reason, StopReason::StopSequence);
    assert_eq!(result.stop_sequence_matched, Some("}".to_string()));
    
    // 6. SSE Response
    use worker_orcd::http::sse::InferenceEvent;
    let event = InferenceEvent::End {
        tokens_out: result.token_count(),
        decode_time_ms: result.decode_time_ms,
        stop_reason: result.stop_reason,
        stop_sequence_matched: result.stop_sequence_matched,
    };
    
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("stop_sequence"));
    assert!(json.contains("\"}\""));
}

#[test]
fn test_complete_pipeline_all_parameters() {
    // 1. HTTP Request with all parameters
    let json = r#"{
        "job_id": "e2e-all-params",
        "prompt": "Test",
        "max_tokens": 50,
        "temperature": 0.7,
        "seed": 42,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "stop": ["\\n\\n", "END"],
        "min_p": 0.05
    }"#;
    
    let req: ExecuteRequest = serde_json::from_str(json).unwrap();
    
    // 2. Validation
    let validation = req.validate_all();
    assert!(validation.is_ok(), "Validation should pass");
    
    // 3. Configuration
    let config = SamplingConfig::from_request(&req);
    assert!(config.has_advanced_sampling());
    assert!(config.has_stop_sequences());
    
    // 4. Consistency check
    let consistency = config.validate_consistency();
    assert!(consistency.is_ok(), "Parameters should be consistent");
    
    // 5. Execution
    let executor = InferenceExecutor::new(config);
    assert_eq!(executor.config().top_p, 0.9);
    assert_eq!(executor.config().top_k, 50);
    assert_eq!(executor.config().repetition_penalty, 1.1);
    assert_eq!(executor.config().min_p, 0.05);
}

// ============================================================================
// Error Handling Integration Tests
// ============================================================================

#[test]
fn test_validation_error_propagation() {
    let json = r#"{
        "job_id": "test-invalid",
        "prompt": "Test",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1.5,
        "repetition_penalty": 3.0
    }"#;
    
    let req: ExecuteRequest = serde_json::from_str(json).unwrap();
    
    let validation = req.validate_all();
    assert!(validation.is_err());
    
    let errors = validation.unwrap_err();
    assert!(errors.errors.len() >= 2);
    
    let fields: Vec<_> = errors.errors.iter().map(|e| e.field.as_str()).collect();
    assert!(fields.contains(&"top_p"));
    assert!(fields.contains(&"repetition_penalty"));
}

#[test]
fn test_stop_sequence_validation_errors() {
    let json = r#"{
        "job_id": "test-stop-invalid",
        "prompt": "Test",
        "max_tokens": 100,
        "temperature": 0.7,
        "stop": ["", "valid", "another", "third", "fourth", "fifth"]
    }"#;
    
    let req: ExecuteRequest = serde_json::from_str(json).unwrap();
    
    let validation = req.validate_all();
    assert!(validation.is_err());
    
    let errors = validation.unwrap_err();
    
    // Should have errors for: empty sequence + too many sequences
    let stop_errors: Vec<_> = errors.errors.iter()
        .filter(|e| e.field == "stop")
        .collect();
    assert!(stop_errors.len() >= 2);
}

// ============================================================================
// Performance Integration Tests
// ============================================================================

#[test]
fn test_large_generation_with_stop_sequences() {
    let req = ExecuteRequest {
        job_id: "test-large".to_string(),
        prompt: "Long text".to_string(),
        max_tokens: 1000,
        temperature: 0.7,
        seed: Some(42),
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
        stop: vec!["\n\n".to_string()],
        min_p: 0.0,
    };
    
    let mut config = SamplingConfig::from_request(&req);
    config.stop_sequences = vec![vec![13, 13]];
    
    let mut executor = InferenceExecutor::new(config);
    
    // Generate many tokens
    for i in 0..500 {
        executor.add_token(format!("token{}", i), 100 + i);
    }
    
    // Stop sequence
    executor.add_token("\n".to_string(), 13);
    let should_continue = executor.add_token("\n".to_string(), 13);
    
    assert!(!should_continue);
    
    let result = executor.finalize();
    assert_eq!(result.stop_reason, StopReason::StopSequence);
    assert_eq!(result.token_count(), 502);
}

#[test]
fn test_seed_generation_consistency() {
    let req1 = ExecuteRequest {
        job_id: "test-seed-1".to_string(),
        prompt: "Test".to_string(),
        max_tokens: 10,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    };
    
    let req2 = ExecuteRequest {
        job_id: "test-seed-2".to_string(),
        prompt: "Test".to_string(),
        max_tokens: 10,
        temperature: 0.7,
        seed: Some(42),  // Same seed
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    };
    
    let config1 = SamplingConfig::from_request(&req1);
    let config2 = SamplingConfig::from_request(&req2);
    
    assert_eq!(config1.seed, config2.seed);
    assert_eq!(config1.seed, 42);
}

// ============================================================================
// Documentation Tests
// ============================================================================

#[test]
fn test_sampling_mode_descriptions() {
    // Greedy
    let greedy = ExecuteRequest {
        job_id: "test".to_string(),
        prompt: "Test".to_string(),
        max_tokens: 10,
        temperature: 0.0,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    };
    let config = SamplingConfig::from_request(&greedy);
    assert_eq!(config.sampling_mode(), "greedy");
    
    // Basic stochastic
    let basic = ExecuteRequest {
        job_id: "test".to_string(),
        prompt: "Test".to_string(),
        max_tokens: 10,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    };
    let config = SamplingConfig::from_request(&basic);
    assert!(config.sampling_mode().contains("stochastic"));
    assert!(config.sampling_mode().contains("temp=0.70"));
    
    // Advanced stochastic
    let advanced = ExecuteRequest {
        job_id: "test".to_string(),
        prompt: "Test".to_string(),
        max_tokens: 10,
        temperature: 0.7,
        seed: Some(42),
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
        stop: vec![],
        min_p: 0.05,
    };
    let config = SamplingConfig::from_request(&advanced);
    let mode = config.sampling_mode();
    assert!(mode.contains("top_p=0.90"));
    assert!(mode.contains("top_k=50"));
    assert!(mode.contains("rep_penalty=1.10"));
    assert!(mode.contains("min_p=0.05"));
}

// ---
// Built by Foundation-Alpha 🏗️
