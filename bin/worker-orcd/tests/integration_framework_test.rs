//! Integration test framework validation
//!
//! Tests the test framework itself to ensure it works correctly.
//!
//! # Spec References
//! - M0-W-1820: Integration test framework

use worker_orcd::http::sse::{InferenceEvent, StopReason};
use worker_orcd::tests::integration::{
    assert_event_order, extract_tokens, make_test_request, TestConfig, TestModel, TestPrompts,
    WorkerTestHarness,
};

// ============================================================================
// Helper Function Tests
// ============================================================================

#[test]
fn test_event_order_validation_valid() {
    let events = vec![
        InferenceEvent::Started {
            job_id: "test".to_string(),
            model: "test".to_string(),
            started_at: "2025-10-05T00:00:00Z".to_string(),
        },
        InferenceEvent::Token { t: "Hello".to_string(), i: 0 },
        InferenceEvent::Token { t: " world".to_string(), i: 1 },
        InferenceEvent::End {
            tokens_out: 2,
            decode_time_ms: 100,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        },
    ];

    assert!(assert_event_order(&events).is_ok());
}

#[test]
fn test_event_order_validation_empty() {
    let events = vec![];
    assert!(assert_event_order(&events).is_err());
}

#[test]
fn test_event_order_validation_no_started() {
    let events = vec![
        InferenceEvent::Token { t: "Hello".to_string(), i: 0 },
        InferenceEvent::End {
            tokens_out: 1,
            decode_time_ms: 100,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        },
    ];

    assert!(assert_event_order(&events).is_err());
}

#[test]
fn test_event_order_validation_no_terminal() {
    let events = vec![
        InferenceEvent::Started {
            job_id: "test".to_string(),
            model: "test".to_string(),
            started_at: "2025-10-05T00:00:00Z".to_string(),
        },
        InferenceEvent::Token { t: "Hello".to_string(), i: 0 },
    ];

    assert!(assert_event_order(&events).is_err());
}

#[test]
fn test_extract_tokens_multiple() {
    let events = vec![
        InferenceEvent::Started {
            job_id: "test".to_string(),
            model: "test".to_string(),
            started_at: "2025-10-05T00:00:00Z".to_string(),
        },
        InferenceEvent::Token { t: "Hello".to_string(), i: 0 },
        InferenceEvent::Token { t: " world".to_string(), i: 1 },
        InferenceEvent::Token { t: "!".to_string(), i: 2 },
        InferenceEvent::End {
            tokens_out: 3,
            decode_time_ms: 100,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        },
    ];

    let tokens = extract_tokens(&events);
    assert_eq!(tokens, vec!["Hello", " world", "!"]);
}

#[test]
fn test_extract_tokens_none() {
    let events = vec![
        InferenceEvent::Started {
            job_id: "test".to_string(),
            model: "test".to_string(),
            started_at: "2025-10-05T00:00:00Z".to_string(),
        },
        InferenceEvent::End {
            tokens_out: 0,
            decode_time_ms: 100,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        },
    ];

    let tokens = extract_tokens(&events);
    assert!(tokens.is_empty());
}

// ============================================================================
// Fixture Tests
// ============================================================================

#[test]
fn test_qwen_model_fixture() {
    let model = TestModel::qwen2_5_0_5b();

    assert_eq!(model.name, "Qwen2.5-0.5B");
    assert_eq!(model.num_layers, 24);
    assert_eq!(model.num_kv_heads, 2);
    assert_eq!(model.head_dim, 64);
    assert_eq!(model.vocab_size, 151936);
}

#[test]
fn test_mock_model_fixture() {
    let model = TestModel::mock();

    assert_eq!(model.name, "Mock");
    assert_eq!(model.num_layers, 2);
    assert_eq!(model.num_kv_heads, 2);
}

#[test]
fn test_default_config() {
    let config = TestConfig::default();

    assert_eq!(config.gpu_device, 0);
    assert_eq!(config.timeout_secs, 30);
    assert_eq!(config.max_tokens, 10);
}

#[test]
fn test_fast_config() {
    let config = TestConfig::fast();

    assert_eq!(config.max_tokens, 5);
    assert!(config.timeout_secs <= 10);
}

#[test]
fn test_long_config() {
    let config = TestConfig::long();

    assert_eq!(config.max_tokens, 100);
    assert!(config.timeout_secs >= 60);
}

#[test]
fn test_prompts_not_empty() {
    assert!(!TestPrompts::simple().is_empty());
    assert!(!TestPrompts::short().is_empty());
    assert!(!TestPrompts::long().is_empty());
    assert!(!TestPrompts::json().is_empty());
    assert!(!TestPrompts::with_stop().is_empty());
}

#[test]
fn test_make_test_request() {
    let req = make_test_request("test-1", "Hello world", 50);

    assert_eq!(req.job_id, "test-1");
    assert_eq!(req.prompt, "Hello world");
    assert_eq!(req.max_tokens, 50);
    assert_eq!(req.temperature, 0.7);
    assert_eq!(req.seed, Some(42));
    assert_eq!(req.top_p, 1.0);
    assert_eq!(req.top_k, 0);
    assert_eq!(req.repetition_penalty, 1.0);
    assert!(req.stop.is_empty());
    assert_eq!(req.min_p, 0.0);
}

// ============================================================================
// Framework Tests (require worker binary)
// ============================================================================

// Note: These tests require the worker binary to be built.
// They are marked with #[ignore] by default and run with:
// cargo test --test integration_framework_test -- --ignored

#[tokio::test]
#[ignore = "Requires worker binary"]
async fn test_harness_start_mock() {
    let harness = WorkerTestHarness::start_mock().await.unwrap();

    // Verify health endpoint
    let health = harness.health().await.unwrap();
    assert!(health.get("status").is_some());

    // Harness automatically cleaned up on drop
}

#[tokio::test]
#[ignore = "Requires worker binary and model"]
async fn test_harness_start_with_model() {
    let model = TestModel::qwen2_5_0_5b();

    if !model.exists() {
        eprintln!("Skipping test: model not found at {:?}", model.path);
        return;
    }

    let harness = WorkerTestHarness::start(model.path.to_str().unwrap(), 0).await.unwrap();

    // Verify health endpoint
    let health = harness.health().await.unwrap();
    assert!(health.get("status").is_some());
}

#[tokio::test]
#[ignore = "Requires worker binary"]
async fn test_harness_execute_request() {
    let harness = WorkerTestHarness::start_mock().await.unwrap();

    let req = make_test_request("test-1", TestPrompts::simple(), 5);

    // Send execute request
    let response = harness.execute(req).await.unwrap();

    // Should get SSE stream
    assert_eq!(response.status(), 200);
    assert!(response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("text/event-stream"));
}

// ---
// Built by Foundation-Alpha 🏗️
