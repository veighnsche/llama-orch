//! Integration tests for worker-http
//!
//! Tests complete HTTP workflows including validation, SSE streaming, and error handling.

use async_trait::async_trait;
use worker_common::{InferenceResult, SamplingConfig, StopReason};
use worker_http::{backend::InferenceBackend, sse::InferenceEvent, validation::ExecuteRequest};

// Mock backend for testing
struct MockBackend {
    healthy: bool,
    vram: u64,
}

impl MockBackend {
    fn new() -> Self {
        Self { healthy: true, vram: 8_000_000_000 }
    }

    fn unhealthy() -> Self {
        Self { healthy: false, vram: 0 }
    }
}

#[async_trait]
impl InferenceBackend for MockBackend {
    async fn execute(
        &self,
        _prompt: &str,
        config: &SamplingConfig,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        let tokens = vec!["Hello".to_string(), " world".to_string(), "!".to_string()];
        let token_ids = vec![100, 200, 300];

        Ok(InferenceResult::max_tokens(tokens, token_ids, config.seed, 100))
    }

    async fn cancel(&self, _job_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    fn vram_usage(&self) -> u64 {
        self.vram
    }

    fn is_healthy(&self) -> bool {
        self.healthy
    }
}

#[test]
fn test_execute_request_validation_workflow() {
    let valid_req = ExecuteRequest {
        job_id: "test-123".to_string(),
        prompt: "Hello world".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        seed: Some(42),
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
        stop: vec!["\n\n".to_string()],
        min_p: 0.05,
    };

    assert!(valid_req.validate().is_ok());
    assert!(valid_req.validate_all().is_ok());
}

#[test]
fn test_execute_request_multiple_validation_errors() {
    let invalid_req = ExecuteRequest {
        job_id: "".to_string(),
        prompt: "".to_string(),
        max_tokens: 0,
        temperature: 3.0,
        seed: Some(42),
        top_p: 1.5,
        top_k: 0,
        repetition_penalty: 3.0,
        stop: vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ],
        min_p: 1.5,
    };

    // validate() returns first error
    assert!(invalid_req.validate().is_err());

    // validate_all() returns all errors
    let result = invalid_req.validate_all();
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.errors.len() >= 7);
}

#[test]
fn test_sse_event_serialization_workflow() {
    // Started event
    let started = InferenceEvent::Started {
        job_id: "test-123".to_string(),
        model: "Qwen2.5-0.5B".to_string(),
        started_at: "2025-10-05T15:00:00Z".to_string(),
    };
    let json = serde_json::to_string(&started).unwrap();
    assert!(json.contains("\"type\":\"started\""));
    assert!(!started.is_terminal());

    // Token events
    let token = InferenceEvent::Token { t: "Hello".to_string(), i: 0 };
    let json = serde_json::to_string(&token).unwrap();
    assert!(json.contains("\"type\":\"token\""));
    assert!(!token.is_terminal());

    // End event
    let end = InferenceEvent::End {
        tokens_out: 10,
        decode_time_ms: 1000,
        stop_reason: StopReason::MaxTokens,
        stop_sequence_matched: None,
    };
    let json = serde_json::to_string(&end).unwrap();
    assert!(json.contains("\"type\":\"end\""));
    assert!(end.is_terminal());

    // Error event
    let error = InferenceEvent::Error {
        code: "INFERENCE_FAILED".to_string(),
        message: "Test error".to_string(),
    };
    let json = serde_json::to_string(&error).unwrap();
    assert!(json.contains("\"type\":\"error\""));
    assert!(error.is_terminal());
}

#[test]
fn test_complete_inference_workflow() {
    // 1. Create request
    let req = ExecuteRequest {
        job_id: "workflow-test".to_string(),
        prompt: "Write a haiku".to_string(),
        max_tokens: 50,
        temperature: 0.7,
        seed: Some(42),
        top_p: 0.9,
        top_k: 50,
        repetition_penalty: 1.1,
        stop: vec!["\n\n".to_string()],
        min_p: 0.05,
    };

    // 2. Validate request
    assert!(req.validate().is_ok());

    // 3. Convert to SamplingConfig
    let config = SamplingConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
        min_p: req.min_p,
        stop_sequences: vec![],
        stop_strings: req.stop.clone(),
        seed: req.seed.unwrap_or(42),
        max_tokens: req.max_tokens,
    };

    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.seed, 42);
}

#[test]
fn test_backend_health_check() {
    let healthy_backend = MockBackend::new();
    assert!(healthy_backend.is_healthy());
    assert_eq!(healthy_backend.vram_usage(), 8_000_000_000);

    let unhealthy_backend = MockBackend::unhealthy();
    assert!(!unhealthy_backend.is_healthy());
    assert_eq!(unhealthy_backend.vram_usage(), 0);
}

#[tokio::test]
async fn test_backend_execute() {
    let backend = MockBackend::new();
    let config = SamplingConfig {
        temperature: 0.7,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        min_p: 0.0,
        stop_sequences: vec![],
        stop_strings: vec![],
        seed: 42,
        max_tokens: 100,
    };

    let result = backend.execute("Hello", &config).await.unwrap();
    assert_eq!(result.tokens.len(), 3);
    assert_eq!(result.tokens[0], "Hello");
    assert_eq!(result.tokens[1], " world");
    assert_eq!(result.tokens[2], "!");
}

#[tokio::test]
async fn test_backend_cancel() {
    let backend = MockBackend::new();
    let result = backend.cancel("test-job").await;
    assert!(result.is_ok());
}

#[test]
fn test_validation_error_response_structure() {
    let req = ExecuteRequest {
        job_id: "".to_string(),
        prompt: "".to_string(),
        max_tokens: 0,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    };

    let result = req.validate_all();
    assert!(result.is_err());

    let errors = result.unwrap_err();
    assert!(errors.errors.len() >= 3);

    // Verify error structure
    for error in &errors.errors {
        assert!(!error.field.is_empty());
        assert!(!error.constraint.is_empty());
        assert!(!error.message.is_empty());
    }
}

#[test]
fn test_stop_reason_all_variants() {
    let reasons = vec![
        StopReason::MaxTokens,
        StopReason::Eos,
        StopReason::StopSequence,
        StopReason::Cancelled,
        StopReason::Error,
    ];

    for reason in reasons {
        let json = serde_json::to_string(&reason).unwrap();
        assert!(!json.is_empty());

        // Verify deserialization
        let _deserialized: StopReason = serde_json::from_str(&json).unwrap();
    }
}

#[test]
fn test_inference_event_ordering() {
    // Events should be emitted in order: Started -> Token* -> End/Error
    let events = vec![
        InferenceEvent::Started {
            job_id: "test".to_string(),
            model: "test".to_string(),
            started_at: "2025-10-05T15:00:00Z".to_string(),
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

    // Verify non-terminal events come before terminal
    for (i, event) in events.iter().enumerate() {
        if i < events.len() - 1 {
            assert!(!event.is_terminal(), "Non-final event should not be terminal");
        } else {
            assert!(event.is_terminal(), "Final event should be terminal");
        }
    }
}

#[test]
fn test_unicode_in_tokens() {
    let unicode_tokens = vec!["Hello", " 世界", "🌍", "مرحبا", "Привет"];

    for (i, token) in unicode_tokens.iter().enumerate() {
        let event = InferenceEvent::Token { t: token.to_string(), i: i as u32 };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(token) || json.contains("\\u"));

        // Verify deserialization preserves unicode
        let deserialized: InferenceEvent = serde_json::from_str(&json).unwrap();
        if let InferenceEvent::Token { t, .. } = deserialized {
            assert_eq!(t, *token);
        }
    }
}

#[test]
fn test_request_deserialization_with_defaults() {
    let minimal_json = r#"{
        "job_id": "test",
        "prompt": "Hello",
        "max_tokens": 100
    }"#;

    let req: ExecuteRequest = serde_json::from_str(minimal_json).unwrap();
    assert_eq!(req.job_id, "test");
    assert_eq!(req.prompt, "Hello");
    assert_eq!(req.max_tokens, 100);
    assert_eq!(req.temperature, 1.0);
    assert_eq!(req.top_p, 1.0);
    assert_eq!(req.top_k, 0);
    assert_eq!(req.repetition_penalty, 1.0);
    assert_eq!(req.min_p, 0.0);
    assert!(req.stop.is_empty());
}

#[test]
fn test_request_serialization_roundtrip() {
    let original = ExecuteRequest {
        job_id: "roundtrip-test".to_string(),
        prompt: "Test prompt".to_string(),
        max_tokens: 150,
        temperature: 0.8,
        seed: Some(999),
        top_p: 0.95,
        top_k: 40,
        repetition_penalty: 1.2,
        stop: vec!["END".to_string()],
        min_p: 0.02,
    };

    let json = serde_json::to_string(&original).unwrap();
    let deserialized: ExecuteRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(original.job_id, deserialized.job_id);
    assert_eq!(original.prompt, deserialized.prompt);
    assert_eq!(original.max_tokens, deserialized.max_tokens);
    assert_eq!(original.temperature, deserialized.temperature);
    assert_eq!(original.seed, deserialized.seed);
    assert_eq!(original.top_p, deserialized.top_p);
    assert_eq!(original.top_k, deserialized.top_k);
    assert_eq!(original.repetition_penalty, deserialized.repetition_penalty);
    assert_eq!(original.stop, deserialized.stop);
    assert_eq!(original.min_p, deserialized.min_p);
}

// ---
// Verified by Testing Team 🔍
