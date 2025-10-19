//! Smoke test: Verify all public APIs work out-of-the-box for foundation engineers.
//!
//! This test simulates a real-world usage scenario covering:
//! - Builder pattern API
//! - Function-based API
//! - Auto-injection
//! - Correlation IDs
//! - HTTP context propagation
//! - Secret redaction
//! - Test capture adapter
//! - All constants and helpers

use observability_narration_core::{
    // Utilities
    current_timestamp_ms,
    // Correlation IDs
    generate_correlation_id,
    // HTTP helpers
    http::{extract_context_from_headers, inject_context_into_headers},
    // Core functions
    narrate,
    narrate_auto,
    narrate_warn,
    // Redaction
    redact_secrets,
    service_identity,
    validate_correlation_id,
    // Test support
    CaptureAdapter,
    // Builder
    Narration,
    // Types
    NarrationFields,
    RedactionPolicy,
    // Constants - Actions
    ACTION_ADMISSION,
    ACTION_CANCEL,
    ACTION_DISPATCH,
    ACTION_ENQUEUE,
    ACTION_HEARTBEAT_SEND,
    ACTION_INFERENCE_COMPLETE,
    ACTION_INFERENCE_START,
    ACTION_READY_CALLBACK,
    ACTION_SEAL,
    ACTION_SPAWN,
    ACTOR_INFERENCE_ENGINE,
    // Constants - Actors
    ACTOR_ORCHESTRATORD,
    ACTOR_POOL_MANAGERD,
    ACTOR_VRAM_RESIDENCY,
    ACTOR_WORKER_ORCD,
};
use serial_test::serial;

/// Smoke test: Builder pattern API
#[test]
#[serial(capture_adapter)]
fn smoke_builder_pattern() {
    let adapter = CaptureAdapter::install();
    let job_id = "job-123";
    let req_id = "req-abc";

    // Foundation engineer writes this:
    Narration::new(ACTOR_ORCHESTRATORD, ACTION_ENQUEUE, job_id)
        .human(format!("Enqueued job {job_id}"))
        .correlation_id(req_id)
        .emit();

    // Verify it works
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].actor, ACTOR_ORCHESTRATORD);
    assert_eq!(captured[0].action, ACTION_ENQUEUE);
    assert_eq!(captured[0].target, job_id);
    assert!(captured[0].human.contains("Enqueued"));
    assert_eq!(captured[0].correlation_id, Some(req_id.to_string()));
}

/// Smoke test: Function-based API with constants
#[test]
#[serial(capture_adapter)]
fn smoke_function_api() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer writes this:
    narrate(NarrationFields {
        actor: ACTOR_POOL_MANAGERD,
        action: ACTION_SPAWN,
        target: "GPU0".to_string(),
        human: "Spawning worker".to_string(),
        pool_id: Some("default".into()),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].actor, ACTOR_POOL_MANAGERD);
}

/// Smoke test: Auto-injection
#[test]
#[serial(capture_adapter)]
fn smoke_auto_injection() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer writes this:
    narrate_auto(NarrationFields {
        actor: ACTOR_WORKER_ORCD,
        action: ACTION_INFERENCE_START,
        target: "job-456".to_string(),
        human: "Starting inference".to_string(),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    // Auto-injection should add these:
    assert!(captured[0].emitted_by.is_some());
    assert!(captured[0].emitted_at_ms.is_some());
}

/// Smoke test: Error narration
#[test]
#[serial(capture_adapter)]
fn smoke_error_narration() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer writes this:
    Narration::new(ACTOR_POOL_MANAGERD, ACTION_SPAWN, "GPU0")
        .human("Failed to spawn worker")
        .error_kind("ResourceExhausted")
        .emit_error();

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].error_kind, Some("ResourceExhausted".to_string()));
}

/// Smoke test: Correlation ID generation and validation
#[test]
fn smoke_correlation_ids() {
    // Foundation engineer writes this:
    let correlation_id = generate_correlation_id();

    // Should be valid UUID
    assert!(validate_correlation_id(&correlation_id).is_some());

    // Invalid IDs should be rejected
    assert!(validate_correlation_id("invalid").is_none());
}

/// Smoke test: HTTP context propagation
#[test]
fn smoke_http_context() {
    use observability_narration_core::http::HeaderLike;
    use std::collections::HashMap;

    // Simple HeaderLike implementation for testing
    struct TestHeaders(HashMap<String, String>);

    impl HeaderLike for TestHeaders {
        fn get_str(&self, name: &str) -> Option<String> {
            self.0.get(name).cloned()
        }

        fn insert_str(&mut self, name: &str, value: &str) {
            self.0.insert(name.to_string(), value.to_string());
        }
    }

    // Foundation engineer writes this:
    let mut headers = TestHeaders(HashMap::new());
    headers
        .0
        .insert("X-Correlation-Id".to_string(), "550e8400-e29b-41d4-a716-446655440000".to_string());

    // Extract from incoming request
    let (correlation_id, _, _, _) = extract_context_from_headers(&headers);
    assert!(correlation_id.is_some());

    // Inject into outgoing request
    let mut outgoing = TestHeaders(HashMap::new());
    inject_context_into_headers(&mut outgoing, correlation_id.as_deref(), None, None, None);
    assert!(outgoing.0.contains_key("X-Correlation-Id"));
}

/// Smoke test: Secret redaction
#[test]
fn smoke_secret_redaction() {
    // Foundation engineer writes this:
    let text_with_secret = "Authorization: Bearer abc123xyz";
    let redacted = redact_secrets(text_with_secret, RedactionPolicy::default());

    // Secret should be redacted
    assert!(redacted.contains("[REDACTED]"));
    assert!(!redacted.contains("abc123xyz"));
}

/// Smoke test: All ID fields work
#[test]
#[serial(capture_adapter)]
fn smoke_all_id_fields() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer writes this:
    Narration::new(ACTOR_ORCHESTRATORD, ACTION_DISPATCH, "job-789")
        .human("Dispatching job")
        .correlation_id("req-xyz")
        .session_id("session-123")
        .job_id("job-789")
        .task_id("task-456")
        .pool_id("default")
        .replica_id("replica-1")
        .worker_id("worker-gpu0-r1")
        .emit();

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].correlation_id, Some("req-xyz".to_string()));
    assert_eq!(captured[0].session_id, Some("session-123".to_string()));
    assert_eq!(captured[0].job_id, Some("job-789".to_string()));
    assert_eq!(captured[0].task_id, Some("task-456".to_string()));
    assert_eq!(captured[0].pool_id, Some("default".to_string()));
    assert_eq!(captured[0].replica_id, Some("replica-1".to_string()));
    assert_eq!(captured[0].worker_id, Some("worker-gpu0-r1".to_string()));
}

/// Smoke test: Performance metrics fields
#[test]
#[serial(capture_adapter)]
fn smoke_performance_metrics() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer writes this:
    Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_COMPLETE, "job-999")
        .human("Completed inference")
        .duration_ms(150)
        .tokens_in(100)
        .tokens_out(50)
        .decode_time_ms(120)
        .emit();

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].duration_ms, Some(150));
    assert_eq!(captured[0].tokens_in, Some(100));
    assert_eq!(captured[0].tokens_out, Some(50));
    assert_eq!(captured[0].decode_time_ms, Some(120));
}

/// Smoke test: Engine/model context fields
#[test]
#[serial(capture_adapter)]
fn smoke_engine_context() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer writes this:
    Narration::new(ACTOR_INFERENCE_ENGINE, "load", "llama-7b")
        .human("Loading model")
        .engine("llamacpp")
        .engine_version("v1.2.3")
        .model_ref("llama-7b-q4")
        .device("GPU0")
        .emit();

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].engine, Some("llamacpp".to_string()));
    assert_eq!(captured[0].engine_version, Some("v1.2.3".to_string()));
    assert_eq!(captured[0].model_ref, Some("llama-7b-q4".to_string()));
    assert_eq!(captured[0].device, Some("GPU0".to_string()));
}

/// Smoke test: Error context fields
#[test]
#[serial(capture_adapter)]
fn smoke_error_context() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer writes this:
    Narration::new(ACTOR_POOL_MANAGERD, ACTION_SPAWN, "GPU0")
        .human("Spawn failed, retrying")
        .error_kind("ResourceExhausted")
        .retry_after_ms(5000)
        .backoff_ms(2000)
        .emit_warn();

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].error_kind, Some("ResourceExhausted".to_string()));
    assert_eq!(captured[0].retry_after_ms, Some(5000));
    assert_eq!(captured[0].backoff_ms, Some(2000));
}

/// Smoke test: Queue context fields
#[test]
#[serial(capture_adapter)]
fn smoke_queue_context() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer writes this:
    Narration::new(ACTOR_ORCHESTRATORD, ACTION_ADMISSION, "session-abc")
        .human("Queued request")
        .queue_position(3)
        .predicted_start_ms(420)
        .emit();

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].queue_position, Some(3));
    assert_eq!(captured[0].predicted_start_ms, Some(420));
}

/// Smoke test: Story mode
#[test]
#[serial(capture_adapter)]
fn smoke_story_mode() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer writes this:
    Narration::new(ACTOR_ORCHESTRATORD, "request", "pool-managerd")
        .human("Requesting capacity")
        .story("\"Do you have room?\" asked orchestratord. \"Yes!\" replied pool-managerd.")
        .emit();

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert!(captured[0].story.is_some());
    assert!(captured[0].story.as_ref().unwrap().contains("asked orchestratord"));
}

/// Smoke test: Cute mode (requires feature)
#[test]
#[serial(capture_adapter)]
#[cfg(feature = "cute-mode")]
fn smoke_cute_mode() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer writes this:
    Narration::new(ACTOR_VRAM_RESIDENCY, ACTION_SEAL, "llama-7b")
        .human("Sealed model in VRAM")
        .cute("Tucked llama-7b into its cozy VRAM nest! 🛏️")
        .emit();

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert!(captured[0].cute.is_some());
}

/// Smoke test: Utility functions
#[test]
fn smoke_utilities() {
    // Foundation engineer writes this:
    let timestamp = current_timestamp_ms();
    assert!(timestamp > 0);

    let identity = service_identity();
    assert!(identity.contains("@"));
    assert!(identity.contains("observability-narration-core"));
}

/// Smoke test: Test capture adapter helpers
#[test]
#[serial(capture_adapter)]
fn smoke_capture_helpers() {
    let adapter = CaptureAdapter::install();

    Narration::new(ACTOR_ORCHESTRATORD, ACTION_ENQUEUE, "job-123")
        .human("Enqueued job for processing")
        .correlation_id("req-abc")
        .emit();

    // Foundation engineer uses these helpers:
    adapter.assert_includes("Enqueued");
    adapter.assert_field("actor", ACTOR_ORCHESTRATORD);
    adapter.assert_field("correlation_id", "req-abc");
    adapter.assert_correlation_id_present();
    adapter.assert_provenance_present();
}

/// Smoke test: Multiple narrations in sequence
#[test]
#[serial(capture_adapter)]
fn smoke_multiple_narrations() {
    let adapter = CaptureAdapter::install();
    let correlation_id = generate_correlation_id();

    // Foundation engineer writes a request lifecycle:

    // 1. Request received
    Narration::new(ACTOR_ORCHESTRATORD, ACTION_ADMISSION, "session-123")
        .human("Received inference request")
        .correlation_id(&correlation_id)
        .emit();

    // 2. Job enqueued
    Narration::new(ACTOR_ORCHESTRATORD, ACTION_ENQUEUE, "job-456")
        .human("Enqueued job")
        .correlation_id(&correlation_id)
        .queue_position(1)
        .emit();

    // 3. Job dispatched
    Narration::new(ACTOR_ORCHESTRATORD, ACTION_DISPATCH, "job-456")
        .human("Dispatched to worker")
        .correlation_id(&correlation_id)
        .worker_id("worker-gpu0-r1")
        .emit();

    // 4. Inference started
    Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_START, "job-456")
        .human("Starting inference")
        .correlation_id(&correlation_id)
        .emit();

    // 5. Inference completed
    Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_COMPLETE, "job-456")
        .human("Completed inference")
        .correlation_id(&correlation_id)
        .duration_ms(150)
        .tokens_in(100)
        .tokens_out(50)
        .emit();

    // Verify complete lifecycle captured
    let captured = adapter.captured();
    assert_eq!(captured.len(), 5);

    // All events should have the same correlation ID
    for event in &captured {
        assert_eq!(event.correlation_id, Some(correlation_id.clone()));
    }
}

/// Smoke test: Error handling flow
#[test]
#[serial(capture_adapter)]
fn smoke_error_flow() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer handles an error:
    Narration::new(ACTOR_POOL_MANAGERD, ACTION_SPAWN, "GPU0")
        .human("Failed to spawn: insufficient VRAM")
        .error_kind("ResourceExhausted")
        .retry_after_ms(5000)
        .emit_error();

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert!(captured[0].human.contains("Failed"));
    assert!(captured[0].error_kind.is_some());
}

/// Smoke test: Secret redaction in narration
#[test]
#[serial(capture_adapter)]
fn smoke_secret_redaction_in_narration() {
    let adapter = CaptureAdapter::install();
    let secret_token = "abc123xyz";

    // Foundation engineer accidentally includes a secret:
    Narration::new(ACTOR_WORKER_ORCD, "auth", "api")
        .human(format!("Authenticating with Bearer {secret_token}"))
        .emit();

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    // Secret should be redacted
    assert!(!captured[0].human.contains(secret_token));
    assert!(captured[0].human.contains("[REDACTED]"));
}

/// Smoke test: All action constants exist
#[test]
fn smoke_action_constants() {
    // Foundation engineer can use all these:
    let _actions = [
        ACTION_ADMISSION,
        ACTION_ENQUEUE,
        ACTION_DISPATCH,
        ACTION_SPAWN,
        ACTION_READY_CALLBACK,
        ACTION_HEARTBEAT_SEND,
        ACTION_INFERENCE_START,
        ACTION_INFERENCE_COMPLETE,
        ACTION_CANCEL,
        ACTION_SEAL,
    ];
    // Just verify they compile and are accessible
}

/// Smoke test: All actor constants exist
#[test]
fn smoke_actor_constants() {
    // Foundation engineer can use all these:
    let _actors = [
        ACTOR_ORCHESTRATORD,
        ACTOR_POOL_MANAGERD,
        ACTOR_WORKER_ORCD,
        ACTOR_INFERENCE_ENGINE,
        ACTOR_VRAM_RESIDENCY,
    ];
    // Just verify they compile and are accessible
}

/// Smoke test: Redaction policy customization
#[test]
fn smoke_custom_redaction_policy() {
    // Foundation engineer customizes redaction:
    let policy = RedactionPolicy {
        mask_bearer_tokens: true,
        mask_api_keys: true,
        mask_jwt_tokens: true,
        mask_private_keys: true,
        mask_url_passwords: true,
        mask_uuids: false, // Usually safe
        replacement: "[REDACTED]".to_string(),
    };

    let text = "Bearer abc123";
    let redacted = redact_secrets(text, policy);
    assert!(redacted.contains("[REDACTED]"));
}

/// Smoke test: Builder with all optional fields
#[test]
#[serial(capture_adapter)]
fn smoke_builder_all_fields() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer uses every field:
    Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_COMPLETE, "job-999")
        .human("Completed inference successfully")
        .correlation_id("req-xyz")
        .session_id("session-abc")
        .job_id("job-999")
        .task_id("task-111")
        .pool_id("default")
        .replica_id("replica-2")
        .worker_id("worker-gpu0-r1")
        .duration_ms(150)
        .tokens_in(100)
        .tokens_out(50)
        .decode_time_ms(120)
        .engine("llamacpp")
        .engine_version("v1.2.3")
        .model_ref("llama-7b")
        .device("GPU0")
        .source_location("main.rs:42")
        .emit();

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    // Spot check a few fields
    assert_eq!(captured[0].job_id, Some("job-999".to_string()));
    assert_eq!(captured[0].duration_ms, Some(150));
    assert_eq!(captured[0].engine, Some("llamacpp".to_string()));
}

/// Smoke test: Warn level narration
#[test]
#[serial(capture_adapter)]
fn smoke_warn_level() {
    let adapter = CaptureAdapter::install();

    // Foundation engineer emits a warning:
    narrate_warn(NarrationFields {
        actor: ACTOR_POOL_MANAGERD,
        action: ACTION_HEARTBEAT_SEND,
        target: "worker-gpu0-r1".to_string(),
        human: "Heartbeat timeout, worker may be unresponsive".to_string(),
        worker_id: Some("worker-gpu0-r1".into()),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
}

/// Smoke test: Documentation examples compile
#[test]
#[serial(capture_adapter)]
fn smoke_readme_examples() {
    let adapter = CaptureAdapter::install();

    // Example from README - Builder pattern:
    let job_id = "job-123";
    let req_id = "req-abc";
    let pool_id = "default";

    Narration::new(ACTOR_ORCHESTRATORD, ACTION_ENQUEUE, job_id)
        .human(format!("Enqueued job {job_id}"))
        .correlation_id(req_id)
        .pool_id(pool_id)
        .emit();

    // Example from README - Function-based:
    narrate(NarrationFields {
        actor: ACTOR_ORCHESTRATORD,
        action: ACTION_ENQUEUE,
        target: job_id.to_string(),
        human: format!("Enqueued job {job_id} for pool {pool_id}"),
        correlation_id: Some(req_id.to_string()),
        pool_id: Some(pool_id.to_string()),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 2);
}
