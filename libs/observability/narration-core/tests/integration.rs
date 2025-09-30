// Integration tests for narration-core
// Tests ORCH-3300..3312 requirements

use observability_narration_core::{narrate, CaptureAdapter, NarrationFields, RedactionPolicy, redact_secrets};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn test_narration_basic() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    narrate(NarrationFields {
        actor: "orchestratord",
        action: "admission",
        target: "session-abc123".to_string(),
        human: "Accepted request; queued at position 3".to_string(),
        correlation_id: Some("req-xyz".into()),
        session_id: Some("session-abc123".into()),
        pool_id: Some("default".into()),
        queue_position: Some(3),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].actor, "orchestratord");
    assert_eq!(captured[0].action, "admission");
    assert!(captured[0].human.contains("Accepted request"));
}

#[test]
fn test_correlation_id_propagation() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    let correlation_id = "req-123".to_string();

    // Simulate multi-service flow
    narrate(NarrationFields {
        actor: "orchestratord",
        action: "admission",
        target: "session-1".to_string(),
        human: "Accepted request".to_string(),
        correlation_id: Some(correlation_id.clone()),
        ..Default::default()
    });

    narrate(NarrationFields {
        actor: "pool-managerd",
        action: "spawn",
        target: "GPU0".to_string(),
        human: "Spawning engine".to_string(),
        correlation_id: Some(correlation_id.clone()),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 2);
    
    // Both events should have same correlation_id
    assert_eq!(captured[0].correlation_id, Some(correlation_id.clone()));
    assert_eq!(captured[1].correlation_id, Some(correlation_id));
}

#[test]
fn test_redaction_in_narration() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Authorization: Bearer secret123".to_string(),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    
    // Secret should be redacted
    assert!(!captured[0].human.contains("secret123"));
    assert!(captured[0].human.contains("[REDACTED]"));
}

#[test]
fn test_capture_adapter_assertions() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    narrate(NarrationFields {
        actor: "pool-managerd",
        action: "spawn",
        target: "GPU0".to_string(),
        human: "Spawning engine llamacpp-v1".to_string(),
        pool_id: Some("default".into()),
        correlation_id: Some("req-xyz".into()),
        ..Default::default()
    });

    // Test assertion helpers
    adapter.assert_includes("Spawning engine");
    adapter.assert_field("actor", "pool-managerd");
    adapter.assert_field("action", "spawn");
    adapter.assert_field("pool_id", "default");
    adapter.assert_correlation_id_present();
}

#[test]
fn test_full_field_taxonomy() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    narrate(NarrationFields {
        actor: "orchestratord",
        action: "admission",
        target: "session-123".to_string(),
        human: "Accepted request".to_string(),
        correlation_id: Some("req-xyz".into()),
        session_id: Some("session-123".into()),
        job_id: Some("job-456".into()),
        task_id: Some("task-789".into()),
        pool_id: Some("default".into()),
        replica_id: Some("r0".into()),
        queue_position: Some(3),
        predicted_start_ms: Some(420),
        engine: Some("llamacpp".into()),
        engine_version: Some("v1".into()),
        model_ref: Some("model0".into()),
        device: Some("GPU0".into()),
        tokens_in: Some(100),
        tokens_out: Some(50),
        decode_time_ms: Some(250),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].correlation_id, Some("req-xyz".into()));
    assert_eq!(captured[0].session_id, Some("session-123".into()));
    assert_eq!(captured[0].pool_id, Some("default".into()));
}

#[test]
fn test_legacy_human_function() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    #[allow(deprecated)]
    observability_narration_core::human("test", "action", "target", "message");

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].actor, "test");
    assert_eq!(captured[0].action, "action");
    assert_eq!(captured[0].human, "message");
}

#[test]
fn test_redaction_policy_custom() {
    let text = "Bearer token123 and api_key=secret456";
    
    // Default policy
    let redacted = redact_secrets(text, RedactionPolicy::default());
    assert!(redacted.contains("[REDACTED]"));
    
    // Custom replacement
    let policy = RedactionPolicy {
        replacement: "***".to_string(),
        ..Default::default()
    };
    let redacted = redact_secrets(text, policy);
    assert!(redacted.contains("***"));
}

#[test]
fn test_multiple_narrations() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    for i in 0..5 {
        narrate(NarrationFields {
            actor: "test",
            action: "test",
            target: format!("target-{}", i),
            human: format!("Message {}", i),
            ..Default::default()
        });
    }

    let captured = adapter.captured();
    assert_eq!(captured.len(), 5);
}

#[test]
fn test_clear_captured() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });

    assert_eq!(adapter.captured().len(), 1);
    
    adapter.clear();
    assert_eq!(adapter.captured().len(), 0);
}
// ===== Provenance Tests =====
// Tests audit trail and debugging metadata

#[test]
fn test_provenance_emitted_by() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    narrate(NarrationFields {
        actor: "orchestratord",
        action: "admission",
        target: "session-123".to_string(),
        human: "Accepted request".to_string(),
        emitted_by: Some("orchestratord@0.1.0".into()),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].emitted_by, Some("orchestratord@0.1.0".into()));
}

#[test]
fn test_provenance_timestamp() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    narrate(NarrationFields {
        actor: "pool-managerd",
        action: "spawn",
        target: "GPU0".to_string(),
        human: "Spawning engine".to_string(),
        emitted_by: Some("pool-managerd@0.1.0".into()),
        emitted_at_ms: Some(now),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].emitted_at_ms, Some(now));
}

#[test]
fn test_provenance_trace_context() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    narrate(NarrationFields {
        actor: "engine-provisioner",
        action: "build",
        target: "llamacpp-v1".to_string(),
        human: "Building engine".to_string(),
        trace_id: Some("trace-abc123".into()),
        span_id: Some("span-xyz789".into()),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].trace_id, Some("trace-abc123".into()));
}

#[test]
fn test_provenance_source_location() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        source_location: Some("provenance.rs:75".into()),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    // Source location is in fields but not captured (dev-only)
}

#[test]
fn test_provenance_assertion_helpers() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    narrate(NarrationFields {
        actor: "orchestratord",
        action: "admission",
        target: "session-123".to_string(),
        human: "Accepted".to_string(),
        emitted_by: Some("orchestratord@0.1.0".into()),
        emitted_at_ms: Some(1234567890),
        trace_id: Some("trace-123".into()),
        ..Default::default()
    });

    adapter.assert_field("emitted_by", "orchestratord@0.1.0");
    adapter.assert_field("trace_id", "trace-123");
    adapter.assert_provenance_present();
}

#[test]
fn test_multi_service_provenance() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    let trace_id = "trace-multi-service-123".to_string();

    // Orchestratord
    narrate(NarrationFields {
        actor: "orchestratord",
        action: "admission",
        target: "session-1".to_string(),
        human: "Accepted".to_string(),
        emitted_by: Some("orchestratord@0.1.0".into()),
        trace_id: Some(trace_id.clone()),
        ..Default::default()
    });

    // Pool-managerd
    narrate(NarrationFields {
        actor: "pool-managerd",
        action: "spawn",
        target: "GPU0".to_string(),
        human: "Spawning".to_string(),
        emitted_by: Some("pool-managerd@0.1.0".into()),
        trace_id: Some(trace_id.clone()),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 2);
    
    // Both have same trace_id
    assert_eq!(captured[0].trace_id, Some(trace_id.clone()));
    assert_eq!(captured[1].trace_id, Some(trace_id));
    
    // Different emitters
    assert_eq!(captured[0].emitted_by, Some("orchestratord@0.1.0".into()));
    assert_eq!(captured[1].emitted_by, Some("pool-managerd@0.1.0".into()));
}

#[test]
fn test_provenance_optional() {
    let adapter = CaptureAdapter::install();
    adapter.clear();

    // Narration without provenance should still work
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test".to_string(),
        ..Default::default()
    });

    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].emitted_by, None);
    assert_eq!(captured[0].emitted_at_ms, None);
    assert_eq!(captured[0].trace_id, None);
}
