//! Integration test for TEAM-201's centralized formatting
//!
//! TEAM-203: Verify stderr and SSE formats match exactly
//! TEAM-276: Updated to use new API (take_job_receiver instead of subscribe_to_job)
//!
//! Created by: TEAM-203

use observability_narration_core::{sse_sink, NarrationFields};

#[tokio::test]
#[serial_test::serial(capture_adapter)]
async fn test_formatted_field_matches_stderr_format() {
    // TEAM-276: init() removed, create_job_channel() is sufficient
    sse_sink::create_job_channel("format-test".to_string(), 100);

    // TEAM-276: subscribe_to_job() → take_job_receiver()
    let mut rx = sse_sink::take_job_receiver("format-test").unwrap();

    let fields = NarrationFields {
        actor: "test-actor",
        action: "test-action",
        target: "target".to_string(),
        human: "Test message".to_string(),
        job_id: Some("format-test".to_string()),
        ..Default::default()
    };

    sse_sink::send(&fields);

    // TEAM-276: MPSC uses recv() not try_recv() for async
    let event = rx.recv().await.unwrap();

    // Formatted field should match: "[actor     ] action         : message"
    // Actor: 10 chars left-aligned
    // Action: 15 chars left-aligned
    assert!(event.formatted.starts_with("[test-actor]"));
    assert!(event.formatted.contains("test-action    :"));
    assert!(event.formatted.ends_with("Test message"));

    sse_sink::remove_job_channel("format-test");
}

#[tokio::test]
#[serial_test::serial(capture_adapter)]
async fn test_formatted_with_padding() {
    // TEAM-276: init() removed, create_job_channel() is sufficient
    sse_sink::create_job_channel("format-test-2".to_string(), 100);

    // TEAM-276: subscribe_to_job() → take_job_receiver()
    let mut rx = sse_sink::take_job_receiver("format-test-2").unwrap();

    let fields = NarrationFields {
        actor: "abc",
        action: "xyz",
        target: "test".to_string(),
        human: "Short".to_string(),
        job_id: Some("format-test-2".to_string()),
        ..Default::default()
    };

    sse_sink::send(&fields);

    // TEAM-276: MPSC uses recv() not try_recv() for async
    let event = rx.recv().await.unwrap();

    // Should pad to 10 chars for actor, 15 for action
    assert_eq!(event.formatted, "[abc       ] xyz            : Short");

    sse_sink::remove_job_channel("format-test-2");
}

// TEAM-204: Removed test_formatted_uses_redacted_human - narration doesn't redact
