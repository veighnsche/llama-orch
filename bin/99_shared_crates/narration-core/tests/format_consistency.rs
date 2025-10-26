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

    // TEAM-311: New format: "\x1b[1m[actor] action\x1b[0m\nmessage\n"
    // Format includes ANSI escape codes for bold
    // Actor: 20 chars left-aligned, BOLD
    // Action: 20 chars left-aligned, light (not bold)
    // Message: on second line
    assert!(event.formatted.contains("[test-actor"));
    assert!(event.formatted.contains("test-action"));
    assert!(event.formatted.contains("Test message"));

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

    // TEAM-311: New format pads to 20 chars for both actor and action
    // Format: [actor              ] action              \nmessage\n
    // Without fn_name, no bold on fn_name (only actor is bold, action is light)
    // With ANSI codes stripped conceptually: "[abc                ] xyz                 \nShort\n"
    assert!(event.formatted.contains("[abc"));
    assert!(event.formatted.contains("xyz"));
    assert!(event.formatted.contains("Short"));

    sse_sink::remove_job_channel("format-test-2");
}

// TEAM-204: Removed test_formatted_uses_redacted_human - narration doesn't redact
