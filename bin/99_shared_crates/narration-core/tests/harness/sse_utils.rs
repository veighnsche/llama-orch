// TEAM-302: SSE testing utilities
//!
//! Utility functions for testing SSE streams and narration events.
//!
//! # Purpose
//!
//! Provides reusable helpers for:
//! - Collecting events from string channels
//! - Asserting event sequences
//! - Verifying narration event fields
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::harness::sse_utils;
//!
//! let events = sse_utils::collect_events_until_done(&mut rx, 5).await;
//! sse_utils::assert_sequence(&events, &["event1", "event2"]);
//! ```

use tokio::time::{timeout, Duration};

/// Collect events from string channel until [DONE] marker
///
/// TEAM-302: Useful for tests that work with raw SSE strings
///
/// # Arguments
///
/// * `rx` - String channel receiver
/// * `timeout_secs` - Maximum seconds to wait for events
///
/// # Returns
///
/// Vector of collected event strings (excluding [DONE] marker)
pub async fn collect_events_until_done(
    rx: &mut tokio::sync::mpsc::Receiver<String>,
    timeout_secs: u64,
) -> Vec<String> {
    let mut events = Vec::new();
    
    loop {
        match timeout(Duration::from_secs(timeout_secs), rx.recv()).await {
            Ok(Some(line)) => {
                if line.contains("[DONE]") {
                    break;
                }
                events.push(line);
            }
            Ok(None) => break, // Channel closed
            Err(_) => break,   // Timeout
        }
    }
    
    events
}

/// Assert event sequence matches expected pattern
///
/// TEAM-302: Verifies that events contain expected substrings in order
///
/// # Panics
///
/// Panics if:
/// - Event count doesn't match
/// - Any event doesn't contain expected substring
pub fn assert_sequence(events: &[String], expected: &[&str]) {
    assert_eq!(
        events.len(),
        expected.len(),
        "Event count mismatch. Got {} events, expected {}.\nGot: {:?}\nExpected: {:?}",
        events.len(),
        expected.len(),
        events,
        expected
    );
    
    for (i, (event, expected_substr)) in events.iter().zip(expected.iter()).enumerate() {
        assert!(
            event.contains(expected_substr),
            "Event {} '{}' doesn't contain '{}'",
            i,
            event,
            expected_substr
        );
    }
}

/// Assert narration event contains expected fields
///
/// TEAM-302: Flexible assertion for narration event fields
///
/// # Arguments
///
/// * `event` - The narration event to check
/// * `actor` - Expected actor (None to skip check)
/// * `action` - Expected action (None to skip check)
/// * `message` - Expected message substring (None to skip check)
///
/// # Panics
///
/// Panics if any specified field doesn't match
pub fn assert_event_contains(
    event: &observability_narration_core::output::sse_sink::NarrationEvent,
    actor: Option<&str>,
    action: Option<&str>,
    message: Option<&str>,
) {
    if let Some(expected_actor) = actor {
        assert_eq!(
            event.actor, expected_actor,
            "Actor mismatch: expected '{}', got '{}'",
            expected_actor, event.actor
        );
    }
    
    if let Some(expected_action) = action {
        assert_eq!(
            event.action, expected_action,
            "Action mismatch: expected '{}', got '{}'",
            expected_action, event.action
        );
    }
    
    if let Some(expected_msg) = message {
        assert!(
            event.human.contains(expected_msg),
            "Message '{}' doesn't contain '{}'",
            event.human,
            expected_msg
        );
    }
}

/// Assert event sequence contains all expected actions
///
/// TEAM-302: Verifies that all expected actions appear in event list
///
/// # Panics
///
/// Panics if any expected action is missing
pub fn assert_contains_actions(
    events: &[observability_narration_core::output::sse_sink::NarrationEvent],
    expected_actions: &[&str],
) {
    let actual_actions: Vec<&str> = events.iter().map(|e| e.action.as_str()).collect();
    
    for expected in expected_actions {
        assert!(
            actual_actions.contains(expected),
            "Expected action '{}' not found in events: {:?}",
            expected,
            actual_actions
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use observability_narration_core::output::sse_sink::NarrationEvent;

    // TEAM-302: Unit tests for utility functions
    
    #[test]
    fn test_assert_sequence_success() {
        let events = vec![
            "event with foo".to_string(),
            "event with bar".to_string(),
        ];
        
        // Should not panic
        assert_sequence(&events, &["foo", "bar"]);
    }
    
    #[test]
    #[should_panic(expected = "Event count mismatch")]
    fn test_assert_sequence_count_mismatch() {
        let events = vec!["event1".to_string()];
        assert_sequence(&events, &["event1", "event2"]);
    }
    
    #[test]
    #[should_panic(expected = "doesn't contain")]
    fn test_assert_sequence_content_mismatch() {
        let events = vec!["event with foo".to_string()];
        assert_sequence(&events, &["bar"]);
    }
    
    #[test]
    fn test_assert_event_contains_all_fields() {
        let event = NarrationEvent {
            formatted: "[test-actor] test-action   : test message".to_string(),
            actor: "test-actor".to_string(),
            action: "test-action".to_string(),
            target: "test-target".to_string(),
            human: "test message".to_string(),
            cute: None,
            story: None,
            correlation_id: None,
            job_id: None,
            emitted_by: None,
            emitted_at_ms: None,
        };
        
        // Should not panic
        assert_event_contains(&event, Some("test-actor"), Some("test-action"), Some("test message"));
    }
    
    #[test]
    fn test_assert_event_contains_partial() {
        let event = NarrationEvent {
            formatted: "[test-actor] test-action   : test message".to_string(),
            actor: "test-actor".to_string(),
            action: "test-action".to_string(),
            target: "test-target".to_string(),
            human: "test message".to_string(),
            cute: None,
            story: None,
            correlation_id: None,
            job_id: None,
            emitted_by: None,
            emitted_at_ms: None,
        };
        
        // Should not panic - only checking action
        assert_event_contains(&event, None, Some("test-action"), None);
    }
}
