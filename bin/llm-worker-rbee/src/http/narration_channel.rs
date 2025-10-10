//! Narration SSE channel management
//!
//! TEAM-039: Provides thread-local channel for narration events to flow into SSE stream
//! during inference requests.

use crate::http::sse::InferenceEvent;
use std::cell::RefCell;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

thread_local! {
    /// Thread-local storage for the current request's narration sender
    /// TEAM-039: Allows narrate() to send events into the SSE stream
    static NARRATION_SENDER: RefCell<Option<UnboundedSender<InferenceEvent>>> = RefCell::new(None);
}

/// Create a new narration channel and store the sender in thread-local storage
///
/// Returns the receiver that should be merged with the token stream.
/// The sender is automatically cleaned up when the receiver is dropped.
pub fn create_channel() -> UnboundedReceiver<InferenceEvent> {
    let (tx, rx) = unbounded_channel();

    NARRATION_SENDER.with(|sender| {
        *sender.borrow_mut() = Some(tx);
    });

    rx
}

/// Get the current narration sender (if in request context)
///
/// Returns None if not in a request context or if the channel was closed.
pub fn get_sender() -> Option<UnboundedSender<InferenceEvent>> {
    NARRATION_SENDER.with(|sender| sender.borrow().clone())
}

/// Clear the narration sender from thread-local storage
///
/// Should be called when the request completes to avoid memory leaks.
pub fn clear_sender() {
    NARRATION_SENDER.with(|sender| {
        *sender.borrow_mut() = None;
    });
}

/// Send a narration event to the current request's SSE stream (if active)
///
/// Returns true if the event was sent, false if no active request context.
pub fn send_narration(event: InferenceEvent) -> bool {
    if let Some(tx) = get_sender() {
        // Ignore send errors (channel closed is expected when request ends)
        tx.send(event).is_ok()
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_creation() {
        let mut rx = create_channel();

        // Sender should be available
        assert!(get_sender().is_some());

        // Send a test event
        let sent = send_narration(InferenceEvent::Narration {
            actor: "test".to_string(),
            action: "test".to_string(),
            target: "test".to_string(),
            human: "Test message".to_string(),
            cute: None,
            story: None,
            correlation_id: None,
            job_id: None,
        });
        assert!(sent);

        // Receive the event
        let event = rx.try_recv().unwrap();
        assert!(matches!(event, InferenceEvent::Narration { .. }));
    }

    #[test]
    fn test_clear_sender() {
        let _rx = create_channel();
        assert!(get_sender().is_some());

        clear_sender();
        assert!(get_sender().is_none());

        // Sending should fail after clear
        let sent = send_narration(InferenceEvent::Narration {
            actor: "test".to_string(),
            action: "test".to_string(),
            target: "test".to_string(),
            human: "Test".to_string(),
            cute: None,
            story: None,
            correlation_id: None,
            job_id: None,
        });
        assert!(!sent);
    }

    #[test]
    fn test_no_sender_by_default() {
        // Clear any existing sender from previous tests
        clear_sender();

        assert!(get_sender().is_none());

        let sent = send_narration(InferenceEvent::Narration {
            actor: "test".to_string(),
            action: "test".to_string(),
            target: "test".to_string(),
            human: "Test".to_string(),
            cute: None,
            story: None,
            correlation_id: None,
            job_id: None,
        });
        assert!(!sent);
    }
}

// ---
// Created by: TEAM-039 (narration plumbing implementation)
