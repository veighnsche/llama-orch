//! Integration test helper functions
//!
//! Provides utilities for SSE parsing, event validation, and common assertions.
//!
//! # Spec References
//! - M0-W-1820: Integration test framework

use worker_http::sse::InferenceEvent;
use worker_http::validation::ExecuteRequest;
use std::time::Duration;
use tokio::time::timeout;

/// Error type for helper functions
#[derive(Debug, thiserror::Error)]
pub enum HelperError {
    #[error("SSE parsing failed: {0}")]
    ParseFailed(String),
    
    #[error("Event order validation failed: {0}")]
    OrderValidationFailed(String),
    
    #[error("Timeout waiting for events")]
    Timeout,
    
    #[error("Invalid event: {0}")]
    InvalidEvent(String),
}

/// Collect all SSE events from response stream
///
/// Parses SSE stream and deserializes events.
/// Times out after 30 seconds to prevent hanging tests.
///
/// # Arguments
///
/// * `response` - HTTP response with SSE stream
///
/// # Returns
///
/// Vector of parsed InferenceEvent objects
///
/// # Errors
///
/// Returns error if:
/// - Stream parsing fails
/// - JSON deserialization fails
/// - Timeout expires (30s)
pub async fn collect_sse_events(
    mut response: reqwest::Response,
) -> Result<Vec<InferenceEvent>, HelperError> {
    let mut events = Vec::new();
    let mut buffer = Vec::new();
    
    let result = timeout(Duration::from_secs(30), async {
        while let Some(chunk) = response.chunk().await.map_err(|e| HelperError::ParseFailed(e.to_string()))? {
            buffer.extend_from_slice(&chunk);
            
            // Convert to string for processing
            let text = String::from_utf8_lossy(&buffer);
            let mut text_str = text.to_string();
            
            // Process complete SSE messages
            loop {
                let pos = match text_str.find("\n\n") {
                    Some(p) => p,
                    None => break,
                };
                
                let message = text_str[..pos].to_string();
                text_str = text_str[pos + 2..].to_string();
                
                // Parse SSE message
                if let Some(event) = parse_sse_message(&message)? {
                    let is_terminal = event.is_terminal();
                    events.push(event);
                    
                    // Stop if terminal event
                    if is_terminal {
                        return Ok(events);
                    }
                }
            }
            
            // Update buffer with remaining text
            buffer = text_str.into_bytes();
        }
        
        Ok(events)
    })
    .await
    .map_err(|_| HelperError::Timeout)??;
    
    Ok(result)
}

/// Parse single SSE message
fn parse_sse_message(message: &str) -> Result<Option<InferenceEvent>, HelperError> {
    let mut data = None;
    
    for line in message.lines() {
        if let Some(d) = line.strip_prefix("data: ") {
            data = Some(d.to_string());
        }
    }
    
    if let Some(data_str) = data {
        let event: InferenceEvent = serde_json::from_str(&data_str)
            .map_err(|e| HelperError::ParseFailed(format!("JSON parse error: {}", e)))?;
        Ok(Some(event))
    } else {
        Ok(None)
    }
}

/// Assert that events follow expected order
///
/// Validates:
/// - First event is Started
/// - Zero or more Token events
/// - Last event is End or Error
/// - No events after terminal event
///
/// # Arguments
///
/// * `events` - Vector of events to validate
///
/// # Returns
///
/// Ok if order is valid, Err with description if invalid
pub fn assert_event_order(events: &[InferenceEvent]) -> Result<(), HelperError> {
    if events.is_empty() {
        return Err(HelperError::OrderValidationFailed(
            "No events received".to_string(),
        ));
    }
    
    // First event must be Started
    if !matches!(events[0], InferenceEvent::Started { .. }) {
        return Err(HelperError::OrderValidationFailed(format!(
            "First event must be Started, got {:?}",
            events[0]
        )));
    }
    
    // Last event must be terminal (End or Error)
    let last = events.last().unwrap();
    if !last.is_terminal() {
        return Err(HelperError::OrderValidationFailed(format!(
            "Last event must be terminal (End or Error), got {:?}",
            last
        )));
    }
    
    // Middle events should be Token or Metrics
    for (i, event) in events.iter().enumerate().skip(1).take(events.len() - 2) {
        match event {
            InferenceEvent::Token { .. } | InferenceEvent::Metrics { .. } => {}
            _ => {
                return Err(HelperError::OrderValidationFailed(format!(
                    "Event {} should be Token or Metrics, got {:?}",
                    i, event
                )));
            }
        }
    }
    
    Ok(())
}

/// Extract token strings from events
///
/// Filters Token events and extracts the token text.
///
/// # Arguments
///
/// * `events` - Vector of events
///
/// # Returns
///
/// Vector of token strings in order
pub fn extract_tokens(events: &[InferenceEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            InferenceEvent::Token { t, .. } => Some(t.clone()),
            _ => None,
        })
        .collect()
}

/// Extract end event from events
///
/// Finds and returns the End event if present.
pub fn extract_end_event(events: &[InferenceEvent]) -> Option<&InferenceEvent> {
    events.iter().find(|e| matches!(e, InferenceEvent::End { .. }))
}

/// Assert that response contains expected number of tokens
pub fn assert_token_count(events: &[InferenceEvent], expected: usize) -> Result<(), HelperError> {
    let tokens = extract_tokens(events);
    if tokens.len() != expected {
        return Err(HelperError::OrderValidationFailed(format!(
            "Expected {} tokens, got {}",
            expected,
            tokens.len()
        )));
    }
    Ok(())
}

/// Assert that inference completed successfully (not error/cancelled)
pub fn assert_successful_completion(events: &[InferenceEvent]) -> Result<(), HelperError> {
    use worker_http::sse::StopReason;
    
    let end_event = extract_end_event(events)
        .ok_or_else(|| HelperError::OrderValidationFailed("No End event found".to_string()))?;
    
    if let InferenceEvent::End { stop_reason, .. } = end_event {
        match stop_reason {
            StopReason::MaxTokens | StopReason::StopSequence => Ok(()),
            StopReason::Error => Err(HelperError::OrderValidationFailed(
                "Inference ended with error".to_string(),
            )),
            StopReason::Cancelled => Err(HelperError::OrderValidationFailed(
                "Inference was cancelled".to_string(),
            )),
        }
    } else {
        Err(HelperError::InvalidEvent("Expected End event".to_string()))
    }
}

/// Create test execute request with defaults
pub fn make_test_request(
    job_id: &str,
    prompt: &str,
    max_tokens: u32,
) -> ExecuteRequest {
    ExecuteRequest {
        job_id: job_id.to_string(),
        prompt: prompt.to_string(),
        max_tokens,
        temperature: 0.7,
        seed: Some(42),
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        stop: vec![],
        min_p: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use worker_http::sse::StopReason;

    fn make_started() -> InferenceEvent {
        InferenceEvent::Started {
            job_id: "test".to_string(),
            model: "test".to_string(),
            started_at: "2025-10-05T00:00:00Z".to_string(),
        }
    }

    fn make_token(t: &str, i: u32) -> InferenceEvent {
        InferenceEvent::Token {
            t: t.to_string(),
            i,
        }
    }

    fn make_end() -> InferenceEvent {
        InferenceEvent::End {
            tokens_out: 10,
            decode_time_ms: 1000,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        }
    }

    #[test]
    fn test_assert_event_order_valid() {
        let events = vec![
            make_started(),
            make_token("Hello", 0),
            make_token(" world", 1),
            make_end(),
        ];
        
        assert!(assert_event_order(&events).is_ok());
    }

    #[test]
    fn test_assert_event_order_empty() {
        let events = vec![];
        assert!(assert_event_order(&events).is_err());
    }

    #[test]
    fn test_assert_event_order_no_started() {
        let events = vec![
            make_token("Hello", 0),
            make_end(),
        ];
        
        assert!(assert_event_order(&events).is_err());
    }

    #[test]
    fn test_assert_event_order_no_terminal() {
        let events = vec![
            make_started(),
            make_token("Hello", 0),
        ];
        
        assert!(assert_event_order(&events).is_err());
    }

    #[test]
    fn test_extract_tokens() {
        let events = vec![
            make_started(),
            make_token("Hello", 0),
            make_token(" world", 1),
            make_token("!", 2),
            make_end(),
        ];
        
        let tokens = extract_tokens(&events);
        assert_eq!(tokens, vec!["Hello", " world", "!"]);
    }

    #[test]
    fn test_extract_tokens_none() {
        let events = vec![make_started(), make_end()];
        let tokens = extract_tokens(&events);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_assert_token_count() {
        let events = vec![
            make_started(),
            make_token("A", 0),
            make_token("B", 1),
            make_end(),
        ];
        
        assert!(assert_token_count(&events, 2).is_ok());
        assert!(assert_token_count(&events, 3).is_err());
    }

    #[test]
    fn test_assert_successful_completion() {
        let events = vec![
            make_started(),
            make_token("test", 0),
            make_end(),
        ];
        
        assert!(assert_successful_completion(&events).is_ok());
    }

    #[test]
    fn test_assert_successful_completion_error() {
        let events = vec![
            make_started(),
            InferenceEvent::Error {
                code: "TEST".to_string(),
                message: "test error".to_string(),
            },
        ];
        
        assert!(assert_successful_completion(&events).is_err());
    }

    #[test]
    fn test_make_test_request() {
        let req = make_test_request("test-1", "Hello", 100);
        
        assert_eq!(req.job_id, "test-1");
        assert_eq!(req.prompt, "Hello");
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.temperature, 0.7);
        assert_eq!(req.seed, Some(42));
    }
}

// ---
// Built by Foundation-Alpha 🏗️
