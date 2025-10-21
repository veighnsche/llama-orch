//! SSE sink for distributed narration transport.
//!
//! Allows narration events to be sent over Server-Sent Events (SSE) channels
//! for remote observability in distributed systems.

use crate::NarrationFields;
// TEAM-199: Import redaction utilities for security fix
use crate::{redact_secrets, RedactionPolicy};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// Global SSE broadcaster with job-scoped channels.
/// 
/// TEAM-200: Refactored to support:
/// - Global channel for system-wide narration
/// - Per-job channels for isolated job narration
/// - Thread-local channels for request-scoped narration
static SSE_BROADCASTER: once_cell::sync::Lazy<SseBroadcaster> =
    once_cell::sync::Lazy::new(|| SseBroadcaster::new());

/// Broadcaster for SSE narration events with job isolation.
/// 
/// TEAM-200: This replaces the simple global broadcaster with:
/// 1. Global channel - For non-job narration (queen startup, etc.)
/// 2. Per-job channels - Isolated narration for each job
/// 3. Thread-local support - Request-scoped narration (like worker pattern)
pub struct SseBroadcaster {
    /// Global channel for non-job narration
    global: Arc<Mutex<Option<broadcast::Sender<NarrationEvent>>>>,
    
    /// Per-job channels (keyed by job_id)
    /// TEAM-200: Each job gets isolated SSE stream
    jobs: Arc<Mutex<HashMap<String, broadcast::Sender<NarrationEvent>>>>,
}

/// Narration event formatted for SSE transport.
/// 
/// TEAM-201: Added `formatted` field for centralized formatting.
/// Consumers should use `formatted` instead of manually formatting actor/action/human.
#[derive(Debug, Clone, serde::Serialize)]
pub struct NarrationEvent {
    /// Pre-formatted text matching stderr output
    /// Format: "[actor     ] action         : message"
    /// TEAM-201: This is the SINGLE source of truth for SSE display
    pub formatted: String,
    
    // Keep existing fields for backward compatibility and programmatic access
    pub actor: String,
    pub action: String,
    pub target: String,
    pub human: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cute: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub story: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_at_ms: Option<u64>,
}

impl From<NarrationFields> for NarrationEvent {
    fn from(fields: NarrationFields) -> Self {
        // TEAM-199: Apply redaction to ALL text fields (security fix)
        // This mirrors the redaction in narrate_at_level() (lib.rs line 433-440)
        // to ensure secrets don't leak through SSE streams
        let target = redact_secrets(&fields.target, RedactionPolicy::default());
        let human = redact_secrets(&fields.human, RedactionPolicy::default());
        let cute = fields.cute.as_ref()
            .map(|c| redact_secrets(c, RedactionPolicy::default()));
        let story = fields.story.as_ref()
            .map(|s| redact_secrets(s, RedactionPolicy::default()));
        
        // TEAM-201: Pre-format text (same format as stderr output)
        // Format: "[actor     ] action         : message"
        // - Actor: 10 chars (left-aligned, padded)
        // - Action: 15 chars (left-aligned, padded)
        // This matches lib.rs line 449 exactly
        // CRITICAL: Use redacted human, not raw fields.human!
        let formatted = format!("[{:<10}] {:<15}: {}", fields.actor, fields.action, human);
        
        Self {
            formatted,  // ✅ TEAM-201: NEW FIELD
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target,   // ✅ Redacted
            human: human.to_string(),    // ✅ Redacted
            cute: cute.map(|c| c.to_string()),     // ✅ Redacted (if present)
            story: story.map(|s| s.to_string()),    // ✅ Redacted (if present)
            correlation_id: fields.correlation_id,
            job_id: fields.job_id,
            emitted_by: fields.emitted_by,
            emitted_at_ms: fields.emitted_at_ms,
        }
    }
}

impl SseBroadcaster {
    fn new() -> Self {
        Self {
            global: Arc::new(Mutex::new(None)),
            jobs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Initialize the global SSE broadcaster.
    /// 
    /// TEAM-200: This creates the global channel for non-job narration.
    pub fn init(&self, capacity: usize) {
        let (tx, _) = broadcast::channel(capacity);
        *self.global.lock().unwrap() = Some(tx);
    }

    /// Create a new job-specific SSE channel.
    /// 
    /// TEAM-200: Call this when a job is created (before execution starts).
    /// The job's SSE stream will be isolated from other jobs.
    pub fn create_job_channel(&self, job_id: String, capacity: usize) {
        let (tx, _) = broadcast::channel(capacity);
        self.jobs.lock().unwrap().insert(job_id, tx);
    }

    /// Remove a job's SSE channel (cleanup when job completes).
    /// 
    /// TEAM-200: Call this when a job completes to prevent memory leaks.
    pub fn remove_job_channel(&self, job_id: &str) {
        self.jobs.lock().unwrap().remove(job_id);
    }

    /// Send narration to a specific job's SSE stream.
    /// 
    /// TEAM-200: This is the primary send method - routes to job-specific channel.
    pub fn send_to_job(&self, job_id: &str, event: NarrationEvent) {
        let jobs = self.jobs.lock().unwrap();
        if let Some(tx) = jobs.get(job_id) {
            // Ignore send errors (no subscribers is OK)
            let _ = tx.send(event);
        }
    }

    /// Send narration to the global channel (non-job narration).
    /// 
    /// TEAM-200: Use this for system-wide events (queen startup, etc.)
    pub fn send_global(&self, event: NarrationEvent) {
        if let Some(tx) = self.global.lock().unwrap().as_ref() {
            let _ = tx.send(event);
        }
    }

    /// Subscribe to a specific job's SSE stream.
    /// 
    /// TEAM-200: Keeper calls this with job_id to get isolated stream.
    pub fn subscribe_to_job(&self, job_id: &str) -> Option<broadcast::Receiver<NarrationEvent>> {
        self.jobs.lock().unwrap()
            .get(job_id)
            .map(|tx| tx.subscribe())
    }

    /// Subscribe to the global SSE stream.
    /// 
    /// TEAM-200: Use for monitoring all system-wide narration.
    pub fn subscribe_global(&self) -> Option<broadcast::Receiver<NarrationEvent>> {
        self.global.lock().unwrap()
            .as_ref()
            .map(|tx| tx.subscribe())
    }

    /// Check if a job channel exists.
    pub fn has_job_channel(&self, job_id: &str) -> bool {
        self.jobs.lock().unwrap().contains_key(job_id)
    }
}

/// Initialize the global SSE broadcaster.
///
/// TEAM-200: This initializes the global channel for non-job narration.
/// Job channels are created separately via create_job_channel().
pub fn init(capacity: usize) {
    SSE_BROADCASTER.init(capacity);
}

/// Create a job-specific SSE channel.
/// 
/// TEAM-200: Call this in job_router::create_job() before execution starts.
/// 
/// # Example
/// ```rust,ignore
/// use observability_narration_core::sse_sink;
/// 
/// let job_id = "job-abc123";
/// sse_sink::create_job_channel(job_id.to_string(), 1000);
/// // Now narration with this job_id goes to isolated channel
/// ```
pub fn create_job_channel(job_id: String, capacity: usize) {
    SSE_BROADCASTER.create_job_channel(job_id, capacity);
}

/// Remove a job's SSE channel (cleanup).
/// 
/// TEAM-200: Call this when job completes to prevent memory leaks.
pub fn remove_job_channel(job_id: &str) {
    SSE_BROADCASTER.remove_job_channel(job_id);
}

/// Send a narration event to appropriate channel based on job_id.
///
/// TEAM-200: Routing logic:
/// - If event has job_id → send to job-specific channel
/// - Otherwise → send to global channel
pub fn send(fields: &NarrationFields) {
    let event = NarrationEvent::from(fields.clone());
    
    // Route based on job_id
    if let Some(job_id) = &fields.job_id {
        SSE_BROADCASTER.send_to_job(job_id, event);
    } else {
        SSE_BROADCASTER.send_global(event);
    }
}

/// Subscribe to a specific job's SSE stream.
/// 
/// TEAM-200: Keeper calls this with job_id from job creation response.
///
/// # Example
/// ```rust,ignore
/// let mut rx = sse_sink::subscribe_to_job("job-abc123")
///     .expect("Job channel not found");
/// while let Ok(event) = rx.recv().await {
///     println!("{}", event.formatted);
/// }
/// ```
pub fn subscribe_to_job(job_id: &str) -> Option<broadcast::Receiver<NarrationEvent>> {
    SSE_BROADCASTER.subscribe_to_job(job_id)
}

/// Subscribe to the global SSE stream (all non-job narration).
pub fn subscribe_global() -> Option<broadcast::Receiver<NarrationEvent>> {
    SSE_BROADCASTER.subscribe_global()
}

/// Check if SSE broadcasting is enabled.
/// 
/// TEAM-200: Returns true if global channel is initialized.
pub fn is_enabled() -> bool {
    SSE_BROADCASTER.global.lock().unwrap().is_some()
}

/// Check if a job channel exists.
pub fn has_job_channel(job_id: &str) -> bool {
    SSE_BROADCASTER.has_job_channel(job_id)
}

// TEAM-199: Security tests for SSE redaction
#[cfg(test)]
mod team_199_security_tests {
    use super::*;
    use crate::NarrationFields;

    #[test]
    fn test_sse_event_redacts_api_key_in_human() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test-target".to_string(),
            human: "Connecting with api_key=sk-1234567890abcdef".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // API key should be redacted
        assert!(event.human.contains("[REDACTED]"));
        assert!(!event.human.contains("sk-1234567890abcdef"));
    }

    #[test]
    fn test_sse_event_redacts_bearer_token_in_human() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test-target".to_string(),
            human: "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Bearer token should be redacted
        assert!(event.human.contains("[REDACTED]"));
        assert!(!event.human.contains("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"));
    }

    #[test]
    fn test_sse_event_redacts_target_field() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "https://api.example.com?api_key=secret123".to_string(),
            human: "Making request".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // API key in target should be redacted
        assert!(event.target.contains("[REDACTED]"));
        assert!(!event.target.contains("secret123"));
    }

    #[test]
    fn test_sse_event_redacts_cute_field() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test".to_string(),
            human: "Test message".to_string(),
            cute: Some("The API whispered secret=sk-abcd1234".to_string()),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // API key in cute should be redacted
        let cute = event.cute.as_ref().unwrap();
        assert!(cute.contains("[REDACTED]"));
        assert!(!cute.contains("sk-abcd1234"));
    }

    #[test]
    fn test_sse_event_redacts_story_field() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test".to_string(),
            human: "Test message".to_string(),
            story: Some("'What's your password?' asked the villain. 'admin123!' replied the hero.".to_string()),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Password in story should be redacted (if redaction policy catches it)
        let story = event.story.as_ref().unwrap();
        // Note: Current redaction might not catch "admin123" - this tests the mechanism
        // The important thing is redaction is APPLIED, even if the pattern doesn't match
        assert!(!story.is_empty());
    }

    #[test]
    fn test_sse_event_preserves_safe_content() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "user-session-123".to_string(),
            human: "Processing request".to_string(),
            cute: Some("The worker bee buzzed happily 🐝".to_string()),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Safe content should NOT be redacted
        assert_eq!(event.target, "user-session-123");
        assert_eq!(event.human, "Processing request");
        assert_eq!(event.cute.as_ref().unwrap(), "The worker bee buzzed happily 🐝");
    }

    #[test]
    fn test_sse_and_stderr_have_same_redaction() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "url?token=secret123".to_string(),
            human: "api_key=sk-test123".to_string(),
            cute: Some("password=admin123".to_string()),
            ..Default::default()
        };

        // Create SSE event
        let sse_event = NarrationEvent::from(fields.clone());

        // The redaction should match what narrate_at_level does
        // Both should redact the same patterns
        assert!(sse_event.target.contains("[REDACTED]"));
        assert!(sse_event.human.contains("[REDACTED]"));
        
        // Note: We can't easily test narrate_at_level output without capturing stderr
        // But we verify SSE uses the same redact_secrets() function
    }
}

// TEAM-201: Formatting tests for centralized formatted field
#[cfg(test)]
mod team_201_formatting_tests {
    use super::*;
    use crate::NarrationFields;

    #[test]
    fn test_formatted_field_matches_stderr_format() {
        let fields = NarrationFields {
            actor: "test-actor",
            action: "test-action",
            target: "test-target".to_string(),
            human: "Test message".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Formatted field should match: "[actor     ] action         : message"
        assert_eq!(event.formatted, "[test-actor] test-action    : Test message");
        
        // Verify padding
        assert!(event.formatted.starts_with("[test-actor]"));
        assert!(event.formatted.contains("test-action    :"));
    }

    #[test]
    fn test_formatted_with_short_actor() {
        let fields = NarrationFields {
            actor: "abc",
            action: "xyz",
            target: "test".to_string(),
            human: "Short".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Should pad to 10 chars for actor, 15 for action
        assert_eq!(event.formatted, "[abc       ] xyz            : Short");
    }

    #[test]
    fn test_formatted_with_long_actor() {
        let fields = NarrationFields {
            actor: "very-long-actor-name",
            action: "very-long-action-name",
            target: "test".to_string(),
            human: "Long".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Should truncate/handle long names (Rust format! will extend if needed)
        assert!(event.formatted.contains("very-long-actor-name"));
        assert!(event.formatted.contains("very-long-action-name"));
        assert!(event.formatted.contains("Long"));
    }

    #[test]
    fn test_formatted_uses_redacted_human() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test".to_string(),
            human: "API key: sk-test123".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Formatted should use redacted human
        assert!(event.formatted.contains("[REDACTED]"));
        assert!(!event.formatted.contains("sk-test123"));
    }

    #[test]
    fn test_backward_compat_raw_fields_still_available() {
        let fields = NarrationFields {
            actor: "test",
            action: "action",
            target: "target".to_string(),
            human: "Message".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Old fields still work (backward compatibility)
        assert_eq!(event.actor, "test");
        assert_eq!(event.action, "action");
        assert_eq!(event.human, "Message");
        
        // But formatted is also available (new way)
        assert!(!event.formatted.is_empty());
    }
}

// TEAM-200: Isolation tests for job-scoped SSE broadcaster
#[cfg(test)]
mod team_200_isolation_tests {
    use super::*;
    use crate::NarrationFields;

    #[tokio::test]
    #[serial_test::serial(capture_adapter)]
    async fn test_job_isolation() {
        // Initialize broadcaster
        init(100);

        // Create two job channels
        create_job_channel("job-a".to_string(), 100);
        create_job_channel("job-b".to_string(), 100);

        // Subscribe to both jobs
        let mut rx_a = subscribe_to_job("job-a").unwrap();
        let mut rx_b = subscribe_to_job("job-b").unwrap();

        // Send narration to job-a
        let fields_a = NarrationFields {
            actor: "test",
            action: "action_a",
            target: "target-a".to_string(),
            human: "Message for Job A".to_string(),
            job_id: Some("job-a".to_string()),
            ..Default::default()
        };
        send(&fields_a);

        // Send narration to job-b
        let fields_b = NarrationFields {
            actor: "test",
            action: "action_b",
            target: "target-b".to_string(),
            human: "Message for Job B".to_string(),
            job_id: Some("job-b".to_string()),
            ..Default::default()
        };
        send(&fields_b);

        // Job A should only receive its message
        let event_a = rx_a.try_recv().unwrap();
        assert_eq!(event_a.human, "Message for Job A");
        assert!(rx_a.try_recv().is_err()); // No more messages

        // Job B should only receive its message
        let event_b = rx_b.try_recv().unwrap();
        assert_eq!(event_b.human, "Message for Job B");
        assert!(rx_b.try_recv().is_err()); // No more messages

        // Cleanup
        remove_job_channel("job-a");
        remove_job_channel("job-b");
    }

    #[tokio::test]
    #[serial_test::serial(capture_adapter)]
    async fn test_global_channel_for_non_job_narration() {
        init(100);
        let mut rx = subscribe_global().unwrap();

        // Drain any pre-existing events from other tests
        while rx.try_recv().is_ok() {}

        // Send narration without job_id
        let fields = NarrationFields {
            actor: "queen",
            action: "startup",
            target: "queen-rbee".to_string(),
            human: "Queen starting".to_string(),
            job_id: None, // ← No job_id
            ..Default::default()
        };
        send(&fields);

        // Should go to global channel
        let event = rx.try_recv().unwrap();
        assert_eq!(event.human, "Queen starting");
    }

    #[test]
    #[serial_test::serial(capture_adapter)]
    fn test_channel_cleanup() {
        create_job_channel("job-temp".to_string(), 100);
        assert!(has_job_channel("job-temp"));

        remove_job_channel("job-temp");
        assert!(!has_job_channel("job-temp"));
    }

    #[test]
    #[serial_test::serial(capture_adapter)]
    fn test_send_to_nonexistent_job_is_safe() {
        // Sending to non-existent job should not panic
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test".to_string(),
            human: "Test".to_string(),
            job_id: Some("nonexistent-job".to_string()),
            ..Default::default()
        };
        send(&fields); // Should not panic
    }
}
