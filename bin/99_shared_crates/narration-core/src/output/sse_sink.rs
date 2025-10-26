//! SSE sink for distributed narration transport.
//!
//! Allows narration events to be sent over Server-Sent Events (SSE) channels
//! for remote observability in distributed systems.
//!
//! # Architecture
//!
//! This module implements job-scoped SSE channels for secure, isolated narration streaming:
//!
//! 1. **Job Isolation**: Each job gets its own MPSC channel (sender + receiver)
//! 2. **Fail-Fast Security**: Events without job_id are dropped (prevents privacy leaks)
//! 3. **Single Receiver**: MPSC semantics ensure only one consumer per job
//! 4. **Automatic Cleanup**: Channels are removed when jobs complete
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! // 1. Create job channel (in job_router before execution)
//! sse_sink::create_job_channel(job_id.clone(), 1000);
//!
//! // 2. Take receiver (in SSE endpoint handler)
//! let mut rx = sse_sink::take_job_receiver(&job_id)?;
//!
//! // 3. Stream events to client
//! while let Some(event) = rx.recv().await {
//!     send_sse(&event.formatted).await?;
//! }
//!
//! // 4. Cleanup (when job completes)
//! sse_sink::remove_job_channel(&job_id);
//! ```
//!
//! # Security
//!
//! CRITICAL: Global channels are a privacy hazard. Inference data from Job A
//! could leak to subscribers of Job B. This module enforces job isolation.

use crate::NarrationFields;
use crate::format::format_message; // TEAM-310: Use centralized formatting
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

// ============================================================================
// CONSTANTS
// ============================================================================
// TEAM-276: Extract magic numbers for maintainability

/// Default channel capacity for job-scoped SSE streams
///
/// TEAM-276: Currently unused but defined for future API improvements
/// (e.g., create_job_channel_with_default_capacity). Kept for documentation.
#[allow(dead_code)]
const DEFAULT_CHANNEL_CAPACITY: usize = 1000;

// ============================================================================
// DEPRECATED CONSTANTS (TEAM-310)
// ============================================================================
// 
// ⚠️ DEPRECATED: ACTOR_WIDTH and ACTION_WIDTH moved to format.rs module
// 
// Old location (pre-TEAM-310):
//   const ACTOR_WIDTH: usize = 10;
//   const ACTION_WIDTH: usize = 15;
// 
// New location (TEAM-310):
//   observability_narration_core::format::ACTOR_WIDTH (now 20)
//   observability_narration_core::format::ACTION_WIDTH (now 20)
// 
// Migration: Import from format module instead:
//   use observability_narration_core::format::{ACTOR_WIDTH, ACTION_WIDTH};
// 
// See: TEAM_310_FORMAT_MODULE.md for full migration guide

// ============================================================================
// GLOBAL REGISTRY
// ============================================================================

/// Global SSE channel registry with job-scoped channels.
///
/// TEAM-200: Refactored to support per-job channels
/// TEAM-204: CRITICAL SECURITY FIX - Removed global channel (privacy hazard)
/// TEAM-262: Renamed from SSE_BROADCASTER to SSE_CHANNEL_REGISTRY
static SSE_CHANNEL_REGISTRY: once_cell::sync::Lazy<SseChannelRegistry> =
    once_cell::sync::Lazy::new(SseChannelRegistry::new);

// ============================================================================
// CORE TYPES
// ============================================================================

/// Job-scoped SSE channel registry.
///
/// TEAM-204: SECURITY FIX - Job-scoped channels ONLY. No global channel.
/// If narration has no job_id, it's dropped (fail-fast).
///
/// TEAM-205: SIMPLIFIED - Use MPSC instead of broadcast to eliminate race conditions.
/// MPSC has simpler semantics (single receiver) and no "Closed" issues.
///
/// TEAM-262: RENAMED - "Broadcaster" was misleading (it's a registry of isolated channels)
/// TEAM-262: Renamed struct and static, updated all references (8 locations)
///
/// CRITICAL: Global channels are a privacy hazard - inference data
/// from Job A could leak to subscribers of Job B.
pub struct SseChannelRegistry {
    /// Per-job senders (keyed by job_id) - for emitting events
    senders: Arc<Mutex<HashMap<String, mpsc::Sender<NarrationEvent>>>>,
    /// Per-job receivers (keyed by job_id) - taken once by SSE handler
    receivers: Arc<Mutex<HashMap<String, mpsc::Receiver<NarrationEvent>>>>,
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
        // TEAM-204: No redaction needed - job isolation provides security
        // Developers need full context for debugging

        // TEAM-310: Use centralized formatting from format.rs module
        // 
        // ⚠️ DEPRECATED: Inline formatting removed (pre-TEAM-310)
        // Old code (REMOVED):
        //   let formatted = format!(
        //       "[{:<10}] {:<15}: {}",
        //       fields.actor, fields.action, fields.human
        //   );
        // 
        // New code (TEAM-310): Use centralized format_message()
        let formatted = format_message(fields.actor, fields.action, &fields.human);

        Self {
            formatted,
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target: fields.target,
            human: fields.human,
            cute: fields.cute,
            story: fields.story,
            correlation_id: fields.correlation_id,
            job_id: fields.job_id,
            emitted_by: fields.emitted_by,
            emitted_at_ms: fields.emitted_at_ms,
        }
    }
}

// ============================================================================
// REGISTRY IMPLEMENTATION
// ============================================================================

impl SseChannelRegistry {
    /// Create a new empty registry.
    fn new() -> Self {
        Self {
            senders: Arc::new(Mutex::new(HashMap::new())),
            receivers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new job-specific SSE channel.
    ///
    /// TEAM-200: Call this when a job is created (before execution starts).
    /// The job's SSE stream will be isolated from other jobs.
    ///
    /// TEAM-205: SIMPLIFIED - Use MPSC for predictable single-receiver semantics.
    pub fn create_job_channel(&self, job_id: String, capacity: usize) {
        let (tx, rx) = mpsc::channel(capacity);
        self.senders.lock().unwrap().insert(job_id.clone(), tx);
        self.receivers.lock().unwrap().insert(job_id, rx);
    }

    /// Remove a job's SSE channel (cleanup when job completes).
    ///
    /// TEAM-200: Call this when a job completes to prevent memory leaks.
    pub fn remove_job_channel(&self, job_id: &str) {
        self.senders.lock().unwrap().remove(job_id);
        self.receivers.lock().unwrap().remove(job_id);
    }

    /// Send narration to a specific job's SSE stream.
    ///
    /// TEAM-204: SECURITY FIX - FAIL FAST if job channel doesn't exist.
    /// Better to lose narration than leak it to wrong subscribers.
    ///
    /// TEAM-205: SIMPLIFIED - Use MPSC try_send (works in both sync and async contexts).
    ///
    /// # Behavior
    ///
    /// - If channel exists: Attempts to send event (may fail if full/closed)
    /// - If channel doesn't exist: **Drops event silently** (fail-fast security)
    /// - If channel is full: **Drops event silently** (backpressure)
    ///
    /// All failures are intentional - we prioritize security over completeness.
    pub fn send_to_job(&self, job_id: &str, event: NarrationEvent) {
        let senders = self.senders.lock().unwrap();
        if let Some(tx) = senders.get(job_id) {
            // TEAM-276: try_send works in both sync and async contexts
            // Failures (full/closed) are intentionally ignored
            let _ = tx.try_send(event);
        }
        // TEAM-276: If channel doesn't exist, DROP THE EVENT (fail-fast)
        // This is intentional - better to lose narration than leak sensitive data
    }

    /// Try to send narration to a specific job's SSE stream (returns success/failure).
    ///
    /// TEAM-298: Phase 1 - Opportunistic SSE delivery
    /// Returns true if sent successfully, false otherwise.
    ///
    /// # Behavior
    ///
    /// - Returns `true` if event was successfully sent to channel
    /// - Returns `false` if:
    ///   - Channel doesn't exist (not an error - stdout has the narration!)
    ///   - Channel is full (backpressure)
    ///   - Channel is closed (job completed)
    ///
    /// # Why This Matters
    ///
    /// Narration works in two modes:
    /// 1. **Primary**: Always emitted to stderr (guaranteed visibility)
    /// 2. **Bonus**: Sent to SSE if channel exists (remote observability)
    ///
    /// This function makes SSE truly optional - narration never fails.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Narration before channel creation (works!)
    /// n!("early", "This goes to stdout only");
    ///
    /// // Create channel
    /// sse_sink::create_job_channel(job_id, 1000);
    ///
    /// // Narration after channel creation (goes to both!)
    /// n!("later", "This goes to stdout AND SSE");
    /// ```
    pub fn try_send_to_job(&self, job_id: &str, event: NarrationEvent) -> bool {
        let senders = self.senders.lock().unwrap();
        if let Some(tx) = senders.get(job_id) {
            // Try to send, return true only if successful
            tx.try_send(event).is_ok()
        } else {
            // Channel doesn't exist - not an error, stdout has the narration
            false
        }
    }

    /// Take the receiver for a specific job's SSE stream.
    ///
    /// TEAM-200: Keeper calls this with job_id to get isolated stream.
    /// TEAM-205: SIMPLIFIED - Take receiver (can only be called once per job).
    ///
    /// This can only be called once per job - the receiver is moved out.
    pub fn take_job_receiver(&self, job_id: &str) -> Option<mpsc::Receiver<NarrationEvent>> {
        self.receivers.lock().unwrap().remove(job_id)
    }

    /// Check if a job channel exists.
    ///
    /// Returns `true` if a channel has been created for this job_id and not yet removed.
    pub fn has_job_channel(&self, job_id: &str) -> bool {
        self.senders.lock().unwrap().contains_key(job_id)
    }

    /// Get the number of active job channels (for monitoring/debugging).
    ///
    /// TEAM-276: Added for observability
    pub fn active_channel_count(&self) -> usize {
        self.senders.lock().unwrap().len()
    }
}

// TEAM-204: REMOVED global channel initialization (security fix)
// All narration is job-scoped or dropped (fail-fast)

// ============================================================================
// PUBLIC API
// ============================================================================

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
    SSE_CHANNEL_REGISTRY.create_job_channel(job_id, capacity);
}

/// Remove a job's SSE channel (cleanup).
///
/// TEAM-200: Call this when job completes to prevent memory leaks.
pub fn remove_job_channel(job_id: &str) {
    SSE_CHANNEL_REGISTRY.remove_job_channel(job_id);
}

/// Send a narration event to job-specific channel.
///
/// TEAM-204: SECURITY FIX - FAIL FAST, only job-scoped narration is sent.
/// If no job_id, event is DROPPED (intentional).
///
/// This prevents sensitive inference data from leaking to global subscribers.
///
/// # Security Guarantees
///
/// - **No job_id**: Event is dropped (fail-fast)
/// - **Invalid job_id**: Event is dropped (fail-fast)
/// - **Channel full**: Event is dropped (backpressure)
/// - **Receiver closed**: Event is dropped (cleanup)
///
/// All drops are intentional - we prioritize security over completeness.
pub fn send(fields: &NarrationFields) {
    // TEAM-276: Early return for clarity
    let Some(job_id) = &fields.job_id else {
        // SECURITY: No job_id = DROP (fail-fast, prevent privacy leaks)
        return;
    };

    let event = NarrationEvent::from(fields.clone());
    SSE_CHANNEL_REGISTRY.send_to_job(job_id, event);
}

/// Try to send a narration event to job-specific channel (returns success/failure).
///
/// TEAM-298: Phase 1 - Opportunistic SSE delivery
///
/// Returns `true` if event was successfully sent to SSE channel, `false` otherwise.
/// **Failure is not an error** - narration always goes to stdout regardless.
///
/// # Why Use This
///
/// This function makes the SSE delivery status visible to callers:
/// - `true`: Event delivered to SSE stream (remote observability working)
/// - `false`: Event not delivered to SSE (but stdout has it!)
///
/// # Behavior
///
/// Returns `false` if:
/// - No `job_id` in fields (security: prevent leaks)
/// - Channel doesn't exist (narration before channel creation - OK!)
/// - Channel is full (backpressure)
/// - Channel is closed (job already completed)
///
/// All `false` cases are valid - stdout always has the narration.
///
/// # Example
/// ```rust,ignore
/// use observability_narration_core::{n, sse_sink, NarrationContext, with_narration_context};
///
/// let job_id = "job-123";
/// let ctx = NarrationContext::new().with_job_id(job_id);
///
/// with_narration_context(ctx, async {
///     // Before channel creation (goes to stdout only)
///     n!("early", "Early narration");  // → stdout ✅, SSE ❌
///     
///     // Create channel
///     sse_sink::create_job_channel(job_id.to_string(), 1000);
///     
///     // After channel creation (goes to both!)
///     n!("later", "Later narration");  // → stdout ✅, SSE ✅
/// }).await;
/// ```
pub fn try_send(fields: &NarrationFields) -> bool {
    // TEAM-298: Early return if no job_id (security: prevent leaks)
    let Some(job_id) = &fields.job_id else {
        return false;  // No job_id = can't route to SSE
    };

    let event = NarrationEvent::from(fields.clone());
    SSE_CHANNEL_REGISTRY.try_send_to_job(job_id, event)
}

/// Take the receiver for a specific job's SSE stream.
///
/// TEAM-200: Keeper calls this with job_id from job creation response.
/// TEAM-205: SIMPLIFIED - Can only be called once per job.
///
/// # Example
/// ```rust,ignore
/// let mut rx = sse_sink::take_job_receiver("job-abc123")
///     .expect("Job channel not found");
/// while let Some(event) = rx.recv().await {
///     println!("{}", event.formatted);
/// }
/// ```
pub fn take_job_receiver(job_id: &str) -> Option<mpsc::Receiver<NarrationEvent>> {
    SSE_CHANNEL_REGISTRY.take_job_receiver(job_id)
}

/// Check if SSE broadcasting is enabled.
///
/// TEAM-200: Always true now (job-scoped channels are always available).
///
/// # Note
///
/// This function exists for backward compatibility. SSE channels are always
/// available - they're created on-demand per job.
pub fn is_enabled() -> bool {
    true
}

/// Check if a job channel exists.
///
/// Returns `true` if a channel has been created for this job_id and not yet removed.
pub fn has_job_channel(job_id: &str) -> bool {
    SSE_CHANNEL_REGISTRY.has_job_channel(job_id)
}

/// Get the number of active job channels (for monitoring/debugging).
///
/// TEAM-276: Added for observability
pub fn active_channel_count() -> usize {
    SSE_CHANNEL_REGISTRY.active_channel_count()
}

// TEAM-204: Removed obsolete redaction tests
// Redaction was a byproduct of the global channel security flaw
// With job isolation, no redaction needed - developers need full context for debugging

// ============================================================================
// TESTS
// ============================================================================

// TEAM-201: Formatting tests for centralized formatted field
#[cfg(test)]
mod team_201_formatting_tests {
    use super::*;
    use crate::NarrationFields;

    // TEAM-276: Test helper for creating minimal fields
    fn minimal_fields(actor: &'static str, action: &'static str, human: &str) -> NarrationFields {
        NarrationFields {
            actor,
            action,
            target: "test-target".to_string(),
            human: human.to_string(),
            ..Default::default()
        }
    }

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

        // Formatted field should match new format: bold first line, message on second line, trailing newline
        // Format: \x1b[1m[actor              ] action              \x1b[0m\nmessage\n
        // test-actor (10 chars) + 10 spaces = 20, test-action (11 chars) + 9 spaces = 20
        assert_eq!(event.formatted, "\x1b[1m[test-actor          ] test-action         \x1b[0m\nTest message\n");

        // Verify components
        assert!(event.formatted.contains("[test-actor"));
        assert!(event.formatted.contains("test-action"));
        assert!(event.formatted.contains("\nTest message"));
    }

    #[test]
    fn test_formatted_with_short_actor() {
        // TEAM-276: Use helper for cleaner test
        let fields = minimal_fields("abc", "xyz", "Short");
        let event = NarrationEvent::from(fields);

        // Should pad to ACTOR_WIDTH (20) chars for actor, ACTION_WIDTH (20) for action
        // TEAM-310: New format with bold, newline, and trailing newline
        assert_eq!(event.formatted, "\x1b[1m[abc                 ] xyz                 \x1b[0m\nShort\n");
    }

    #[test]
    fn test_formatted_with_long_actor() {
        // TEAM-276: Use helper for cleaner test
        let fields = minimal_fields("very-long-actor-name", "very-long-action-name", "Long");
        let event = NarrationEvent::from(fields);

        // Should not truncate - format! will extend if needed
        // TEAM-310: Verify bold codes and newline
        assert!(event.formatted.contains("very-long-actor-name"));
        assert!(event.formatted.contains("very-long-action-name"));
        assert!(event.formatted.contains("\nLong")); // Message on new line
        assert!(event.formatted.contains("\x1b[1m")); // Bold
        assert!(event.formatted.contains("\x1b[0m")); // Reset
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

    // TEAM-276: Test helper for creating job-scoped fields
    fn job_fields(job_id: &str, action: &'static str, message: &str) -> NarrationFields {
        NarrationFields {
            actor: "test",
            action,
            target: format!("target-{}", job_id),
            human: message.to_string(),
            job_id: Some(job_id.to_string()),
            ..Default::default()
        }
    }

    #[tokio::test]
    #[serial_test::serial(capture_adapter)]
    async fn test_job_isolation() {
        // Create two job channels (no global init needed)
        create_job_channel("job-a".to_string(), 100);
        create_job_channel("job-b".to_string(), 100);

        // Take receivers for both jobs
        let mut rx_a = take_job_receiver("job-a").unwrap();
        let mut rx_b = take_job_receiver("job-b").unwrap();

        // TEAM-276: Use helper for cleaner test
        send(&job_fields("job-a", "action_a", "Message for Job A"));
        send(&job_fields("job-b", "action_b", "Message for Job B"));

        // Job A should only receive its message
        let event_a = rx_a.recv().await.unwrap();
        assert_eq!(event_a.human, "Message for Job A");
        assert!(rx_a.try_recv().is_err()); // No more messages

        // Job B should only receive its message
        let event_b = rx_b.recv().await.unwrap();
        assert_eq!(event_b.human, "Message for Job B");
        assert!(rx_b.try_recv().is_err()); // No more messages

        // Cleanup
        remove_job_channel("job-a");
        remove_job_channel("job-b");
    }

    // TEAM-204: REMOVED global channel test - global channels are a privacy hazard

    #[test]
    #[serial_test::serial(capture_adapter)]
    fn test_channel_cleanup() {
        create_job_channel("job-temp".to_string(), 100);
        assert!(has_job_channel("job-temp"));

        remove_job_channel("job-temp");
        assert!(!has_job_channel("job-temp"));
    }

    #[tokio::test]
    #[serial_test::serial(capture_adapter)]
    async fn test_send_to_nonexistent_job_drops_event() {
        // TEAM-204: SECURITY FIX - When job channel doesn't exist, event is DROPPED (fail-fast)
        // This prevents sensitive data from leaking to wrong subscribers

        // TEAM-276: Use helper for cleaner test
        send(&job_fields("nonexistent-job", "test", "Test message for nonexistent job"));

        // Event is dropped - this is intentional and correct
        // Better to lose narration than leak sensitive inference data
        assert!(!has_job_channel("nonexistent-job"));
    }

    #[tokio::test]
    #[serial_test::serial(capture_adapter)]
    async fn test_race_condition_narration_before_channel_creation() {
        // TEAM-204: SECURITY FIX - Race condition where narration happens before create_job_channel()
        // Event is DROPPED (fail-fast) - this is correct behavior

        let job_id = "race-condition-job";

        // 1. Emit narration (job channel doesn't exist yet!)
        // TEAM-276: Use helper for cleaner test
        send(&job_fields(job_id, "early_narration", "This happened before channel was created!"));

        // 2. Event is DROPPED (no channel exists) - this is intentional
        assert!(!has_job_channel(job_id));

        // 3. Now create the job channel
        create_job_channel(job_id.to_string(), 100);
        let mut job_rx = take_job_receiver(job_id).unwrap();

        // 4. Future narration should go to job channel
        // TEAM-276: Use helper for cleaner test
        send(&job_fields(job_id, "later_narration", "This happened after channel was created!"));

        // 5. This event should be in JOB channel
        let event2 = job_rx.recv().await.expect("Should be in job channel");
        assert_eq!(event2.human, "This happened after channel was created!");

        remove_job_channel(job_id);
    }
}

// ============================================================================
// TEAM-276 REFACTORING SUMMARY
// ============================================================================
//
// **Improvements Made:**
//
// 1. **Enhanced Documentation**
//    - Added comprehensive module-level docs with architecture overview
//    - Added usage pattern examples
//    - Improved inline documentation for complex logic
//    - Added security guarantees documentation
//
// 2. **Code Organization**
//    - Grouped code into logical sections with clear headers
//    - Constants section for magic numbers (ACTOR_WIDTH, ACTION_WIDTH, DEFAULT_CHANNEL_CAPACITY)
//    - Public API section for exported functions
//    - Tests section with clear module boundaries
//
// 3. **Maintainability**
//    - Extracted magic numbers into named constants
//    - Added test helper functions (minimal_fields, job_fields)
//    - Improved error handling documentation
//    - Added active_channel_count() for observability
//
// 4. **Code Quality**
//    - Used early returns for clarity (send function)
//    - Named format width parameters for readability
//    - Improved test readability with helper functions
//    - Enhanced behavior documentation (send_to_job)
//
// **Historical Context Preserved:**
// - TEAM-200: Job-scoped channels foundation
// - TEAM-201: Centralized formatting
// - TEAM-204: Security fixes (fail-fast, no global channel)
// - TEAM-205: MPSC simplification
// - TEAM-262: Naming improvements (SSE_CHANNEL_REGISTRY)
//
// **No Breaking Changes:**
// - All public API preserved
// - All tests still pass
// - All historical team comments preserved
// - Backward compatibility maintained
//
// ============================================================================
