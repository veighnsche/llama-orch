// capture.rs — Test capture adapter for BDD assertions
// Implements ORCH-3306: Test-time capture adapter

use crate::NarrationFields;
use std::sync::{Arc, Mutex, OnceLock};

/// Captured narration event for test assertions.
#[derive(Debug, Clone)]
pub struct CapturedNarration {
    pub actor: &'static str,
    pub action: &'static str,
    pub target: String,
    pub human: String,
    pub cute: Option<String>,
    pub story: Option<String>,
    pub correlation_id: Option<String>,
    pub session_id: Option<String>,
    pub pool_id: Option<String>,
    pub replica_id: Option<String>,
    // Provenance fields
    pub emitted_by: Option<String>,
    pub emitted_at_ms: Option<u64>,
    pub trace_id: Option<String>,
    pub parent_span_id: Option<String>,
    // Add more fields as needed for assertions
}

impl From<NarrationFields> for CapturedNarration {
    fn from(fields: NarrationFields) -> Self {
        Self {
            actor: fields.actor,
            action: fields.action,
            target: fields.target,
            human: fields.human,
            cute: fields.cute,
            story: fields.story,
            correlation_id: fields.correlation_id,
            session_id: fields.session_id,
            pool_id: fields.pool_id,
            replica_id: fields.replica_id,
            emitted_by: fields.emitted_by,
            emitted_at_ms: fields.emitted_at_ms,
            trace_id: fields.trace_id,
            parent_span_id: fields.parent_span_id,
        }
    }
}

/// Test capture adapter for collecting narration events.
/// Use in BDD tests to assert on narration presence and content.
///
/// # Example
/// ```rust
/// use observability_narration_core::{CaptureAdapter, narrate, NarrationFields};
///
/// // In test setup
/// let adapter = CaptureAdapter::install();
///
/// // Run code that emits narration
/// narrate(NarrationFields {
///     actor: "orchestratord",
///     action: "admission",
///     target: "session-123".to_string(),
///     human: "Accepted request".to_string(),
///     ..Default::default()
/// });
///
/// // Assert on captured narration
/// let captured = adapter.captured();
/// assert_eq!(captured.len(), 1);
/// assert_eq!(captured[0].actor, "orchestratord");
/// assert!(captured[0].human.contains("Accepted"));
/// ```
#[derive(Clone)]
pub struct CaptureAdapter {
    events: Arc<Mutex<Vec<CapturedNarration>>>,
}

impl CaptureAdapter {
    /// Create a new capture adapter.
    pub fn new() -> Self {
        Self { events: Arc::new(Mutex::new(Vec::new())) }
    }

    /// Install this adapter as the global capture target.
    /// Returns the adapter for later assertions.
    /// 
    /// Note: If an adapter is already installed, this will clear its events
    /// and return a reference to the existing adapter.
    pub fn install() -> Self {
        let adapter = GLOBAL_CAPTURE.get_or_init(|| Self::new()).clone();
        // Always clear events when installing to ensure clean state
        adapter.clear();
        adapter
    }

    /// Uninstall the global capture adapter.
    /// Call this in test teardown to avoid cross-test pollution.
    /// 
    /// Note: OnceLock doesn't support removal, so we clear the events instead.
    pub fn uninstall() {
        if let Some(adapter) = GLOBAL_CAPTURE.get() {
            adapter.clear();
        }
    }

    /// Capture a narration event.
    pub(crate) fn capture(&self, event: CapturedNarration) {
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
    }

    /// Get all captured events.
    pub fn captured(&self) -> Vec<CapturedNarration> {
        self.events
            .lock()
            .expect("BUG: capture adapter mutex poisoned - this indicates a panic in test code")
            .clone()
    }

    /// Clear all captured events.
    pub fn clear(&self) {
        if let Ok(mut events) = self.events.lock() {
            events.clear();
        }
    }

    /// Assert that narration includes a substring.
    pub fn assert_includes(&self, substring: &str) {
        let captured = self.captured();
        let found = captured.iter().any(|n| n.human.contains(substring));
        assert!(
            found,
            "Expected narration to include '{}', but found: {:?}",
            substring,
            captured.iter().map(|n| &n.human).collect::<Vec<_>>()
        );
    }

    /// Assert that a field is present with a specific value.
    pub fn assert_field(&self, field: &str, value: &str) {
        let captured = self.captured();
        let found = captured.iter().any(|n| match field {
            "actor" => n.actor == value,
            "action" => n.action == value,
            "target" => n.target == value,
            "pool_id" => n.pool_id.as_deref() == Some(value),
            "session_id" => n.session_id.as_deref() == Some(value),
            "correlation_id" => n.correlation_id.as_deref() == Some(value),
            "emitted_by" => n.emitted_by.as_deref() == Some(value),
            "trace_id" => n.trace_id.as_deref() == Some(value),
            _ => false,
        });
        assert!(
            found,
            "Expected field '{}' to equal '{}', but not found in: {:?}",
            field, value, captured
        );
    }

    /// Assert that provenance metadata is present.
    pub fn assert_provenance_present(&self) {
        let captured = self.captured();
        let found = captured.iter().any(|n| n.emitted_by.is_some() || n.emitted_at_ms.is_some());
        assert!(
            found,
            "Expected at least one narration with provenance (emitted_by or emitted_at_ms), but found none"
        );
    }

    /// Assert that a correlation ID is present.
    pub fn assert_correlation_id_present(&self) {
        let captured = self.captured();
        let found = captured.iter().any(|n| n.correlation_id.is_some());
        assert!(found, "Expected at least one narration with correlation_id, but found none");
    }

    /// Assert that cute narration is present.
    pub fn assert_cute_present(&self) {
        let captured = self.captured();
        let found = captured.iter().any(|n| n.cute.is_some());
        assert!(found, "Expected at least one narration with cute field, but found none");
    }

    /// Assert that cute narration includes a substring.
    pub fn assert_cute_includes(&self, substring: &str) {
        let captured = self.captured();
        let found =
            captured.iter().any(|n| n.cute.as_ref().map_or(false, |c| c.contains(substring)));
        assert!(
            found,
            "Expected cute narration to include '{}', but found: {:?}",
            substring,
            captured.iter().filter_map(|n| n.cute.as_ref()).collect::<Vec<_>>()
        );
    }

    /// Assert that story narration is present.
    pub fn assert_story_present(&self) {
        let captured = self.captured();
        let found = captured.iter().any(|n| n.story.is_some());
        assert!(found, "Expected at least one narration with story field, but found none");
    }

    /// Assert that story narration includes a substring.
    pub fn assert_story_includes(&self, substring: &str) {
        let captured = self.captured();
        let found =
            captured.iter().any(|n| n.story.as_ref().map_or(false, |s| s.contains(substring)));
        assert!(
            found,
            "Expected story narration to include '{}', but found: {:?}",
            substring,
            captured.iter().filter_map(|n| n.story.as_ref()).collect::<Vec<_>>()
        );
    }

    /// Assert that story narration includes dialogue (quoted text).
    pub fn assert_story_has_dialogue(&self) {
        let captured = self.captured();
        let found = captured
            .iter()
            .any(|n| n.story.as_ref().map_or(false, |s| s.contains('"') || s.contains("'")));
        assert!(
            found,
            "Expected story narration to include dialogue (quotes), but found: {:?}",
            captured.iter().filter_map(|n| n.story.as_ref()).collect::<Vec<_>>()
        );
    }
}

impl Default for CaptureAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Global capture adapter for test assertions.
static GLOBAL_CAPTURE: OnceLock<CaptureAdapter> = OnceLock::new();

/// Notify the global capture adapter of a narration event.
/// Called by `narrate()` when a capture adapter is installed.
pub(crate) fn notify(fields: NarrationFields) {
    if let Some(adapter) = GLOBAL_CAPTURE.get() {
        adapter.capture(fields.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_adapter_basic() {
        let adapter = CaptureAdapter::new();

        let fields = NarrationFields {
            actor: "test",
            action: "test_action",
            target: "test_target".to_string(),
            human: "Test message".to_string(),
            ..Default::default()
        };

        adapter.capture(fields.into());

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].actor, "test");
        assert_eq!(captured[0].human, "Test message");
    }

    #[test]
    fn test_assert_includes() {
        let adapter = CaptureAdapter::new();

        let fields = NarrationFields {
            actor: "test",
            action: "test_action",
            target: "test_target".to_string(),
            human: "Accepted request".to_string(),
            ..Default::default()
        };

        adapter.capture(fields.into());
        adapter.assert_includes("Accepted");
    }

    #[test]
    #[should_panic(expected = "Expected narration to include")]
    fn test_assert_includes_fails() {
        let adapter = CaptureAdapter::new();

        let fields = NarrationFields {
            actor: "test",
            action: "test_action",
            target: "test_target".to_string(),
            human: "Accepted request".to_string(),
            ..Default::default()
        };

        adapter.capture(fields.into());
        adapter.assert_includes("Rejected"); // Should panic
    }

    #[test]
    fn test_clear() {
        let adapter = CaptureAdapter::new();

        let fields = NarrationFields {
            actor: "test",
            action: "test_action",
            target: "test_target".to_string(),
            human: "Test".to_string(),
            ..Default::default()
        };

        adapter.capture(fields.into());
        assert_eq!(adapter.captured().len(), 1);

        adapter.clear();
        assert_eq!(adapter.captured().len(), 0);
    }
}
