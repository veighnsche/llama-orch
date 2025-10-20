//! Builder pattern for ergonomic narration API.
//!
//! Provides a fluent builder interface that reduces boilerplate and improves readability.
//!
//! # Example
//! ```rust
//! use observability_narration_core::Narration;
//!
//! Narration::new("orchestratord", "enqueue", "job-123")
//!     .human("Enqueued job for processing")
//!     .correlation_id("req-abc")
//!     .emit();
//! ```

use crate::NarrationFields;
use serde_json::Value;

/// Builder for constructing narration events with a fluent API.
///
/// # Example
/// ```rust
/// use observability_narration_core::Narration;
///
/// // Basic usage
/// Narration::new("worker-orcd", "execute", "job-123")
///     .human("Executing inference request")
///     .correlation_id("req-abc")
///     .duration_ms(150)
///     .emit();
///
/// // With error
/// Narration::new("pool-managerd", "spawn", "GPU0")
///     .human("Failed to spawn worker")
///     .error_kind("ResourceExhausted")
///     .emit_error();
/// ```
pub struct Narration {
    fields: NarrationFields,
}

impl Narration {
    /// Create a new narration builder with required fields.
    ///
    /// # Arguments
    /// - `actor`: Service name (use constants like `ACTOR_ORCHESTRATORD`)
    /// - `action`: Action performed (use constants like `ACTION_ENQUEUE`)
    /// - `target`: Target of the action (e.g., job ID, worker ID)
    ///
    /// # Example
    /// ```rust
    /// use observability_narration_core::{Narration, ACTOR_ORCHESTRATORD};
    ///
    /// let narration = Narration::new(ACTOR_ORCHESTRATORD, "enqueue", "job-123");
    /// ```
    pub fn new(actor: &'static str, action: &'static str, target: impl Into<String>) -> Self {
        Self {
            fields: NarrationFields {
                actor,
                action,
                target: target.into(),
                human: String::new(),
                ..Default::default()
            },
        }
    }

    /// Set the human-readable description.
    ///
    /// This field is automatically redacted for secrets.
    pub fn human(mut self, msg: impl Into<String>) -> Self {
        self.fields.human = msg.into();
        self
    }

    /// Set the correlation ID for request tracking.
    pub fn correlation_id(mut self, id: impl Into<String>) -> Self {
        self.fields.correlation_id = Some(id.into());
        self
    }

    /// Set the session ID.
    pub fn session_id(mut self, id: impl Into<String>) -> Self {
        self.fields.session_id = Some(id.into());
        self
    }

    /// Set the job ID.
    pub fn job_id(mut self, id: impl Into<String>) -> Self {
        self.fields.job_id = Some(id.into());
        self
    }

    /// Set the task ID.
    pub fn task_id(mut self, id: impl Into<String>) -> Self {
        self.fields.task_id = Some(id.into());
        self
    }

    /// Set the pool ID.
    pub fn pool_id(mut self, id: impl Into<String>) -> Self {
        self.fields.pool_id = Some(id.into());
        self
    }

    /// Set the replica ID.
    pub fn replica_id(mut self, id: impl Into<String>) -> Self {
        self.fields.replica_id = Some(id.into());
        self
    }

    /// Set the worker ID.
    pub fn worker_id(mut self, id: impl Into<String>) -> Self {
        self.fields.worker_id = Some(id.into());
        self
    }

    /// Set the hive ID.
    /// TEAM-185: Added for multi-hive rbee operations
    pub fn hive_id(mut self, id: impl Into<String>) -> Self {
        self.fields.hive_id = Some(id.into());
        self
    }

    /// Set the operation name (e.g., "worker_spawn", "infer", "model_download").
    /// This is for dynamic operation names in job-based systems.
    /// TEAM-185: Added for job-based systems to track dynamic operation names
    pub fn operation(mut self, op: impl Into<String>) -> Self {
        self.fields.operation = Some(op.into());
        self
    }

    /// Set the cute narration message (requires `cute-mode` feature).
    #[cfg(feature = "cute-mode")]
    pub fn cute(mut self, msg: impl Into<String>) -> Self {
        self.fields.cute = Some(msg.into());
        self
    }

    /// Set the story narration message.
    pub fn story(mut self, msg: impl Into<String>) -> Self {
        self.fields.story = Some(msg.into());
        self
    }

    /// Set the operation duration in milliseconds.
    pub fn duration_ms(mut self, ms: u64) -> Self {
        self.fields.duration_ms = Some(ms);
        self
    }

    /// Set the error kind/category.
    pub fn error_kind(mut self, kind: impl Into<String>) -> Self {
        self.fields.error_kind = Some(kind.into());
        self
    }

    /// Set the retry delay in milliseconds.
    pub fn retry_after_ms(mut self, ms: u64) -> Self {
        self.fields.retry_after_ms = Some(ms);
        self
    }

    /// Set the backoff duration in milliseconds.
    pub fn backoff_ms(mut self, ms: u64) -> Self {
        self.fields.backoff_ms = Some(ms);
        self
    }

    /// Set the queue position.
    pub fn queue_position(mut self, pos: usize) -> Self {
        self.fields.queue_position = Some(pos);
        self
    }

    /// Set the predicted start time in milliseconds.
    pub fn predicted_start_ms(mut self, ms: u64) -> Self {
        self.fields.predicted_start_ms = Some(ms);
        self
    }

    /// Set the engine name.
    pub fn engine(mut self, name: impl Into<String>) -> Self {
        self.fields.engine = Some(name.into());
        self
    }

    /// Set the engine version.
    pub fn engine_version(mut self, version: impl Into<String>) -> Self {
        self.fields.engine_version = Some(version.into());
        self
    }

    /// Set the model reference.
    pub fn model_ref(mut self, model: impl Into<String>) -> Self {
        self.fields.model_ref = Some(model.into());
        self
    }

    /// Set the device identifier.
    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.fields.device = Some(device.into());
        self
    }

    /// Set the input token count.
    pub fn tokens_in(mut self, count: u64) -> Self {
        self.fields.tokens_in = Some(count);
        self
    }

    /// Set the output token count.
    pub fn tokens_out(mut self, count: u64) -> Self {
        self.fields.tokens_out = Some(count);
        self
    }

    /// Set the decode time in milliseconds.
    pub fn decode_time_ms(mut self, ms: u64) -> Self {
        self.fields.decode_time_ms = Some(ms);
        self
    }

    /// Set the source location (file:line).
    pub fn source_location(mut self, location: impl Into<String>) -> Self {
        self.fields.source_location = Some(location.into());
        self
    }

    /// Format JSON as a CLI table and append to human message.
    ///
    /// The human/cute/story message becomes the title, and the table is appended below.
    ///
    /// - Arrays of objects → table with columns
    /// - Single objects → key-value table
    /// - Other values → pretty JSON
    ///
    /// # Example
    /// ```rust,ignore
    /// let json = serde_json::json!([{"id": "w1", "status": "ready"}]);
    /// Narration::new("rbee-keeper", "list", "workers")
    ///     .human("Workers:")
    ///     .table(&json)
    ///     .emit();
    /// ```
    pub fn table(mut self, json: &serde_json::Value) -> Self {
        let table_content = format_json_as_table(json);
        if !self.fields.human.is_empty() {
            self.fields.human.push_str("\n\n");
        }
        self.fields.human.push_str(&table_content);
        self
    }

    /// Emit the narration at INFO level with auto-injection.
    ///
    /// Automatically injects service identity and timestamp.
    ///
    /// Note: Use the `narrate!` macro instead to capture caller's crate name.
    pub fn emit(self) {
        crate::narrate_auto(self.fields)
    }

    /// Emit with explicit provenance (internal use by macro)
    #[doc(hidden)]
    pub fn emit_with_provenance(mut self, crate_name: &str, crate_version: &str) {
        if self.fields.emitted_by.is_none() {
            self.fields.emitted_by = Some(format!("{}@{}", crate_name, crate_version));
        }
        if self.fields.emitted_at_ms.is_none() {
            self.fields.emitted_at_ms = Some(crate::auto::current_timestamp_ms());
        }
        crate::narrate(self.fields)
    }

    /// Emit the narration at WARN level with auto-injection.
    pub fn emit_warn(self) {
        crate::narrate_warn(self.fields)
    }

    /// Emit the narration at ERROR level with auto-injection.
    pub fn emit_error(self) {
        crate::narrate_error(self.fields)
    }

    /// Emit the narration at DEBUG level with auto-injection.
    #[cfg(feature = "debug-enabled")]
    pub fn emit_debug(self) {
        crate::narrate_debug(self.fields)
    }

    /// Emit the narration at TRACE level with auto-injection.
    #[cfg(feature = "trace-enabled")]
    pub fn emit_trace(self) {
        crate::narrate_trace(self.fields)
    }
}

// ============================================================================
// Table Formatting
// ============================================================================

/// Format JSON as a CLI table.
fn format_json_as_table(json: &Value) -> String {
    match json {
        Value::Array(items) if !items.is_empty() => {
            // Check if all items are objects
            if items.iter().all(|v| v.is_object()) {
                format_array_table(items)
            } else {
                serde_json::to_string_pretty(json).unwrap_or_default()
            }
        }
        Value::Object(map) => format_object_table(map),
        _ => serde_json::to_string_pretty(json).unwrap_or_default(),
    }
}

/// Format array of objects as table with columns.
fn format_array_table(items: &[Value]) -> String {
    if items.is_empty() {
        return String::from("(empty)");
    }

    // Collect all unique keys across all objects
    let mut keys = std::collections::BTreeSet::new();
    for item in items {
        if let Value::Object(map) = item {
            keys.extend(map.keys().cloned());
        }
    }
    let keys: Vec<String> = keys.into_iter().collect();

    // Calculate column widths
    let mut widths: Vec<usize> = keys.iter().map(|k| k.len()).collect();
    for item in items {
        if let Value::Object(map) = item {
            for (i, key) in keys.iter().enumerate() {
                if let Some(val) = map.get(key) {
                    let val_str = format_value_compact(val);
                    widths[i] = widths[i].max(val_str.len());
                }
            }
        }
    }

    // Build header
    let mut output = String::new();
    let header: Vec<String> = keys
        .iter()
        .enumerate()
        .map(|(i, k)| format!("{:width$}", k, width = widths[i]))
        .collect();
    output.push_str(&header.join(" │ "));
    output.push('\n');
    
    // Separator
    let sep: Vec<String> = widths.iter().map(|w| "─".repeat(*w)).collect();
    output.push_str(&sep.join("─┼─"));
    output.push('\n');

    // Data rows
    for item in items {
        if let Value::Object(map) = item {
            let row: Vec<String> = keys
                .iter()
                .enumerate()
                .map(|(i, k)| {
                    let val = map
                        .get(k)
                        .map(|v| format_value_compact(v))
                        .unwrap_or_else(|| String::from("-"));
                    format!("{:width$}", val, width = widths[i])
                })
                .collect();
            output.push_str(&row.join(" │ "));
            output.push('\n');
        }
    }

    output
}

/// Format single object as key-value table.
fn format_object_table(map: &serde_json::Map<String, Value>) -> String {
    let mut rows = Vec::new();
    for (key, value) in map {
        rows.push(format!("{:20} │ {}", key, format_value_compact(value)));
    }
    rows.join("\n")
}

/// Format a JSON value compactly for table display.
fn format_value_compact(value: &Value) -> String {
    match value {
        Value::Null => String::from("null"),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(arr) => format!("[{}]", arr.len()),
        Value::Object(obj) => format!("{{{}}}" , obj.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CaptureAdapter;
    use serial_test::serial;

    #[test]
    #[serial(capture_adapter)]
    fn test_builder_basic() {
        let adapter = CaptureAdapter::install();

        Narration::new("test-actor", "test-action", "test-target").human("Test message").emit();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].actor, "test-actor");
        assert_eq!(captured[0].action, "test-action");
        assert_eq!(captured[0].target, "test-target");
        assert_eq!(captured[0].human, "Test message");
    }

    #[test]
    #[serial(capture_adapter)]
    fn test_builder_with_ids() {
        let adapter = CaptureAdapter::install();

        Narration::new("orchestratord", "enqueue", "job-123")
            .human("Enqueued job")
            .correlation_id("req-abc")
            .job_id("job-123")
            .pool_id("default")
            .emit();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].correlation_id, Some("req-abc".to_string()));
        assert_eq!(captured[0].job_id, Some("job-123".to_string()));
        assert_eq!(captured[0].pool_id, Some("default".to_string()));
    }

    #[test]
    #[serial(capture_adapter)]
    fn test_builder_with_metrics() {
        let adapter = CaptureAdapter::install();

        Narration::new("worker-orcd", "execute", "job-123")
            .human("Completed inference")
            .duration_ms(150)
            .tokens_in(100)
            .tokens_out(50)
            .emit();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].duration_ms, Some(150));
        assert_eq!(captured[0].tokens_in, Some(100));
        assert_eq!(captured[0].tokens_out, Some(50));
    }

    #[test]
    #[serial(capture_adapter)]
    fn test_builder_emit_error() {
        let adapter = CaptureAdapter::install();

        Narration::new("pool-managerd", "spawn", "GPU0")
            .human("Failed to spawn worker")
            .error_kind("ResourceExhausted")
            .emit_error();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].error_kind, Some("ResourceExhausted".to_string()));
    }

    #[test]
    #[serial(capture_adapter)]
    fn test_builder_auto_injection() {
        let adapter = CaptureAdapter::install();

        Narration::new("test", "test", "test").human("Test").emit();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        // Auto-injection should add emitted_by and emitted_at_ms
        assert!(captured[0].emitted_by.is_some());
        assert!(captured[0].emitted_at_ms.is_some());
    }
}
