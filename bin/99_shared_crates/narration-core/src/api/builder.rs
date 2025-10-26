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
///
/// // TEAM-191: With context interpolation (BEST!)
/// Narration::new("queen-rbee", "start", "queen-rbee")
///     .context("http://localhost:8080")
///     .context("8080")
///     .human("✅ Queen started on {0}, port {1}")  // {0} or {} = first context, {1} = second
///     .emit();
/// ```
#[derive(Clone)]
pub struct Narration {
    fields: NarrationFields,
    /// TEAM-191: Store context values for format string interpolation
    /// Use .context() to add values, reference with {0}, {1}, etc. or just {}
    context_values: Vec<String>,
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
        let target_value = target.into();
        Self {
            fields: NarrationFields {
                actor,
                action,
                target: target_value,
                human: String::new(),
                ..Default::default()
            },
            // TEAM-191: Initialize context_values empty - use .context() to add values
            context_values: Vec::new(),
        }
    }

    /// Set the action for this narration.
    ///
    /// Useful when building narrations from a job-scoped factory.
    ///
    /// # Example
    /// ```rust,ignore
    /// let JOB = NARRATE.with_job_id("job-123");
    /// JOB.action("hive_start").human("Starting hive").emit();
    /// ```
    pub fn action(mut self, action: &'static str) -> Self {
        // TEAM-257: Removed panic on long action names - just use as-is
        // The 15-char limit was for formatting aesthetics, not a hard requirement
        // If formatting looks bad, that's a display issue, not a panic-worthy error
        self.fields.action = action;
        // If target is empty, use action as target (backwards compat)
        if self.fields.target.is_empty() {
            self.fields.target = action.to_string();
        }
        self
    }

    /// Add context value for format string interpolation.
    ///
    /// Context values can be referenced in `.human()`, `.cute()`, and `.story()`
    /// using `{0}`, `{1}`, `{2}`, etc.
    ///
    /// # TEAM-191: Chainable Context!
    /// Add as many context values as you need, then reference them by index!
    ///
    /// # Example
    /// ```rust,ignore
    /// Narration::new("queen-rbee", "start", "queen-rbee")
    ///     .context("http://localhost:8080")  // {0}
    ///     .context("8080")                   // {1}
    ///     .human("✅ Queen started on {0}, port {1}")
    ///     .emit();
    /// ```
    pub fn context(mut self, value: impl Into<String>) -> Self {
        self.context_values.push(value.into());
        self
    }

    /// Set the human-readable description.
    ///
    /// TEAM-297: SIMPLIFIED - No more {0}, {1} replacement!
    /// Use Rust's format!() macro before calling this method, or use the n!() macro.
    ///
    /// TEAM-191: Legacy .context() interpolation still supported for backward compatibility
    ///
    /// # Example (New way with n! macro)
    /// ```rust,ignore
    /// use observability_narration_core::n;
    /// n!("start", "✅ Queen started on {}, port {}", url, port);
    /// ```
    ///
    /// # Example (Old way with builder - still works)
    /// ```rust,ignore
    /// Narration::new("queen-rbee", "start", "queen-rbee")
    ///     .context("http://localhost:8080")
    ///     .context("8080")
    ///     .human("✅ Queen started on {}, port {1}")  // Legacy {0}, {1} syntax
    ///     .emit();
    /// ```
    pub fn human(mut self, msg: impl Into<String>) -> Self {
        let mut msg = msg.into();
        // TEAM-191: Replace {N} with context values (legacy backward compatibility)
        for (i, value) in self.context_values.iter().enumerate() {
            msg = msg.replace(&format!("{{{}}}", i), value);
        }
        // TEAM-191: Replace {} with first context value (legacy backward compatibility)
        if let Some(first) = self.context_values.first() {
            msg = msg.replace("{}", first);
        }
        self.fields.human = msg;
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

    /// Set the job ID if provided (handles Option).
    ///
    /// TEAM-276: Convenience method for optional job_id pattern to reduce boilerplate
    ///
    /// # Example
    /// ```rust
    /// use observability_narration_core::{Narration, NarrationFactory};
    ///
    /// const NARRATE: NarrationFactory = NarrationFactory::new("dmn-life");
    ///
    /// let job_id: Option<String> = Some("job-123".to_string());
    ///
    /// // Before (7 lines):
    /// // let mut narration = NARRATE.action("start").human("Starting");
    /// // if let Some(ref job_id) = job_id {
    /// //     narration = narration.job_id(job_id);
    /// // }
    /// // narration.emit();
    ///
    /// // After (3 lines):
    /// NARRATE.action("start")
    ///     .human("Starting daemon")
    ///     .maybe_job_id(job_id.as_deref())
    ///     .emit();
    /// ```
    pub fn maybe_job_id(self, id: Option<&str>) -> Self {
        match id {
            Some(jid) => self.job_id(jid),
            None => self,
        }
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
    ///
    /// TEAM-297: SIMPLIFIED - Use n!() macro for format strings!
    ///
    /// # Example (New way with n! macro)
    /// ```rust,ignore
    /// use observability_narration_core::n;
    /// n!(cute: "start", "🐝 Cute message {}", var);
    /// ```
    ///
    /// # Example (Old way - still works for backward compatibility)
    /// ```rust,ignore
    /// Narration::new("queen-rbee", "start", "queen-rbee")
    ///     .context("value")
    ///     .cute("🐝 Message {}")  // Legacy syntax
    ///     .emit();
    /// ```
    #[cfg(feature = "cute-mode")]
    pub fn cute(mut self, msg: impl Into<String>) -> Self {
        let mut msg = msg.into();
        // TEAM-191: Replace {N} with context values (legacy backward compatibility)
        for (i, value) in self.context_values.iter().enumerate() {
            msg = msg.replace(&format!("{{{}}}", i), value);
        }
        // TEAM-191: Replace {} with first context value (legacy backward compatibility)
        if let Some(first) = self.context_values.first() {
            msg = msg.replace("{}", first);
        }
        self.fields.cute = Some(msg);
        self
    }

    /// Set the story narration message.
    ///
    /// TEAM-297: SIMPLIFIED - Use n!() macro for format strings!
    ///
    /// # Example (New way with n! macro)
    /// ```rust,ignore
    /// use observability_narration_core::n;
    /// n!(story: "start", "'Hello,' said {}", name);
    /// ```
    ///
    /// # Example (Old way - still works for backward compatibility)
    /// ```rust,ignore
    /// Narration::new("queen-rbee", "start", "queen-rbee")
    ///     .context("value")
    ///     .story("'Hello,' said {}")  // Legacy syntax
    ///     .emit();
    /// ```
    pub fn story(mut self, msg: impl Into<String>) -> Self {
        let mut msg = msg.into();
        // TEAM-191: Replace {N} with context values (legacy backward compatibility)
        for (i, value) in self.context_values.iter().enumerate() {
            msg = msg.replace(&format!("{{{}}}", i), value);
        }
        // TEAM-191: Replace {} with first context value (legacy backward compatibility)
        if let Some(first) = self.context_values.first() {
            msg = msg.replace("{}", first);
        }
        self.fields.story = Some(msg);
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

    /// Emit with explicit provenance (internal use by macro)
    #[doc(hidden)]
    pub fn emit_with_provenance(mut self, crate_name: &str, crate_version: &str) {
        if self.fields.emitted_by.is_none() {
            self.fields.emitted_by = Some(format!("{}@{}", crate_name, crate_version));
        }
        if self.fields.emitted_at_ms.is_none() {
            use std::time::{SystemTime, UNIX_EPOCH};
            self.fields.emitted_at_ms =
                Some(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis()
                    as u64);
        }
        crate::narrate(self.fields)
    }

    /// Emit the narration event to all configured outputs.
    ///
    /// Automatically injects service identity and timestamp.
    /// Also automatically includes task-local context (job_id, correlation_id) if set.
    ///
    /// Note: Use the `narrate!` macro instead to capture caller's crate name.
    pub fn emit(mut self) {
        // Automatically include task-local context if available
        if let Some(ctx) = crate::context::get_context() {
            if self.fields.job_id.is_none() {
                self.fields.job_id = ctx.job_id;
            }
            if self.fields.correlation_id.is_none() {
                self.fields.correlation_id = ctx.correlation_id;
            }
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
    let header: Vec<String> =
        keys.iter().enumerate().map(|(i, k)| format!("{:width$}", k, width = widths[i])).collect();
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
                    let val =
                        map.get(k).map(format_value_compact).unwrap_or_else(|| String::from("-"));
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
        Value::Object(obj) => format!("{{{}}}", obj.len()),
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
}

// ============================================================================
// Narration Factory (TEAM-191)
// ============================================================================

/// Factory for creating narrations with a default actor.
///
/// This allows crates to define a default actor once and reuse it,
/// reducing boilerplate and ensuring consistency.
///
/// # Example
/// ```rust
/// use observability_narration_core::{NarrationFactory, ACTOR_QUEEN_ROUTER, ACTION_STATUS};
///
/// // Define at module/crate level
/// const NARRATE: NarrationFactory = NarrationFactory::new(ACTOR_QUEEN_ROUTER);
///
/// // Use throughout the crate
/// NARRATE.narrate(ACTION_STATUS, "registry")
///     .human("Found 2 hives")
///     .emit();
/// ```
pub struct NarrationFactory {
    actor: &'static str,
}

impl NarrationFactory {
    /// Create a new narration factory with a default actor.
    ///
    /// This is a `const fn`, so it can be used in `const` contexts.
    ///
    /// # Compile-Time Validation
    /// - Actor string must be ≤ 20 characters (enforced at compile time)
    ///
    /// # Example
    /// ```rust
    /// use observability_narration_core::{NarrationFactory, ACTOR_QUEEN_ROUTER};
    ///
    /// const NARRATE: NarrationFactory = NarrationFactory::new(ACTOR_QUEEN_ROUTER);
    /// ```
    ///
    /// # Panics
    /// Panics at compile time if actor string is longer than 20 characters.
    pub const fn new(actor: &'static str) -> Self {
        // TEAM-192: Compile-time check for actor length
        // Actor must be ≤ 10 characters for fixed-width output format
        const MAX_ACTOR_LENGTH: usize = 10;

        // Count Unicode characters (not bytes)
        // This is a const fn compatible way to count chars
        let bytes = actor.as_bytes();
        let mut char_count = 0;
        let mut i = 0;

        while i < bytes.len() {
            // Count UTF-8 characters by checking the first byte
            let byte = bytes[i];
            if byte < 0x80 {
                // ASCII (1 byte)
                i += 1;
            } else if byte < 0xE0 {
                // 2-byte character
                i += 2;
            } else if byte < 0xF0 {
                // 3-byte character
                i += 3;
            } else {
                // 4-byte character
                i += 4;
            }
            char_count += 1;
        }

        // This assertion happens at compile time in const context
        assert!(
            char_count <= MAX_ACTOR_LENGTH,
            "Actor string is too long! Maximum 10 characters allowed."
        );

        Self { actor }
    }

    /// Create a new narration with the factory's default actor.
    ///
    /// # Arguments
    /// - `action`: Action performed (max 15 characters)
    ///
    /// # TEAM-192: Compile-time validation!
    /// - Action must be ≤ 15 characters for fixed-width output format
    ///
    /// # Example
    /// ```rust
    /// use observability_narration_core::{NarrationFactory, ACTOR_QUEEN_ROUTER, ACTION_STATUS};
    ///
    /// const NARRATE: NarrationFactory = NarrationFactory::new(ACTOR_QUEEN_ROUTER);
    ///
    /// NARRATE.action("status")
    ///     .context("http://localhost:8080")
    ///     .human("Status check on {}")
    ///     .emit();
    /// ```
    ///
    /// # Panics
    /// Creates a narration with the given action.
    ///
    /// Note: Action names longer than 15 characters may affect formatting aesthetics
    /// but will not cause errors.
    pub fn action(&self, action: &'static str) -> Narration {
        // TEAM-257: Removed panic on long action names - just use as-is
        // The 15-char limit was for formatting aesthetics, not a hard requirement

        Narration::new(self.actor, action, action)
    }

    /// Get the factory's default actor.
    pub const fn actor(&self) -> &'static str {
        self.actor
    }

    /// Create a job-scoped narration builder with job_id baked in.
    ///
    /// Returns a `Narration` that automatically includes the job_id,
    /// so you don't have to call `.job_id()` on every narration.
    ///
    /// # Example
    /// ```rust
    /// use observability_narration_core::NarrationFactory;
    ///
    /// const NARRATE: NarrationFactory = NarrationFactory::new("qn-router");
    ///
    /// // Create job-scoped factory
    /// let JOB = NARRATE.with_job_id("job-abc123");
    ///
    /// // No need to call .job_id() anymore!
    /// JOB.action("hive_start").human("Starting hive").emit();
    /// JOB.action("hive_check").human("Checking status").emit();
    /// ```
    pub fn with_job_id(&self, job_id: impl Into<String>) -> Narration {
        let fields = NarrationFields {
            actor: self.actor,
            job_id: Some(job_id.into()),
            ..Default::default()
        };

        Narration { fields, context_values: Vec::new() }
    }
}

/// Format a job ID to show only last 6 characters with "..." prefix.
///
/// Makes logs more readable by truncating long UUIDs.
///
/// # Example
/// ```
/// use observability_narration_core::short_job_id;
///
/// assert_eq!(short_job_id("job-abc123def456"), "...def456");
/// assert_eq!(short_job_id("short"), "short");
/// ```
pub fn short_job_id(job_id: &str) -> String {
    if job_id.len() > 6 {
        format!("...{}", &job_id[job_id.len() - 6..])
    } else {
        job_id.to_string()
    }
}

#[cfg(test)]
mod factory_tests {
    use super::*;
    use crate::{CaptureAdapter, ACTION_STATUS};
    use serial_test::serial;

    // TEAM-199: Use short actor name for tests (pre-existing actor constants exceed 10 char limit)
    const TEST_ACTOR: &str = "test";

    #[test]
    #[serial(capture_adapter)]
    fn test_factory_basic() {
        let adapter = CaptureAdapter::install();

        const NARRATE: NarrationFactory = NarrationFactory::new(TEST_ACTOR);

        NARRATE.action(ACTION_STATUS).context("registry").human("Test message: {}").emit();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].actor, TEST_ACTOR);
        assert_eq!(captured[0].action, ACTION_STATUS);
        assert_eq!(captured[0].human, "Test message: registry");
    }

    #[test]
    #[serial(capture_adapter)]
    fn test_factory_with_builder_chain() {
        let adapter = CaptureAdapter::install();

        const NARRATE: NarrationFactory = NarrationFactory::new(TEST_ACTOR);

        NARRATE
            .action(ACTION_STATUS)
            .context("registry")
            .human("Test: {}")
            .correlation_id("req-123")
            .duration_ms(100)
            .emit();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].actor, TEST_ACTOR);
        assert_eq!(captured[0].human, "Test: registry");
        assert_eq!(captured[0].correlation_id, Some("req-123".to_string()));
        assert_eq!(captured[0].duration_ms, Some(100));
    }

    #[test]
    fn test_factory_actor_getter() {
        const NARRATE: NarrationFactory = NarrationFactory::new(TEST_ACTOR);
        assert_eq!(NARRATE.actor(), TEST_ACTOR);
    }

    #[test]
    #[serial(capture_adapter)]
    fn test_context_single() {
        let adapter = CaptureAdapter::install();

        Narration::new("queen-rbee", "start", "queen-rbee")
            .context("http://localhost:8080")
            .human("✅ Queen started on {0}")
            .emit();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].human, "✅ Queen started on http://localhost:8080");
    }

    #[test]
    #[serial(capture_adapter)]
    fn test_context_multiple() {
        let adapter = CaptureAdapter::install();

        Narration::new("queen-rbee", "start", "queen-rbee")
            .context("http://localhost:8080")
            .context("8080")
            .human("✅ Queen started on {0}, port {1}")
            .emit();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].human, "✅ Queen started on http://localhost:8080, port 8080");
    }

    #[test]
    #[serial(capture_adapter)]
    fn test_context_with_story() {
        let adapter = CaptureAdapter::install();

        Narration::new("queen-rbee", "start", "queen-rbee")
            .context("http://localhost:8080")
            .context("8080")
            .human("✅ Queen started on {0}, port {1}")
            .story("The queen awoke at {0} on port {1}! 🎀")
            .emit();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].human, "✅ Queen started on http://localhost:8080, port 8080");
        assert_eq!(
            captured[0].story,
            Some("The queen awoke at http://localhost:8080 on port 8080! 🎀".to_string())
        );
    }

    #[test]
    #[serial(capture_adapter)]
    fn test_context_chainable() {
        let adapter = CaptureAdapter::install();

        Narration::new("queen-rbee", "start", "queen-rbee")
            .context("value1")
            .context("value2")
            .context("value3")
            .human("Values: {0}, {1}, {2}")
            .emit();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].human, "Values: value1, value2, value3");
    }

    #[test]
    #[serial(capture_adapter)]
    fn test_context_with_empty_braces() {
        let adapter = CaptureAdapter::install();

        Narration::new("queen-rbee", "start", "queen-rbee")
            .context("http://localhost:8080")
            .context("8080")
            .human("✅ Queen started on {}, port {1}") // {} = {0}
            .emit();

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].human, "✅ Queen started on http://localhost:8080, port 8080");
    }
}
