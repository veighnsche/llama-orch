// TEAM-310: Centralized formatting logic for narration
//! Formatting utilities for narration messages
//!
//! This module centralizes all formatting logic for narration events:
//! - Message formatting (actor/action/human)
//! - Table formatting (JSON arrays/objects)
//! - Value formatting (compact display)
//! - Job ID shortening
//!
//! Previously scattered across: sse_sink.rs, builder.rs, emit.rs

use serde_json::Value;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Actor field width in formatted output (characters)
pub const ACTOR_WIDTH: usize = 20;

/// Action field width in formatted output (characters)
pub const ACTION_WIDTH: usize = 20;

/// Function name field width in formatted output (characters)
pub const FN_NAME_WIDTH: usize = 40;

/// Job ID suffix length for short display (e.g., "...abc123")
pub const SHORT_JOB_ID_SUFFIX: usize = 6;

// ============================================================================
// MESSAGE FORMATTING
// ============================================================================

// TEAM-312: ENTROPY REMOVED - Deleted deprecated format_message()
// Use format_message_with_fn() instead (pass None for fn_name if not needed)

/// Format a narration message with optional function name.
///
/// TEAM-311: Internal formatting function - use `NarrationFields::format()` for public API!
///
/// ⭐ **PREFER:** `NarrationFields::format()` for consistent formatting
///
/// This function contains the actual formatting logic but is exposed for internal use.
/// External code should call `NarrationFields::format()` instead to ensure all formatting
/// goes through the same code path.
///
/// Format:
/// ```text
/// \x1b[1m[actor              ] fn_name            \x1b[0m action              
/// message
/// (blank line)
/// ```
/// - Actor: ACTOR_WIDTH chars (left-aligned, padded) - **BOLD**
/// - Function name: ACTION_WIDTH chars (left-aligned, padded) - **BOLD**
/// - Action: light (not bold)
/// - Message: on new line, no formatting
/// - Trailing newline: separates consecutive narrations
///
/// # Public API
/// ```
/// use observability_narration_core::NarrationFields;
///
/// let fields = NarrationFields {
///     actor: "auto-update",
///     action: "parse_deps",
///     target: "parse_deps".to_string(),
///     human: "Scanning crate".to_string(),
///     fn_name: Some("parse".to_string()),
///     ..Default::default()
/// };
///
/// // ⭐ Use this - central formatting method!
/// let formatted = fields.format();
/// ```
pub fn format_message_with_fn(action: &str, message: &str, fn_name: &str) -> String {
    format!(
        "\x1b[1m{:<width_fn$}\x1b[0m \x1b[2m{:<width_action$}\x1b[0m\n{}\n",
        fn_name,
        action,
        message,
        width_fn = FN_NAME_WIDTH,
        width_action = ACTION_WIDTH
    )
}

// ============================================================================
// JOB ID FORMATTING
// ============================================================================

/// Shorten a job ID to last 6 characters for display.
///
/// # Example
/// ```
/// use observability_narration_core::format::short_job_id;
///
/// assert_eq!(short_job_id("job-abc123def456"), "...def456");
/// assert_eq!(short_job_id("short"), "short");
/// ```
pub fn short_job_id(job_id: &str) -> String {
    if job_id.len() > SHORT_JOB_ID_SUFFIX {
        format!("...{}", &job_id[job_id.len() - SHORT_JOB_ID_SUFFIX..])
    } else {
        job_id.to_string()
    }
}

// ============================================================================
// TABLE FORMATTING (JSON)
// ============================================================================

/// Format a JSON array as a table.
///
/// Used for displaying structured data (e.g., hive lists, worker lists).
///
/// # Example
/// ```
/// use observability_narration_core::format::format_array_table;
/// use serde_json::json;
///
/// let data = json!([
///     {"name": "hive1", "status": "running"},
///     {"name": "hive2", "status": "stopped"}
/// ]);
///
/// let table = format_array_table(data.as_array().unwrap());
/// assert!(table.contains("name"));
/// assert!(table.contains("hive1"));
/// ```
pub fn format_array_table(arr: &[Value]) -> String {
    if arr.is_empty() {
        return String::from("(empty)");
    }

    // Collect all unique keys
    let mut keys = Vec::new();
    for item in arr {
        if let Some(obj) = item.as_object() {
            for key in obj.keys() {
                if !keys.contains(key) {
                    keys.push(key.clone());
                }
            }
        }
    }

    if keys.is_empty() {
        return String::from("(no keys)");
    }

    // Calculate column widths
    let mut widths: Vec<usize> = keys.iter().map(|k| k.len()).collect();
    for item in arr {
        if let Some(obj) = item.as_object() {
            for (i, key) in keys.iter().enumerate() {
                if let Some(val) = obj.get(key) {
                    let len = format_value_compact(val).len();
                    if len > widths[i] {
                        widths[i] = len;
                    }
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

    // Build separator
    let separator: Vec<String> = widths.iter().map(|w| "─".repeat(*w)).collect();
    output.push_str(&separator.join("─┼─"));
    output.push('\n');

    // Build rows
    for item in arr {
        if let Some(map) = item.as_object() {
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

/// Format a JSON object as a two-column table (key | value).
///
/// # Example
/// ```
/// use observability_narration_core::format::format_object_table;
/// use serde_json::json;
///
/// let data = json!({"name": "hive1", "status": "running"});
/// let table = format_object_table(data.as_object().unwrap());
/// assert!(table.contains("name"));
/// assert!(table.contains("hive1"));
/// ```
pub fn format_object_table(map: &serde_json::Map<String, Value>) -> String {
    let mut rows = Vec::new();
    for (key, value) in map {
        rows.push(format!("{:20} │ {}", key, format_value_compact(value)));
    }
    rows.join("\n")
}

/// Format a JSON value compactly (for table cells).
///
/// - Primitives: as-is
/// - Arrays: `[N]` (length)
/// - Objects: `{N}` (key count)
pub fn format_value_compact(value: &Value) -> String {
    match value {
        Value::Null => String::from("null"),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(arr) => format!("[{}]", arr.len()),
        Value::Object(obj) => format!("{{{}}}", obj.len()),
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_format_message() {
        // TEAM-312: Use format_message_with_fn() with None for fn_name
        let formatted = format_message_with_fn("queen", "start", "Starting hive", None);
        // Format: Bold first line with actor/action, message on second line, blank line after
        // Actor: 20 chars, Action: 20 chars
        assert_eq!(
            formatted,
            "\x1b[1m[queen               ] start               \x1b[0m\nStarting hive\n"
        );
    }

    #[test]
    fn test_format_message_long_names() {
        // TEAM-312: Use format_message_with_fn() with None for fn_name
        let formatted =
            format_message_with_fn("very-long-actor", "very-long-action", "Message", None);
        assert!(formatted.contains("very-long-actor"));
        assert!(formatted.contains("very-long-action"));
        assert!(formatted.contains("\nMessage\n")); // Message on new line with trailing newline
        assert!(formatted.contains("\x1b[1m")); // Contains bold
        assert!(formatted.contains("\x1b[0m")); // Contains reset
    }

    // TEAM-312: DELETED test_interpolate_context tests
    // Use Rust's format!() macro instead - no need to test legacy {0}, {1} syntax

    #[test]
    fn test_short_job_id() {
        assert_eq!(short_job_id("job-abc123def456"), "...def456");
        assert_eq!(short_job_id("short"), "short");
        assert_eq!(short_job_id(""), "");
    }

    #[test]
    fn test_format_array_table() {
        let data = json!([
            {"name": "hive1", "status": "running"},
            {"name": "hive2", "status": "stopped"}
        ]);

        let table = format_array_table(data.as_array().unwrap());
        assert!(table.contains("name"));
        assert!(table.contains("status"));
        assert!(table.contains("hive1"));
        assert!(table.contains("running"));
        assert!(table.contains("hive2"));
        assert!(table.contains("stopped"));
    }

    #[test]
    fn test_format_array_table_empty() {
        let data = json!([]);
        let table = format_array_table(data.as_array().unwrap());
        assert_eq!(table, "(empty)");
    }

    #[test]
    fn test_format_object_table() {
        let data = json!({"name": "hive1", "status": "running"});
        let table = format_object_table(data.as_object().unwrap());
        assert!(table.contains("name"));
        assert!(table.contains("hive1"));
        assert!(table.contains("status"));
        assert!(table.contains("running"));
    }

    #[test]
    fn test_format_value_compact() {
        assert_eq!(format_value_compact(&json!(null)), "null");
        assert_eq!(format_value_compact(&json!(true)), "true");
        assert_eq!(format_value_compact(&json!(42)), "42");
        assert_eq!(format_value_compact(&json!("hello")), "hello");
        assert_eq!(format_value_compact(&json!([1, 2, 3])), "[3]");
        assert_eq!(format_value_compact(&json!({"a": 1, "b": 2})), "{2}");
    }
}
