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

/// Job ID suffix length for short display (e.g., "...abc123")
pub const SHORT_JOB_ID_SUFFIX: usize = 6;

// ============================================================================
// MESSAGE FORMATTING
// ============================================================================

/// Format a narration message in standard format.
///
/// Format: 
/// ```text
/// \x1b[1m[actor              ] action              \x1b[0m
/// message
/// (blank line)
/// ```
/// - Actor: ACTOR_WIDTH chars (left-aligned, padded)
/// - Action: ACTION_WIDTH chars (left-aligned, padded)
/// - Message: on new line, no formatting
/// - Trailing newline: separates consecutive narrations
/// - First line: bold ANSI escape codes
///
/// # Example
/// ```
/// use observability_narration_core::format::format_message;
///
/// let formatted = format_message("queen", "start", "Starting hive");
/// // First line is bold: [queen              ] start              
/// // Second line: Starting hive
/// // Third line: blank (separates from next narration)
/// ```
pub fn format_message(actor: &str, action: &str, message: &str) -> String {
    format!(
        "\x1b[1m[{:<width_actor$}] {:<width_action$}\x1b[0m\n{}\n",
        actor,
        action,
        message,
        width_actor = ACTOR_WIDTH,
        width_action = ACTION_WIDTH
    )
}

// ============================================================================
// LEGACY CONTEXT INTERPOLATION
// ============================================================================

/// Replace {N} placeholders with context values (legacy backward compatibility).
///
/// TEAM-191: This is the legacy .context() interpolation pattern.
/// New code should use Rust's format!() macro instead.
///
/// # Example
/// ```
/// use observability_narration_core::format::interpolate_context;
///
/// let msg = "Found {0} hives on {1}";
/// let context = vec!["2".to_string(), "localhost".to_string()];
/// let result = interpolate_context(msg, &context);
/// assert_eq!(result, "Found 2 hives on localhost");
/// ```
#[deprecated(
    since = "0.5.0",
    note = "Use Rust's format!() macro instead - this legacy {0}, {1} syntax is deprecated. Use n!() macro for narration."
)]
pub fn interpolate_context(msg: &str, context_values: &[String]) -> String {
    let mut result = msg.to_string();
    
    // Replace {N} with context values
    for (i, value) in context_values.iter().enumerate() {
        result = result.replace(&format!("{{{}}}", i), value);
    }
    
    // Replace {} with first context value (legacy)
    if let Some(first) = context_values.first() {
        result = result.replace("{}", first);
    }
    
    result
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
        let formatted = format_message("queen", "start", "Starting hive");
        // Format: Bold first line with actor/action, message on second line, blank line after
        // Actor: 20 chars, Action: 20 chars
        assert_eq!(formatted, "\x1b[1m[queen               ] start               \x1b[0m\nStarting hive\n");
    }

    #[test]
    fn test_format_message_long_names() {
        // Should not truncate - format! will extend if needed
        let formatted = format_message("very-long-actor", "very-long-action", "Message");
        assert!(formatted.contains("very-long-actor"));
        assert!(formatted.contains("very-long-action"));
        assert!(formatted.contains("\nMessage\n")); // Message on new line with trailing newline
        assert!(formatted.contains("\x1b[1m")); // Contains bold
        assert!(formatted.contains("\x1b[0m")); // Contains reset
    }

    #[test]
    fn test_interpolate_context() {
        let msg = "Found {0} hives on {1}";
        let context = vec!["2".to_string(), "localhost".to_string()];
        let result = interpolate_context(msg, &context);
        assert_eq!(result, "Found 2 hives on localhost");
    }

    #[test]
    fn test_interpolate_context_legacy_braces() {
        let msg = "Found {} hives";
        let context = vec!["2".to_string()];
        let result = interpolate_context(msg, &context);
        assert_eq!(result, "Found 2 hives");
    }

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
