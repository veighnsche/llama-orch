// otel.rs — OpenTelemetry integration for Cloud Profile
// Implements CLOUD_PROFILE_NARRATION_REQUIREMENTS.md Section 1

use crate::NarrationFields;

#[cfg(feature = "otel")]
use opentelemetry::trace::{SpanContext, TraceContextExt};

/// Extract trace context from current OpenTelemetry span.
/// Returns (trace_id, span_id, parent_span_id) as hex strings.
#[cfg(feature = "otel")]
pub fn extract_otel_context() -> (Option<String>, Option<String>, Option<String>) {
    use opentelemetry::Context;

    let ctx = Context::current();
    let span = ctx.span();
    let span_ctx = span.span_context();

    if !span_ctx.is_valid() {
        return (None, None, None);
    }

    let trace_id = Some(format!("{:032x}", span_ctx.trace_id()));
    let span_id = Some(format!("{:016x}", span_ctx.span_id()));

    // Note: OpenTelemetry doesn't expose parent_span_id directly in SpanContext
    // It's tracked internally in the span hierarchy
    let parent_span_id = None;

    (trace_id, span_id, parent_span_id)
}

#[cfg(not(feature = "otel"))]
pub fn extract_otel_context() -> (Option<String>, Option<String>, Option<String>) {
    (None, None, None)
}

/// Narrate with automatic OpenTelemetry context extraction.
///
/// This is the recommended function for Cloud Profile deployments.
/// It automatically extracts trace_id and span_id from the current OTEL context.
///
/// # Example
/// ```rust,ignore
/// use observability_narration_core::{narrate_with_otel_context, NarrationFields};
///
/// // Inside an OTEL span
/// narrate_with_otel_context(NarrationFields {
///     actor: "orchestratord",
///     action: "dispatch",
///     target: "task-123".to_string(),
///     human: "Dispatching task to pool 'default'".to_string(),
///     pool_id: Some("default".into()),
///     ..Default::default()
/// });
/// // trace_id and span_id are automatically extracted from current span
/// ```
pub fn narrate_with_otel_context(mut fields: NarrationFields) {
    let (trace_id, span_id, parent_span_id) = extract_otel_context();

    // Only override if not already set
    if fields.trace_id.is_none() {
        fields.trace_id = trace_id;
    }
    if fields.span_id.is_none() {
        fields.span_id = span_id;
    }
    if fields.parent_span_id.is_none() {
        fields.parent_span_id = parent_span_id;
    }

    crate::narrate(fields);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_otel_context_without_feature() {
        #[cfg(not(feature = "otel"))]
        {
            let (trace_id, span_id, parent_span_id) = extract_otel_context();
            assert_eq!(trace_id, None);
            assert_eq!(span_id, None);
            assert_eq!(parent_span_id, None);
        }
    }

    #[test]
    fn test_narrate_with_otel_context_no_panic() {
        // Should not panic even without OTEL context
        narrate_with_otel_context(NarrationFields {
            actor: "test",
            action: "test",
            target: "test".to_string(),
            human: "Test".to_string(),
            ..Default::default()
        });
    }
}
