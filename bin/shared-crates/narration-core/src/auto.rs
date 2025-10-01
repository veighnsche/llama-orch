// auto.rs — Auto-injection helpers for Cloud Profile
// Implements CLOUD_PROFILE_NARRATION_REQUIREMENTS.md Section 2

use crate::NarrationFields;
use std::time::{SystemTime, UNIX_EPOCH};

/// Get current service name and version from Cargo metadata.
/// Format: "{service_name}@{version}"
pub fn service_identity() -> String {
    format!("{}@{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
}

/// Get current timestamp in milliseconds since Unix epoch.
pub fn current_timestamp_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

/// Inject provenance fields (service identity and timestamp) if not already set.
fn inject_provenance(fields: &mut NarrationFields) {
    if fields.emitted_by.is_none() {
        fields.emitted_by = Some(service_identity());
    }
    if fields.emitted_at_ms.is_none() {
        fields.emitted_at_ms = Some(current_timestamp_ms());
    }
}

/// Narrate with automatic service identity and timestamp injection.
///
/// This is the recommended function for Cloud Profile deployments when
/// you don't need OpenTelemetry context (e.g., background tasks, startup).
///
/// Automatically injects:
/// - `emitted_by`: "{service_name}@{version}"
/// - `emitted_at_ms`: Current Unix timestamp in milliseconds
///
/// # Example
/// ```rust
/// use observability_narration_core::{narrate_auto, NarrationFields};
///
/// narrate_auto(NarrationFields {
///     actor: "pool-managerd",
///     action: "startup",
///     target: "GPU0".to_string(),
///     human: "pool-managerd started successfully".to_string(),
///     ..Default::default()
/// });
/// // emitted_by and emitted_at_ms are automatically injected
/// ```
pub fn narrate_auto(mut fields: NarrationFields) {
    inject_provenance(&mut fields);

    // Only inject if not already set
    if fields.emitted_by.is_none() {
        fields.emitted_by = Some(service_identity());
    }
    if fields.emitted_at_ms.is_none() {
        fields.emitted_at_ms = Some(current_timestamp_ms());
    }
    crate::narrate(fields);
}

/// Narrate with both auto-injection AND OpenTelemetry context.
///
/// This is the MOST COMPLETE function for Cloud Profile deployments.
/// Combines auto-injection (service identity, timestamp) with OTEL context extraction.
///
/// Automatically injects:
/// - `emitted_by`: "{service_name}@{version}"
/// - `emitted_at_ms`: Current Unix timestamp in milliseconds
/// - `trace_id`: From current OTEL span
/// - `span_id`: From current OTEL span
///
/// # Example
/// ```rust,ignore
/// use observability_narration_core::{narrate_full, NarrationFields};
///
/// // Inside an OTEL span
/// narrate_full(NarrationFields {
///     actor: "orchestratord",
///     action: "dispatch",
///     target: "task-123".to_string(),
///     human: "Dispatching task to pool 'default'".to_string(),
///     pool_id: Some("default".into()),
///     ..Default::default()
/// });
/// // All provenance fields are automatically injected
/// ```
pub fn narrate_full(mut fields: NarrationFields) {
    inject_provenance(&mut fields);

    // Inject service identity and timestamp
    if fields.emitted_by.is_none() {
        fields.emitted_by = Some(service_identity());
    }
    if fields.emitted_at_ms.is_none() {
        fields.emitted_at_ms = Some(current_timestamp_ms());
    }

    // Extract OTEL context
    let (trace_id, span_id, parent_span_id) = crate::otel::extract_otel_context();
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

/// Macro for ergonomic auto-injection.
///
/// # Example
/// ```rust,ignore
/// use observability_narration_core::narrate_auto;
///
/// narrate_auto!{
///     actor: "pool-managerd",
///     action: "spawn",
///     target: "GPU0",
///     human: "Spawning engine llamacpp-v1",
///     pool_id: Some("default".into()),
/// };
/// ```
#[macro_export]
macro_rules! narrate_auto {
    ($($field:ident: $value:expr),* $(,)?) => {
        $crate::auto::narrate_auto($crate::NarrationFields {
            $($field: $value,)*
            ..Default::default()
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_identity() {
        let identity = service_identity();
        assert!(identity.contains("@"));
        assert!(identity.contains("observability-narration-core"));
    }

    #[test]
    fn test_current_timestamp_ms() {
        let ts1 = current_timestamp_ms();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let ts2 = current_timestamp_ms();
        assert!(ts2 > ts1);
    }

    #[test]
    fn test_narrate_auto_injects_fields() {
        use crate::CaptureAdapter;

        // Uninstall any existing adapter first
        CaptureAdapter::uninstall();

        let adapter = CaptureAdapter::install();
        adapter.clear();

        narrate_auto(NarrationFields {
            actor: "test",
            action: "test",
            target: "test".to_string(),
            human: "Test".to_string(),
            ..Default::default()
        });

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1, "Expected 1 captured event, got {}", captured.len());
        assert!(captured[0].emitted_by.is_some());
        assert!(captured[0].emitted_at_ms.is_some());
    }

    #[test]
    fn test_narrate_auto_respects_existing_fields() {
        use crate::CaptureAdapter;

        // Uninstall any existing adapter first
        CaptureAdapter::uninstall();

        let adapter = CaptureAdapter::install();
        adapter.clear();

        let custom_identity = "custom-service@1.0.0".to_string();
        let custom_timestamp = 1234567890u64;

        narrate_auto(NarrationFields {
            actor: "test",
            action: "test",
            target: "test".to_string(),
            human: "Test".to_string(),
            emitted_by: Some(custom_identity.clone()),
            emitted_at_ms: Some(custom_timestamp),
            ..Default::default()
        });

        let captured = adapter.captured();
        assert_eq!(captured.len(), 1, "Expected 1 captured event, got {}", captured.len());
        assert_eq!(captured[0].emitted_by, Some(custom_identity));
        assert_eq!(captured[0].emitted_at_ms, Some(custom_timestamp));
    }
}
