# narration-core Behaviors Catalog

**Purpose**: Complete catalog of ALL behaviors in narration-core for BDD test coverage  
**Date**: 2025-09-30  
**Source**: Code flow analysis of all modules and functions

---

## 📋 Table of Contents

1. [Core Narration Behaviors](#core-narration-behaviors)
2. [Auto-Injection Behaviors](#auto-injection-behaviors)
3. [Redaction Behaviors](#redaction-behaviors)
4. [Test Capture Behaviors](#test-capture-behaviors)
5. [OpenTelemetry Integration Behaviors](#opentelemetry-integration-behaviors)
6. [HTTP Header Propagation Behaviors](#http-header-propagation-behaviors)
7. [Field Taxonomy Behaviors](#field-taxonomy-behaviors)
8. [Feature Flag Behaviors](#feature-flag-behaviors)

---

## Core Narration Behaviors

### Basic Narration (`narrate()`)
- **B-CORE-001**: When called with NarrationFields → emit structured tracing event at INFO level
- **B-CORE-002**: When `human` field contains secrets → automatically redact them before emission
- **B-CORE-003**: When `actor` field provided → include in tracing event
- **B-CORE-004**: When `action` field provided → include in tracing event
- **B-CORE-005**: When `target` field provided → include in tracing event
- **B-CORE-006**: When `correlation_id` provided → include in tracing event
- **B-CORE-007**: When `session_id` provided → include in tracing event
- **B-CORE-008**: When `job_id` provided → include in tracing event
- **B-CORE-009**: When `task_id` provided → include in tracing event
- **B-CORE-010**: When `pool_id` provided → include in tracing event
- **B-CORE-011**: When `replica_id` provided → include in tracing event
- **B-CORE-012**: When `worker_id` provided → include in tracing event
- **B-CORE-013**: When test capture adapter installed → notify adapter of narration event
- **B-CORE-014**: When test capture adapter NOT installed → skip notification (no-op)
- **B-CORE-015**: When `emitted_by` provided → include in tracing event
- **B-CORE-016**: When `emitted_at_ms` provided → include in tracing event
- **B-CORE-017**: When `trace_id` provided → include in tracing event
- **B-CORE-018**: When `span_id` provided → include in tracing event
- **B-CORE-019**: When `parent_span_id` provided → include in tracing event
- **B-CORE-020**: When `source_location` provided → include in tracing event

### Legacy Compatibility (`human()`)
- **B-CORE-100**: When called with actor/action/target/msg → convert to NarrationFields
- **B-CORE-101**: When called → emit deprecation warning
- **B-CORE-102**: When called → delegate to `narrate()` function
- **B-CORE-103**: When called → all other fields default to None

---

## Auto-Injection Behaviors

### Service Identity (`service_identity()`)
- **B-AUTO-001**: When called → return "{CARGO_PKG_NAME}@{CARGO_PKG_VERSION}"
- **B-AUTO-002**: When called → format is always "name@version"
- **B-AUTO-003**: When called → uses compile-time environment variables

### Timestamp (`current_timestamp_ms()`)
- **B-AUTO-010**: When called → return Unix timestamp in milliseconds
- **B-AUTO-011**: When SystemTime fails → return 0 (unwrap_or_default)
- **B-AUTO-012**: When called twice → second timestamp >= first timestamp

### Provenance Injection (`inject_provenance()`)
- **B-AUTO-020**: When `emitted_by` is None → inject service_identity()
- **B-AUTO-021**: When `emitted_by` already set → preserve existing value
- **B-AUTO-022**: When `emitted_at_ms` is None → inject current_timestamp_ms()
- **B-AUTO-023**: When `emitted_at_ms` already set → preserve existing value

### Auto Narration (`narrate_auto()`)
- **B-AUTO-030**: When called → inject provenance fields
- **B-AUTO-031**: When called → delegate to narrate()
- **B-AUTO-032**: When `emitted_by` not provided → auto-inject service identity
- **B-AUTO-033**: When `emitted_at_ms` not provided → auto-inject timestamp
- **B-AUTO-034**: When both already provided → preserve existing values

### Full Auto Narration (`narrate_full()`)
- **B-AUTO-040**: When called → inject provenance fields
- **B-AUTO-041**: When called → extract OTEL context
- **B-AUTO-042**: When called → delegate to narrate()
- **B-AUTO-043**: When `trace_id` not provided AND OTEL context available → inject trace_id
- **B-AUTO-044**: When `span_id` not provided AND OTEL context available → inject span_id
- **B-AUTO-045**: When `parent_span_id` not provided AND OTEL context available → inject parent_span_id
- **B-AUTO-046**: When OTEL context not available → trace/span IDs remain None
- **B-AUTO-047**: When trace/span IDs already provided → preserve existing values

---

## Redaction Behaviors

### Redaction Policy
- **B-RED-001**: Default policy → mask_bearer_tokens = true
- **B-RED-002**: Default policy → mask_api_keys = true
- **B-RED-003**: Default policy → mask_uuids = false
- **B-RED-004**: Default policy → replacement = "[REDACTED]"
- **B-RED-005**: Custom policy → can override all settings
- **B-RED-006**: Custom policy → can set custom replacement string

### Bearer Token Redaction
- **B-RED-010**: When text contains "Bearer abc123" → replace with "[REDACTED]"
- **B-RED-011**: When text contains "bearer abc123" (lowercase) → replace with "[REDACTED]"
- **B-RED-012**: When text contains "BEARER abc123" (uppercase) → replace with "[REDACTED]"
- **B-RED-013**: When text contains multiple bearer tokens → replace all
- **B-RED-014**: When mask_bearer_tokens = false → do not redact
- **B-RED-015**: When no bearer tokens present → return text unchanged

### API Key Redaction
- **B-RED-020**: When text contains "api_key=secret" → replace with "[REDACTED]"
- **B-RED-021**: When text contains "apikey=secret" → replace with "[REDACTED]"
- **B-RED-022**: When text contains "key=secret" → replace with "[REDACTED]"
- **B-RED-023**: When text contains "token=secret" → replace with "[REDACTED]"
- **B-RED-024**: When text contains "secret=value" → replace with "[REDACTED]"
- **B-RED-025**: When text contains "password=value" → replace with "[REDACTED]"
- **B-RED-026**: When text contains "api_key: secret" (colon separator) → replace with "[REDACTED]"
- **B-RED-027**: When mask_api_keys = false → do not redact
- **B-RED-028**: When no API keys present → return text unchanged

### UUID Redaction
- **B-RED-030**: When mask_uuids = true AND text contains UUID → replace with "[REDACTED]"
- **B-RED-031**: When mask_uuids = false AND text contains UUID → do not redact
- **B-RED-032**: When UUID format is 8-4-4-4-12 hex → recognize and redact (if enabled)
- **B-RED-033**: When UUID format is invalid → do not redact

### Regex Compilation
- **B-RED-040**: When bearer_token_regex() called first time → compile regex and cache
- **B-RED-041**: When bearer_token_regex() called again → return cached regex
- **B-RED-042**: When api_key_regex() called first time → compile regex and cache
- **B-RED-043**: When api_key_regex() called again → return cached regex
- **B-RED-044**: When uuid_regex() called first time → compile regex and cache
- **B-RED-045**: When uuid_regex() called again → return cached regex
- **B-RED-046**: When regex pattern is invalid → panic with "BUG: {pattern} regex pattern is invalid"

---

## Test Capture Behaviors

### Capture Adapter Installation
- **B-CAP-001**: When install() called → create new adapter instance
- **B-CAP-002**: When install() called → set as global capture target
- **B-CAP-003**: When install() called → return adapter for assertions
- **B-CAP-004**: When install() called multiple times → reuse same global instance
- **B-CAP-005**: When uninstall() called → clear captured events
- **B-CAP-006**: When uninstall() called → global instance remains (OnceLock limitation)

### Event Capture
- **B-CAP-010**: When narrate() called AND adapter installed → capture event
- **B-CAP-011**: When narrate() called AND adapter NOT installed → skip capture
- **B-CAP-012**: When capture() called → append event to internal vector
- **B-CAP-013**: When capture() called AND mutex poisoned → silently fail (test bug)
- **B-CAP-014**: When captured() called → return clone of all events
- **B-CAP-015**: When captured() called AND mutex poisoned → panic with "BUG: capture adapter mutex poisoned"

### Event Clearing
- **B-CAP-020**: When clear() called → empty internal event vector
- **B-CAP-021**: When clear() called AND mutex poisoned → silently fail
- **B-CAP-022**: When clear() called → subsequent captured() returns empty vector

### Assertions
- **B-CAP-030**: When assert_includes(substring) AND found → pass
- **B-CAP-031**: When assert_includes(substring) AND not found → panic with descriptive message
- **B-CAP-032**: When assert_field(field, value) AND found → pass
- **B-CAP-033**: When assert_field(field, value) AND not found → panic with descriptive message
- **B-CAP-034**: When assert_field() supports: actor, action, target, pool_id, session_id, correlation_id, emitted_by, trace_id
- **B-CAP-035**: When assert_correlation_id_present() AND found → pass
- **B-CAP-036**: When assert_correlation_id_present() AND not found → panic
- **B-CAP-037**: When assert_provenance_present() AND emitted_by OR emitted_at_ms present → pass
- **B-CAP-038**: When assert_provenance_present() AND neither present → panic

### Captured Narration Conversion
- **B-CAP-040**: When NarrationFields converted to CapturedNarration → copy actor field
- **B-CAP-041**: When NarrationFields converted to CapturedNarration → copy action field
- **B-CAP-042**: When NarrationFields converted to CapturedNarration → copy target field
- **B-CAP-043**: When NarrationFields converted to CapturedNarration → copy human field
- **B-CAP-044**: When NarrationFields converted to CapturedNarration → copy correlation_id field
- **B-CAP-045**: When NarrationFields converted to CapturedNarration → copy session_id field
- **B-CAP-046**: When NarrationFields converted to CapturedNarration → copy pool_id field
- **B-CAP-047**: When NarrationFields converted to CapturedNarration → copy replica_id field
- **B-CAP-048**: When NarrationFields converted to CapturedNarration → copy emitted_by field
- **B-CAP-049**: When NarrationFields converted to CapturedNarration → copy emitted_at_ms field
- **B-CAP-050**: When NarrationFields converted to CapturedNarration → copy trace_id field
- **B-CAP-051**: When NarrationFields converted to CapturedNarration → copy parent_span_id field

---

## OpenTelemetry Integration Behaviors

### Context Extraction (`extract_otel_context()`)
- **B-OTEL-001**: When `otel` feature enabled AND valid span context → return (trace_id, span_id, None)
- **B-OTEL-002**: When `otel` feature enabled AND invalid span context → return (None, None, None)
- **B-OTEL-003**: When `otel` feature disabled → return (None, None, None)
- **B-OTEL-004**: When trace_id extracted → format as 32-char hex string
- **B-OTEL-005**: When span_id extracted → format as 16-char hex string
- **B-OTEL-006**: When parent_span_id → always return None (not exposed by OTEL API)

### OTEL Narration (`narrate_with_otel_context()`)
- **B-OTEL-010**: When called → extract OTEL context
- **B-OTEL-011**: When called → inject trace_id if not already set
- **B-OTEL-012**: When called → inject span_id if not already set
- **B-OTEL-013**: When called → inject parent_span_id if not already set
- **B-OTEL-014**: When called → delegate to narrate()
- **B-OTEL-015**: When trace_id already set → preserve existing value
- **B-OTEL-016**: When span_id already set → preserve existing value
- **B-OTEL-017**: When no OTEL context available → trace/span IDs remain None

---

## HTTP Header Propagation Behaviors

### Header Constants
- **B-HTTP-001**: Correlation ID header name → "X-Correlation-Id"
- **B-HTTP-002**: Trace ID header name → "X-Trace-Id"
- **B-HTTP-003**: Span ID header name → "X-Span-Id"
- **B-HTTP-004**: Parent Span ID header name → "X-Parent-Span-Id"

### Context Extraction (`extract_context_from_headers()`)
- **B-HTTP-010**: When X-Correlation-Id header present → extract as correlation_id
- **B-HTTP-011**: When X-Trace-Id header present → extract as trace_id
- **B-HTTP-012**: When X-Span-Id header present → extract as span_id
- **B-HTTP-013**: When X-Parent-Span-Id header present → extract as parent_span_id
- **B-HTTP-014**: When header missing → return None for that field
- **B-HTTP-015**: When header value is invalid UTF-8 → return None for that field
- **B-HTTP-016**: When all headers missing → return (None, None, None, None)

### Context Injection (`inject_context_into_headers()`)
- **B-HTTP-020**: When correlation_id provided → insert X-Correlation-Id header
- **B-HTTP-021**: When trace_id provided → insert X-Trace-Id header
- **B-HTTP-022**: When span_id provided → insert X-Span-Id header
- **B-HTTP-023**: When parent_span_id provided → insert X-Parent-Span-Id header
- **B-HTTP-024**: When field is None → skip inserting that header
- **B-HTTP-025**: When all fields are None → insert no headers

### HeaderLike Trait
- **B-HTTP-030**: When get_str(name) called → return header value as String or None
- **B-HTTP-031**: When insert_str(name, value) called → insert header into map
- **B-HTTP-032**: When implemented for HashMap → get_str returns cloned value
- **B-HTTP-033**: When implemented for HashMap → insert_str inserts String key/value

---

## Field Taxonomy Behaviors

### Required Fields
- **B-FIELD-001**: actor field → &'static str, required
- **B-FIELD-002**: action field → &'static str, required
- **B-FIELD-003**: target field → String, required
- **B-FIELD-004**: human field → String, required

### Correlation Fields
- **B-FIELD-010**: correlation_id field → Option<String>
- **B-FIELD-011**: session_id field → Option<String>
- **B-FIELD-012**: job_id field → Option<String>
- **B-FIELD-013**: task_id field → Option<String>
- **B-FIELD-014**: pool_id field → Option<String>
- **B-FIELD-015**: replica_id field → Option<String>
- **B-FIELD-016**: worker_id field → Option<String>

### Contextual Fields
- **B-FIELD-020**: error_kind field → Option<String>
- **B-FIELD-021**: retry_after_ms field → Option<u64>
- **B-FIELD-022**: backoff_ms field → Option<u64>
- **B-FIELD-023**: duration_ms field → Option<u64>
- **B-FIELD-024**: queue_position field → Option<usize>
- **B-FIELD-025**: predicted_start_ms field → Option<u64>

### Engine/Model Fields
- **B-FIELD-030**: engine field → Option<String>
- **B-FIELD-031**: engine_version field → Option<String>
- **B-FIELD-032**: model_ref field → Option<String>
- **B-FIELD-033**: device field → Option<String>

### Performance Fields
- **B-FIELD-040**: tokens_in field → Option<u64>
- **B-FIELD-041**: tokens_out field → Option<u64>
- **B-FIELD-042**: decode_time_ms field → Option<u64>

### Provenance Fields
- **B-FIELD-050**: emitted_by field → Option<String>
- **B-FIELD-051**: emitted_at_ms field → Option<u64>
- **B-FIELD-052**: trace_id field → Option<String>
- **B-FIELD-053**: span_id field → Option<String>
- **B-FIELD-054**: parent_span_id field → Option<String>
- **B-FIELD-055**: source_location field → Option<String>

### Default Values
- **B-FIELD-060**: When NarrationFields::default() → actor = ""
- **B-FIELD-061**: When NarrationFields::default() → action = ""
- **B-FIELD-062**: When NarrationFields::default() → target = ""
- **B-FIELD-063**: When NarrationFields::default() → human = ""
- **B-FIELD-064**: When NarrationFields::default() → all Option fields = None

---

## Feature Flag Behaviors

### `otel` Feature
- **B-FEAT-001**: When `otel` feature enabled → opentelemetry dependency included
- **B-FEAT-002**: When `otel` feature disabled → opentelemetry dependency excluded
- **B-FEAT-003**: When `otel` feature enabled → extract_otel_context() uses real OTEL API
- **B-FEAT-004**: When `otel` feature disabled → extract_otel_context() returns (None, None, None)

### `test-support` Feature
- **B-FEAT-010**: When `test-support` feature enabled → CaptureAdapter available in non-test builds
- **B-FEAT-011**: When `test-support` feature disabled → CaptureAdapter only in test builds
- **B-FEAT-012**: When `test-support` feature enabled → capture::notify() called in narrate()
- **B-FEAT-013**: When `test-support` feature disabled → capture::notify() not called in narrate()
- **B-FEAT-014**: When in test build → CaptureAdapter always available (regardless of feature)

### Default Features
- **B-FEAT-020**: When no features specified → only core narration available
- **B-FEAT-021**: When no features specified → OTEL integration disabled
- **B-FEAT-022**: When no features specified → test capture only in tests

---

## Summary Statistics

**Total Behaviors**: 200+

**By Category**:
- Core Narration: 20 behaviors
- Auto-Injection: 17 behaviors
- Redaction: 26 behaviors
- Test Capture: 42 behaviors
- OpenTelemetry: 17 behaviors
- HTTP Headers: 24 behaviors
- Field Taxonomy: 25 behaviors
- Feature Flags: 13 behaviors

**Coverage Requirements**:
- All behaviors MUST have at least one BDD scenario
- Critical behaviors (B-CORE-*, B-AUTO-*, B-RED-*) MUST have multiple scenarios
- Edge cases (errors, None values, invalid inputs) MUST be tested
- Feature flag combinations MUST be tested

---

## Next Steps

1. Create `.feature` files for each category
2. Implement step definitions in `src/steps/`
3. Create `World` struct with test state
4. Add BDD runner binary
5. Achieve 100% behavior coverage
