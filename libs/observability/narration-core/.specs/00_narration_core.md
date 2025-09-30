# Narration Core — Component Specification (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

Provide a minimal, reusable facade for emitting human-readable narration strings alongside structured logs across the workspace. Standardize the field names, redaction gates, and capture points for proof bundles.

In scope:
- Emission helpers for narration at key events (admission, placement, stream start/end, cancel).
- Redaction helpers that integrate with existing logging setup.
- Optional capture hooks for SSE transcripts and coverage sampling.

Out of scope:
- Full logging setup or tracing subscriber configuration.
- Metrics emission (covered elsewhere).

## Contracts (Facade API)

- `narrate(event: &str, fields: &NarrationFields)` — emits a short narration line and structured fields.
- `NarrationFields { human: String, correlation_id: Option<String>, job_id: Option<String>, session_id: Option<String>, pool_id: Option<String>, replica_id: Option<String> }`.
- Redaction policy: helpers MUST strip or mask secrets (tokens, API keys) and PII.

## Provenance

Every narration event SHOULD include provenance metadata for debugging and audit trails:

- **Emitter identity**: Service name and version (e.g., `orchestratord@0.1.0`, `pool-managerd@0.1.0`)
- **Emission timestamp**: Milliseconds since epoch when the narration was emitted
- **Trace context**: Distributed trace ID if available (for cross-service correlation beyond correlation_id)
- **Source location**: Optional file/line for development builds (stripped in release)

Provenance fields:
- `emitted_by: Option<String>` — Service name and version (e.g., "orchestratord@0.1.0")
- `emitted_at_ms: Option<u64>` — Unix timestamp in milliseconds
- `trace_id: Option<String>` — Distributed trace ID (OpenTelemetry compatible)
- `span_id: Option<String>` — Span ID within the trace
- `source_location: Option<String>` — File:line for dev builds (e.g., "data.rs:155")

Provenance enables:
- **Audit trails**: Who emitted what, when
- **Version correlation**: Track behavior changes across deployments
- **Distributed tracing**: Link narration to OpenTelemetry traces
- **Debug context**: Jump to source in development

## Integration Points

- Orchestratord: admission, placement, stream start/end/cancel hooks.
- Adapter Host: submit/cancel wrappers.
- Provisioners and manager: preflight/build/spawn/readiness narration.

## Observability

- Narration MUST co-exist with JSON logs and pretty console; JSON is authoritative in CI.
- Correlate with `X-Correlation-Id` where present.

## Testing & Proof Bundles

- Include narration coverage excerpts in proof bundles (sampled lines) and, for streams, SSE transcripts with correlation IDs.

## Cloud Profile Requirements (v0.2.0+)

**CRITICAL**: For distributed deployments (CLOUD_PROFILE), narration-core has additional mandatory requirements. See `CLOUD_PROFILE_NARRATION_REQUIREMENTS.md` for full details.

### Summary

1. **OpenTelemetry Integration** (REQUIRED):
   - Auto-extract `trace_id` and `span_id` from current OTEL context
   - Provide `narrate_with_otel_context()` helper
   - Support W3C Trace Context propagation

2. **Service Identity** (REQUIRED):
   - `emitted_by` field is mandatory (not optional)
   - Auto-inject via `narrate_auto()` helper
   - Format: `{service_name}@{version}`

3. **HTTP Header Propagation** (REQUIRED):
   - Extract correlation/trace IDs from HTTP headers
   - Inject correlation/trace IDs into HTTP headers
   - Compatible with `axum`, `reqwest`, `hyper`

4. **Log Aggregation** (REQUIRED):
   - JSON output with consistent field names
   - Explicit `level` field
   - Compatible with Loki, Elasticsearch, CloudWatch

5. **Cross-Service Correlation** (REQUIRED):
   - Correlation IDs MUST propagate across all HTTP calls
   - Trace IDs MUST link to OpenTelemetry spans
   - BDD tests MUST verify correlation across services

**Rationale**: In distributed deployments, a single user request touches 3+ services across 2+ machines. Without proper narration with correlation, debugging is impossible.

## Refinement Opportunities

- Add a small macro for ergonomics in crates that do not want to depend directly on a facade crate.
- Provide structured examples/snippets for common events.
- Optional sampling controls (rate-limit narration under load).
- Automatic provenance injection via macro (capture service name/version at compile time).
- Source location capture in debug builds only (zero overhead in release).
- OpenTelemetry trace context propagation (auto-extract from current span). **← PROMOTED TO REQUIRED FOR CLOUD_PROFILE**
- Provenance aggregation for "who touched this request" summaries.
- Callback webhooks for real-time narration streaming (future optimization).
