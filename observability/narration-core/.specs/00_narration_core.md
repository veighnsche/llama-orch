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

## Integration Points

- Orchestratord: admission, placement, stream start/end/cancel hooks.
- Adapter Host: submit/cancel wrappers.
- Provisioners and manager: preflight/build/spawn/readiness narration.

## Observability

- Narration MUST co-exist with JSON logs and pretty console; JSON is authoritative in CI.
- Correlate with `X-Correlation-Id` where present.

## Testing & Proof Bundles

- Include narration coverage excerpts in proof bundles (sampled lines) and, for streams, SSE transcripts with correlation IDs.

## Refinement Opportunities

- Add a small macro for ergonomics in crates that do not want to depend directly on a facade crate.
- Provide structured examples/snippets for common events.
- Optional sampling controls (rate-limit narration under load).
