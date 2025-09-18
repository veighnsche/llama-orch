# Metrics Contract SPEC — Names/Labels Budgets (v1.0)

Status: Stable (draft)
Applies to: `test-harness/metrics-contract/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-METRICS-7xxx`.

## 1) Names/Labels

- [OC-METRICS-7101] Metric names and required labels MUST conform to `ci/metrics.lint.json`.
- [OC-METRICS-7102] Label cardinality budgets MUST be documented and enforced.

## 2) Traceability

- Linter: [ci/metrics.lint.json](../ci/metrics.lint.json)
- Tests: to be created under `test-harness/metrics-contract/` (e.g., `tests/metrics_lint.rs`). For now, the linter (`ci/metrics.lint.json`) is authoritative.

## 3) SSE Metrics Signals (Client Planning)

- [OC-METRICS-7110] The `metrics` SSE event payloads SHOULD include additive fields helpful for client-side planning under load. Example fields (non-exhaustive, non-breaking if extended):
  - `on_time_probability: number` — probability of on-time completion given current load.
  - `queue_depth: integer` — current queue size for the serving pool/replica.
  - `kv_warmth: boolean` — whether KV is warm for the session.
- [OC-METRICS-7111] When per-session budgets (token/time/cost) are enabled, budget remaining SHOULD be surfaced either in `metrics` events or as response headers to allow clients to adapt.

## Refinement Opportunities

- Define canonical bucket boundaries for `latency_first_token_ms` and `latency_decode_ms` for home profile reference hardware.
- Consider a stable schema for `metrics` SSE payloads to ease SDK evolution while allowing additive fields.
- Provide a compact, cardinality-aware label set for admission-level counters that preserves utility.
