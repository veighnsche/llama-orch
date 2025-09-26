# worker-adapters — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Crate-level behavior: trait conformance, streaming framing, retries/backoff via shared HTTP util, error taxonomy mapping.

## Delegation

- Cross-crate flows covered by root BDD; this crate validates per-adapter behavior locally.

## Test Catalog

- Unit
  - Trait conformance and error taxonomy helpers
  - Retry/backoff policy wiring to `http-util` (capped + jittered)
  - Redaction utilities for headers/params

- Integration
  - End-to-end streaming with stub upstreams (happy path, mid-stream error, timeout)
  - Cancel propagation and disconnect handling

- Contract
  - Provider verify against shared adapter API shapes (token/event frames, error bodies)

## Execution & Tooling

- Run: `cargo test -p worker-adapters -- --nocapture`
- Prefer deterministic jitter in tests (seeded) via `http-util` test hook when available

## Traceability

- Error taxonomy: ORCH‑3330/3331
- SSE error frame (via orchestrator data plane): ORCH‑3406..3409
- Metrics names/labels: `/.specs/metrics/otel-prom.md`

## Refinement Opportunities

- Add stress tests for low-allocation streaming decoder paths.
