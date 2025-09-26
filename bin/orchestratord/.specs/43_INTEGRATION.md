# orchestratord — Integration Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Adapter Host facade wiring; placement → streaming → cancel path; SSE buffering.

## Test Catalog

- Admission → Dispatch wiring
  - Happy path: `POST /v1/tasks` admits and dispatches to Adapter Host facade; stream follows `started → token* → metrics? → end`.
  - Backpressure path: queue full produces 429 with `Retry-After` and `X-Backoff-Ms`; no stream is opened.

- SSE data-plane behavior
  - Micro-batching parameterization: verify event grouping vs per-token emission.
  - Mid-stream failure maps to SSE `event:error` with `{ code, retriable, retry_after_ms? }` and terminates stream.
  - Cancel request mid-stream closes adapter and terminates SSE cleanly.

- Capability discovery → admission guardrails
  - Capabilities returned include ctx/token bounds; over-budget requests are rejected at admission (400/429 taxonomy).

- Minimal Auth seam (if enabled)
  - Authenticated vs unauthenticated flows; correlation id propagation.

## Fixtures & Mocks

- In-process Adapter Host facade with stub worker adapter that can emit:
  - Deterministic token streams
  - Injected mid-stream errors
  - Delay/timeout to exercise buffering and cancel-on-disconnect
- Deterministic clock or timers for predictable buffering assertions.

## Execution

- Run: `cargo test -p orchestratord -- --nocapture`
- Provider verify: pair with `cli/consumer-tests` pact when applicable; record SSE transcripts for proof bundles.

## Traceability

- ORCH‑3406..3409: SSE error frame semantics.
- ORCH‑3093/3094/3096: Capability payload completeness and alignment.
- ORCH‑3016: No silent truncation of budgets.

## Refinement Opportunities

- Add micro-batch parameterization for SSE tests.
