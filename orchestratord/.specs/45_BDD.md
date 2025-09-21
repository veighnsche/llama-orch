# orchestratord — BDD Delegation (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Delegation

- Cross-crate flows covered by `test-harness/bdd`. This doc defines boundaries and references Minimal Auth scenarios.

## Boundaries

- BDD uses only public interfaces (HTTP control/data plane, CLI) and MUST NOT mutate internal state directly.
- Minimal Auth seam MAY be enabled; scenarios cover both unauthenticated and loopback-exempt modes when configured.
- SSE runs MUST validate event ordering, error framing, and cancel behavior; micro-batching MAY be covered via tags.

## Feature Catalog (examples)

- Admission & Backpressure
  - Submit within capacity → stream tokens and end
  - Queue full → 429 with `Retry-After` and `X-Backoff-Ms`

- SSE Streaming & Error Frames
  - `started → token* → metrics? → end` ordering
  - Mid-stream engine error → `event:error { code, retriable, retry_after_ms? }` then termination

- Cancel & Disconnect
  - Client cancel terminates stream deterministically
  - Client disconnect triggers best-effort cancel on server side

- Capabilities & Budgets
  - `GET /v1/capabilities` includes additive fields (engine, versions, ctx_max, max_tokens_out, concurrency, supported_workloads, api_version)
  - Over-budget submit rejected cleanly (400/429 taxonomy); no silent truncation

## Execution

- Run: `cargo test -p test-harness-bdd -- --nocapture`
- Scope to a directory/file via `LLORCH_BDD_FEATURE_PATH`
- Fails on undefined/skipped steps (`fail_on_skipped()`)

## Traceability

- ORCH‑3406..3409: SSE error frames
- ORCH‑3093/3094/3096: Capabilities completeness
- ORCH‑3016: No silent truncation
- ORCH‑3330/3331: Error-class taxonomy

## Refinement Opportunities

- Add links to specific feature files once organized.
