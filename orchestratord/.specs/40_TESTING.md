# orchestratord — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- HTTP control/data endpoints, SSE emitter behavior, auth seams (Minimal Auth), error envelopes.

## Delegated

- Cross-crate BDD resides in `test-harness/bdd`.

## Test Catalog

- Control Plane (HTTP)
  - `GET /v1/capabilities` — includes additive fields per spec (engine, engine_version, sampler_profile_version, ctx_max, max_tokens_out, concurrency/slots, supported_workloads, api_version)
  - `POST /v1/tasks` — validation errors (400), backpressure (429 + headers), internal (5xx)
  - `DELETE /v1/tasks/{id}` — idempotency and correlation id propagation

- Data Plane (SSE)
  - Happy path: `started → token* → metrics? → end` ordering, optional micro-batching
  - Mid-stream error → SSE `event:error` body `{ code, retriable, retry_after_ms? }` and stream terminates
  - Cancel path and client disconnect handling

- Auth Seam (Minimal Auth)
  - Optional auth enabled: rejects unauthenticated (401/403) with redacted headers
  - Loopback-exempt behavior when configured

## Execution & Tooling

- Run: `cargo test -p orchestratord -- --nocapture`
- Provider verify: see `orchestratord/tests/provider_verify.rs` and `contracts/pacts/*`
- Record SSE transcripts for proof bundles; include both micro-batched and per-token forms

## Traceability

- SSE error frames: `/.specs/proposals/2025-09-20-orch-spec-change-close-home-profile-gaps.md` (ORCH‑3406..3409)
- Capabilities completeness: ORCH‑3093/3094/3096
- No silent truncation on budgets: ORCH‑3016
- Error class taxonomy: ORCH‑3330/3331

## Refinement Opportunities

- Add golden SSE transcripts (with/without micro-batching) for proof bundles.
