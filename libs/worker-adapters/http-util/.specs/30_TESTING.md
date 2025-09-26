# http-util — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Philosophy & Policy Alignment

- Tests reflect the contract (Spec → Contract → Tests → Code). See `/.docs/testing/TESTING_POLICY.md` and `/.docs/testing/test-case-discovery-method.md`.
- No skipped tests in committed code; if an environment constraint applies, fail with a clear message.
- Harnesses exercise public surfaces only; for this crate, that means the crate’s public API (client, retry, streaming, redact). No private state shims.
- Determinism by default: jitter tests must support a seeded RNG path to make expectations reproducible.

## Scope

- Client builder defaults; capped + jittered retries; streaming decode helpers; header/secret redaction; error taxonomy mapping hooks per `./40_ERROR_MESSAGING.md`.

## Test Catalog

- Unit (stable IDs; see `./31_UNIT.md` for details)
  - [HTU-UNIT-3101] Builder defaults (timeouts, HTTP/2 keep-alive, TLS verify on)
  - [HTU-UNIT-3102] Retry/backoff policy (cap, base, jitter distribution, max attempts)
  - [HTU-UNIT-3103] Redaction helpers (Authorization/X-API-Key/token patterns)
  - [HTU-UNIT-3104] Streaming decode utilities (newline/SSE-like) low‑alloc path

- Integration (stable IDs; see `./33_INTEGRATION.md`)
  - [HTU-INT-3302] Retry/backoff adherence against stubbed 429/503/timeouts
  - [HTU-INT-3303] Streaming decode conformance and ordering
  - [HTU-INT-3304] Secret redaction in logged errors
  - [HTU-INT-3305] HTTP/2 keep‑alive reuse under load

- Contract (fixtures)
  - Shared token stream shapes for adapters; ensure compatibility across adapters

## Execution

- Local: `cargo test -p worker-adapters-http-util -- --nocapture`
- Deterministic jitter: tests SHOULD accept a seed (e.g., feature flag or env `HTTP_UTIL_TEST_SEED`) to fix RNG for expectations.
- CI: use a local stub (e.g., wiremock) to avoid external calls; no network dependency.

## Traceability

- Contracts: `./00_http_util.md`
- Unit details: `./31_UNIT.md`
- Integration matrix: `./33_INTEGRATION.md`
- Error taxonomy/redaction: `./40_ERROR_MESSAGING.md`
- Root worker adapters norms: `/.specs/35-worker-adapters.md`
- Metrics naming and labels alignment: `/.specs/metrics/otel-prom.md`

## Refinement Opportunities

- Add deterministic retry timeline tests.
 - Expand proof bundle capture: retry timelines, redaction snapshots, and sample streaming transcripts.
