# llamacpp-http — Integration Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- End-to-end streaming against a local llama.cpp server or stub; retries with capped+jittered backoff from http-util.

## Test Catalog

- Streaming Happy Path
  - GIVEN a stub or real llama.cpp endpoint emitting token deltas
  - WHEN the adapter submits a request
  - THEN events are observed `started → token* → metrics? → end` with strictly increasing token indices

- Mid-Stream Failure
  - GIVEN the server emits an error after N tokens
  - THEN adapter surfaces an error mapped to taxonomy; if routed via orchestrator data plane, SSE `event:error { code, retriable, retry_after_ms? }` is produced and stream terminates

- Timeouts & Retries
  - GIVEN slow or flaky upstream
  - WHEN timeouts occur
  - THEN `http-util` policy triggers bounded retries with capped+jittered backoff; non-retriable statuses fail fast

- Cancellation & Disconnect
  - GIVEN an in-flight stream
  - WHEN cancel is requested or client disconnects
  - THEN upstream cancel is attempted and local stream terminates cleanly

## Fixtures & Mocks

- Local stub server reproducing llama.cpp response shapes (delta tokens, errors, slow-start)
- Deterministic jitter via a seeded mode in tests; captured logs to assert redaction

## Execution

- `cargo test -p worker-adapters-llamacpp-http -- --nocapture` (package name may differ)
- Feature-gate real-engine tests; default CI uses stubs

## Traceability

- Error taxonomy: ORCH‑3330/3331
- SSE error frame (if surfaced via orchestrator): ORCH‑3406..3409
- `http-util` retry/backoff policy adoption

## Refinement Opportunities

- Timeout and cancellation propagation scenarios.
