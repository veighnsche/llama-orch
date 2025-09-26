# http-util — Unit Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- HTTP client builder defaults
- Retry/backoff policy (capped + jitter)
- Header redaction helpers
 - Streaming decode helpers (newline/SSE-like)

## Test Catalog

- Builder Defaults
  - Timeouts, HTTP/2 keep-alive, user-agent/header defaults set as expected

- Retry/Backoff Policy
  - Given base delay and cap, computed delays fall within `[base, cap]`
  - Jitter distribution shows variance; deterministic mode available via seeded RNG for tests (set `HTTP_UTIL_TEST_SEED`)
  - Non-retriable status codes (e.g., 400, 401) do not trigger retries

- Redaction Helpers
  - Authorization and secret-bearing headers masked in debug formatting
  - Query/body redaction for known sensitive keys

- Streaming Decode Helpers
  - Decoder preserves ordering and uses low-allocation path (buffer reuse where practical)

## Execution

- Run: `cargo test -p worker-adapters-http-util -- --nocapture`
- Determinism: set `HTTP_UTIL_TEST_SEED` to stabilize jitter expectations.
- No skipped tests in committed code; follow `/.docs/testing/TESTING_POLICY.md`.

## Proof Bundle Outputs (MUST)

Unit tests MUST generate the following artifacts under `libs/worker-adapters/http-util/.proof_bundle/`:

- `retry_timeline.jsonl` — events from HTU-UNIT-3102 (policy, attempt, delay_ms, seed)
- `redacted_errors.*` — snapshots from HTU-UNIT-3103 proving secret scrubbing
- `seeds.txt` — the RNG seed(s) used for deterministic jitter

## Traceability

- Consumed by `worker-adapters/*` crates for streaming and retries
- Metrics alignment per `/.specs/metrics/otel-prom.md`
- Error taxonomy and redaction per `./40_ERROR_MESSAGING.md`

## Refinement Opportunities

- Property tests for retry jitter distribution.
 - Snapshot tests (redacted logs) for proof bundles.
