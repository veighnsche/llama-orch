# http-util — Unit Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- HTTP client builder defaults
- Retry/backoff policy (capped + jitter)
- Header redaction helpers

## Test Catalog

- Builder Defaults
  - Timeouts, HTTP/2 keep-alive, user-agent/header defaults set as expected

- Retry/Backoff Policy
  - Given base delay and cap, computed delays fall within `[base, cap]`
  - Jitter distribution shows variance; deterministic mode available via seeded RNG for tests
  - Non-retriable status codes (e.g., 400, 401) do not trigger retries

- Redaction Helpers
  - Authorization and secret-bearing headers masked in debug formatting
  - Query/body redaction for known sensitive keys

## Execution

- `cargo test -p worker-adapters-http-util -- --nocapture` (adjust package name as needed)

## Traceability

- Consumed by `worker-adapters/*` crates for streaming and retries
- Metrics alignment per `/.specs/metrics/otel-prom.md`

## Refinement Opportunities

- Property tests for retry jitter distribution.
