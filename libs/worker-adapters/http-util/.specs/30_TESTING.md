# http-util — Testing Overview (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Client builder defaults; capped+jittered retries; streaming decode helpers.

## Test Catalog

- Unit
  - Builder defaults (timeouts, HTTP/2 keep-alive, headers)
  - Retry/backoff policy: cap, base, jitter distribution, max attempts
  - Redaction helpers for logs
  - SSE chunk parsing utilities (if provided here)

- Integration
  - Stub server exercising retries (429/503/timeouts) and ensuring policy adherence
  - Streaming decode interop with chunked responses

- Contract
  - Shared fixtures for SSE chunk shapes consumed by adapters

## Execution

- `cargo test -p worker-adapters-http-util -- --nocapture` (actual package name may differ; run from workspace root with `-p http-util`)

## Traceability

- Metrics naming and labels alignment: `/.specs/metrics/otel-prom.md`

## Refinement Opportunities

- Add deterministic retry timeline tests.
