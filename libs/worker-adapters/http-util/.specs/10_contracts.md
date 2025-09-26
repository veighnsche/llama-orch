# HTTP Util — Contracts (v0)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-26
Conformance language: RFC‑2119
Applies to: `libs/worker-adapters/http-util/`

## Purpose

Define the contract that `http-util` PROVIDES to adapter crates, and the contract it EXPECTS from
its consumers. This spec complements `./00_http_util.md` (normative behaviors) and guides
interoperability across all HTTP-based adapters.

## Provided Contracts

- [HTU-CAP-1001] HTTP client builder (`HttpClientConfig`, `make_client`) with sane defaults:
  - connect timeout default 5s (configurable)
  - request timeout default 30s (configurable per request)
  - HTTP/2 enabled where supported; fallback to HTTP/1.1
  - TLS verification ON by default; proxies MUST be explicit
  - connection pool reuse with keep‑alive

- [HTU-CAP-1002] Retry/backoff utility (`with_retries`) for idempotent requests:
  - default policy base 100ms, multiplier 2.0, cap 2s, max attempts 4
  - full jitter: `delay_n = rand(0, min(cap, base*multiplier^n))`
  - error classification helpers for 5xx/429/connect/timeouts vs non‑retriable classes

- [HTU-CAP-1003] Streaming decode helpers (`streaming::decode_*`):
  - parse newline/SSE‑like token streams into ordered events
  - preserve ordering `started → token* → end`; `metrics` optional
  - minimize per‑token heap allocations via reusable buffers where practical

- [HTU-CAP-1004] Redaction helpers (`redact::headers/line`):
  - mask `Authorization`, `X-API-Key`, and common bearer token patterns
  - ensure error/debug formatting does not leak secrets (see `./40_ERROR_MESSAGING.md`)

- [HTU-CAP-1005] Optional auth header injection (`auth::with_bearer`):
  - helper to add `Authorization: Bearer <token>` on requests when adapters opt‑in

## Consumed Contracts (expectations on adapters)

- [HTU-EXP-1101] Provide configuration explicitly:
  - base URL(s), timeouts, retry policy knobs, and optional bearer token
  - `http-util` MUST NOT read env on its own; adapters own env parsing

- [HTU-EXP-1102] Use retries ONLY for idempotent requests:
  - non‑idempotent operations MUST NOT be wrapped with `with_retries`
  - map upstream non‑retriable errors (e.g., 400/401) directly without retrying

- [HTU-EXP-1103] Preserve determinism in tests:
  - when testing jitter/backoff, set a seed (e.g., `HTTP_UTIL_TEST_SEED`) to stabilize outcomes

- [HTU-EXP-1104] Redact secrets in logs/traces:
  - use redaction helpers for headers and lines before emitting at info/error level
  - NEVER log raw tokens or credentials; follow `./40_ERROR_MESSAGING.md`

- [HTU-EXP-1105] Do not bypass the public API:
  - avoid reaching into client internals; rely on provided builders and helpers
  - prefer HTTP/2 by default via `make_client`; do not disable TLS verification unless a test‑only context requires it with explicit justification

- [HTU-EXP-1106] Error taxonomy mapping:
  - when converting upstream errors into adapter `WorkerError`, follow the mapping guidance in `./40_ERROR_MESSAGING.md`
  - include `retriable` and `retry_after_ms` hints when known

## Data Exchange

- Inputs:
  - `HttpClientConfig` for `make_client` (timeouts, HTTP/2 preference, TLS policy)
  - Retry `Policy` for `with_retries` (base, multiplier, cap, attempts)
  - Streaming `Body`/reader for `streaming::decode_*`
- Outputs:
  - `reqwest::Client` (Send + Sync, pool reuse)
  - `Result<T>` from `with_retries` with classified error on failure
  - Ordered `TokenEvent`/frames from streaming decoders (`started → token* → end`, optional `metrics`)
## Concurrency & Performance

- [HTU-PERF-1301] Built clients are `Send + Sync` and intended to be reused across requests; consumers SHOULD clone handles rather than rebuild clients per call.
- [HTU-PERF-1302] Streaming decoder SHOULD avoid per‑token heap allocations; consumers SHOULD pass sinks/buffers that enable reuse where applicable.

## Security & Policy

- [HTU-SEC-1401] TLS verification MUST be enabled by default. Proxies and insecure modes MUST be opt‑in and test‑only with justification.
- [HTU-SEC-1402] Secrets MUST NOT be logged. Redaction MUST be applied to headers and messages at error/info levels.

## Observability

- This crate does not emit metrics itself.
- It provides redaction helpers so adapters can safely log retry decisions and error contexts.
- Adapters are expected to emit request/stream metrics per `/.specs/metrics/otel-prom.md`.

## Testing Expectations

- Unit: builder defaults, retry/backoff jitter (seedable), redaction, streaming decode (low‑alloc path).
- Integration: stubbed retries (429/5xx/timeouts), HTTP/2 keep‑alive reuse, streaming ordering, secret redaction.
- Determinism: provide seeded mode (e.g., `HTTP_UTIL_TEST_SEED`) in tests.
- See `./30_TESTING.md`, `./31_UNIT.md`, `./33_INTEGRATION.md`.

## Test Ownership

- This crate OWNS tests for its builders, retry policy, redaction, and streaming decoders.
- Adapter crates OWN mapping of engine responses to `WorkerError` while using this crate’s helpers.
- Cross‑crate flows are validated in adapter integration tests and root harnesses (BDD, E2E).

## Refinement Opportunities

- Provide shared error‑mapping helpers for common upstreams (OpenAI/vLLM/TGI) atop 40_ERROR_MESSAGING.
- Offer opt‑in metrics hooks (counters/latency) if adapters prefer a shared implementation.
- Add benchmarks for streaming decoder allocation profiles.

## Traceability & References

- Normative behaviors: `./00_http_util.md`
- Error taxonomy & redaction: `./40_ERROR_MESSAGING.md`
- Testing plans: `./30_TESTING.md`, `./31_UNIT.md`, `./33_INTEGRATION.md`
- Root norms: `/.specs/35-worker-adapters.md`, `/.specs/20-orchestratord.md`, `/.specs/metrics/otel-prom.md`
