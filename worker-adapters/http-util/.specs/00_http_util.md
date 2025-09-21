# Worker Adapters — HTTP Util (v0)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-19
Conformance language: RFC‑2119
Applies to: `worker-adapters/http-util/`

## Purpose & Scope

Provide a shared HTTP client and helpers for adapter crates to ensure consistent timeouts, retries with jittered backoff, HTTP/2 keep-alive, streaming decode utilities, and sensitive header redaction.

In scope:
- HTTP client builder with sane defaults (timeouts, connection pool, HTTP/2 where supported).
- Retry/backoff helpers with caps and jitter for idempotent requests.
- Streaming decode helpers for SSE-like or chunked token streams (low-allocation fast path guidance).
- Redaction helpers for logs/errors.

Out of scope:
- Full-blown SDK or orchestrator HTTP client.
- Metrics emission (adapters emit metrics; this crate only provides hooks).

## Contracts (normative)

- [HTU-1001] `build_client(Config) -> HttpClient` MUST construct a client with:
  - connect timeout default 5s (configurable),
  - request timeout default 30s (configurable per call),
  - HTTP/2 enabled where server supports it; otherwise gracefully fallback to HTTP/1.1,
  - connection pool reuse with keep‑alive, and TLS verification ON by default.
- [HTU-1002] `with_retries<F>(f, policy) -> Result<T>` MUST implement capped, jittered retries for idempotent requests only. The default policy:
  - base delay 100ms, multiplier 2.0, max delay 2s, max attempts 4,
  - full jitter: `delay_n = rand(0, min(max_delay, base*multiplier^n))`.
- [HTU-1003] `stream_decode(body, sink) -> Result<()>` MUST decode streaming responses into `TokenEvent` preserving ordering `started → token* → end` and SHOULD use a low‑allocation path (reusable buffers, avoid per‑token heap allocations).
- [HTU-1004] Redaction helpers MUST mask `Authorization`, `X-API-Key`, and common token patterns in headers and error messages; logs MUST NOT contain raw secrets at error or info levels.

## Performance & Determinism

- The streaming decoder SHOULD avoid per-token heap allocations; use reusable buffers.
- Ordering MUST preserve `started → token* → end`; `metrics` frames are optional and additive.

## Security

- TLS verification MUST be on by default; proxies MUST be explicit.
- Secrets MUST NOT be logged; headers must be redacted at error level.

## References

- Root norms: `/.specs/35-worker-adapters.md`.
- Adapter Host integration: `adapter-host/.specs/00_adapter_host.md`.

## Testing Expectations

- See `./33_INTEGRATION.md` for a test matrix covering client initialization, retry/backoff behavior (jitter bounds), streaming decode conformance, and header redaction under error conditions.

## Refinement Opportunities

- Add unified error taxonomy helpers for mapping upstream HTTP errors to `WorkerError`.
- Provide examples for typical upstream APIs (OpenAI/vLLM/TGI) and streaming idiosyncrasies.
- Optional metrics hooks for request/latency counters if needed by adapters.
