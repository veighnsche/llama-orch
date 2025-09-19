# Worker Adapters — HTTP Util (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

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

## Contracts

- `build_client(Config) -> HttpClient` — constructs a client with configured timeouts and connection reuse.
- `with_retries<F>(f, policy) -> Result<T>` — executes an idempotent request with capped, jittered retries.
- `stream_decode(body, sink) -> Result<()>` — decodes streaming responses into `TokenEvent` with minimal allocations.
- Redaction: functions to mask `Authorization`, `X-API-Key`, and known token patterns from logs.

## Performance & Determinism

- The streaming decoder SHOULD avoid per-token heap allocations; use reusable buffers.
- Ordering MUST preserve `started → token* → end`; `metrics` frames are optional and additive.

## Security

- TLS verification MUST be on by default; proxies MUST be explicit.
- Secrets MUST NOT be logged; headers must be redacted at error level.

## References

- Root norms: `/.specs/35-worker-adapters.md`.
- Adapter Host integration: `adapter-host/.specs/00_adapter_host.md`.

## Refinement Opportunities

- Add unified error taxonomy helpers for mapping upstream HTTP errors to `WorkerError`.
- Provide examples for typical upstream APIs (OpenAI/vLLM/TGI) and streaming idiosyncrasies.
- Optional metrics hooks for request/latency counters if needed by adapters.
