# worker-adapters/http-util — Shared HTTP Client & Helpers

Status: Draft
Owner: @llama-orch-maintainers

Purpose
- Provide a tuned shared `reqwest::Client` for all adapters (keep-alive, pool tuning, timeouts).
- Offer retry/backoff helpers with jitter and streaming decode utilities.
- Centralize secret redaction behaviors for adapter logs.

Spec Links
- `.specs/proposals/2025-09-19-adapter-host-and-http-util.md` (ORCH-3610..3613)

Detailed behavior (High / Mid / Low)

- High-level
  - Exposes a single, shared HTTP client and helper layer consumed by HTTP-based adapters to ensure consistent transport behavior and error mapping.

- Mid-level
  - Client configuration: HTTP/2 preferred with keep-alive, connection pool sizing, sane connect/read/write timeouts.
  - Retries: capped attempts with jittered exponential backoff applied only to idempotent requests; instrumentation emits retry counts.
  - Streaming decode: provides low-allocation SSE/JSON line decoders that surface `{started, token, metrics, end, error}` frames in order.
  - Redaction: strips/obfuscates sensitive headers (e.g., `Authorization`) from errors and logs.
  - Metrics/logging: integrates with narration and metrics contracts to emit standard fields and histograms as applicable.

- Low-level
  - Client is constructed once (lazy/static) and reused; per-request knobs allow overriding timeouts safely.
  - Errors map to a shared adapter `WorkerError` taxonomy; includes HTTP status, retriable hints, and correlation IDs when available.
  - Streaming parsers reuse internal buffers to minimize allocations and avoid partial UTF-8 issues.

Refinement Opportunities
- Add per-request override knobs (timeouts/pool hints) with safe defaults.
- Provide a zero-copy SSE/stream decoder for token deltas.
