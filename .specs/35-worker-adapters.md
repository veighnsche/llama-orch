# Worker Adapters — Root Overview

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

Unify adapter expectations across engines (llama.cpp, vLLM, TGI, Triton, OpenAI) and point to per-adapter specs. Adapters translate orchestrator requests into engine-native APIs via a common trait and stream tokens back while enforcing timeouts/retries and error taxonomy.

This root spec complements the central in-repo adapter spec found at `worker-adapters/.specs/00_worker_adapters.md` and the per-adapter root specs (`40-*`, `41-*`, `42-*`, `43-*`, `44-*`).

## Provided Contracts (summary)

- `worker-adapters/adapter-api`: trait `WorkerAdapter` with `health`, `props`, `submit`, `cancel`, `engine_version` and streaming via `TokenEvent`.
- Shared logging/metrics fields as per `README_LLM.md` and `.specs/metrics/otel-prom.md`.

## Consumed Contracts (summary)

- Used by `orchestratord` to serve data-plane requests and cancellations.
- Pool readiness/perf hints reported via `pool-managerd` and surfaced through orchestrator health/capabilities endpoints.

## Cross-Cutting Requirements

- Determinism: replicate `engine_version`, `model_digest`, and sampler profile (when applicable) so streams are reproducible per replica.
- Shared HTTP util (normative for HTTP-based adapters):
  - Adapters MUST use `worker-adapters/http-util` for constructing HTTP clients with consistent timeouts, retries with capped jitter, connection reuse (HTTP/2 keep-alive), and redaction of sensitive headers.
  - Error taxonomy from upstream MUST be mapped to `WorkerError` consistently using the shared helpers.
- Streaming decode path (normative):
  - Streaming MUST preserve `started → token* → end` ordering; `metrics` frames optional and additive.
  - Implementations SHOULD use a low-allocation hot path for token events (avoid per-token heap allocations where possible) and MUST remain deterministic with respect to ordering and token boundary handling.

## Per-Adapter Supplements

- 40-worker-adapters-llamacpp-http.md — llama.cpp HTTP mapping
- 41-worker-adapters-vllm-http.md — vLLM HTTP mapping
- 42-worker-adapters-tgi-http.md — TGI HTTP mapping
- 43-worker-adapters-triton.md — Triton mapping
- 44-worker-adapters-openai-http.md — OpenAI HTTP mapping

## Testing Ownership

- Per-adapter unit/behavior tests live inside each adapter crate.
- Root BDD harness asserts only cross-crate flows (admission→stream, cancel). Determinism suite may bind adapters via the trait.

## Refinement Opportunities

- Optional thin “token” hot-path type to reduce per-event allocations.
- Capability schema for ctx_max/workloads/features across adapters.

## References

- See `adapter-host/.specs/00_adapter_host.md` for the in-process facade used by orchestrator integration.
- See `worker-adapters/http-util/.specs/00_http_util.md` for shared HTTP client and retry/backoff/redaction requirements.
