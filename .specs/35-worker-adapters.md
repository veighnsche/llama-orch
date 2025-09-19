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
- Timeouts/retries: adapters enforce request bounds and backoff with jitter.
- Error taxonomy: map engine errors to `WorkerError` and preserve cause via logs.
- Streaming: preserve `started → token* → end` ordering; `metrics` frames optional.

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

- Shared HTTP client/retry helper crate for adapters.
- Optional thin “token” hot-path type to reduce per-event allocations.
- Capability schema for ctx_max/workloads/features across adapters.
