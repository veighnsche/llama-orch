# Worker Adapters — Component Specification (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## 0) Purpose & Scope

Worker adapters translate orchestrator requests into engine‑native calls and stream token events back. They implement a common trait to unify health, capacity/props, submission/cancel, and engine version exposure across engines (llama.cpp, vLLM, TGI, Triton, OpenAI, etc.).

In scope (shared across adapters):
- Health/readiness and props/capacity signals.
- Submit/cancel streaming API mapping to engine‑native endpoints.
- Error taxonomy mapping and retry/backoff boundaries.
- Capability reporting (ctx_max, workloads, features) and engine_version.
- Determinism knobs exposure (seed handling/pinning where applicable).
- Metrics/logging fields and cardinality limits.

Out of scope:
- Placement and admission policies (orchestrator‑core).
- Engine provisioning (engine‑provisioner) and model staging (model‑provisioner).

## 1) Normative Requirements (RFC‑2119)

- [ADAPT‑3000] Each adapter MUST implement `WorkerAdapter` from `worker-adapters/adapter-api`:
  - `health() -> WorkerHealth { live, ready }` (Ready implies dispatchable).
  - `props() -> WorkerProps { slots_total?, slots_free? }`.
  - `submit(TaskRequest) -> TokenStream` (started → token(s) → end; metrics frames optional).
  - `cancel(task_id)` best‑effort.
  - `engine_version() -> String` provides a stable semantic version.
- [ADAPT‑3001] Token events MUST use `TokenEvent { kind, data }` kinds: `started`, `token`, `metrics`, `end`, `error`.
- [ADAPT‑3002] Errors MUST be mapped to `WorkerError` taxonomy (deadline, pool unavailable, decode timeout, reset, adapter, internal).
- [ADAPT‑3003] Determinism signals (engine_version, seed, sampler profile) MUST be propagated in `started`/`end` frames or logs per README_LLM.
- [ADAPT‑3004] Metrics and logs MUST follow `.specs/metrics/otel-prom.md` and README_LLM fields (job_id, session_id, engine, engine_version, pool_id, replica_id, queue_position, predicted_start_ms, tokens_in, tokens_out, decode_time_ms).
- [ADAPT‑3005] Adapters MUST time‑bound network requests and SHOULD retry idempotent operations with caps and jitter.

## 2) Data Types & Semantics

- `WorkerHealth { live, ready }` — Ready implies the adapter can accept a `submit`.
- `WorkerProps { slots_total?, slots_free? }` — optional capacity signals for placement.
- `TokenEvent` kinds:
  - `started`: metadata including engine_version, prompt tokens_in, etc.
  - `token`: `{ t: string, i: index }` for each decoded token or delta.
  - `metrics`: optional, `{ latency_first_token_ms, latency_decode_ms, tokens_out }`.
  - `end`: summary with tokens_out and totals.
  - `error`: terminal error with mapped taxonomy.

## 3) Interfaces & Contracts

- Provided: `WorkerAdapter` trait implementation per adapter.
- Consumed: `contracts/api-types::TaskRequest` shape for submission; `orchestratord` uses adapters through trait objects.

## 4) Observability

- Emit logs with fields per README_LLM.
- Metrics naming/labels as per `.specs/metrics/otel-prom.md`.

## 5) Security

- Do not log secrets; redact tokens/keys from URLs and headers.
- Respect network egress policy; honor timeouts.

## 6) Testing & Proof Bundle

- Per‑adapter unit tests for mapping, error taxonomy, and retries.
- Cross‑crate BDD covers end‑to‑end flows with `orchestratord`.

## 7) Open Questions

- Standardize adapter capability schema and transport.
- How to represent KV‑reuse and warmup hooks generically?

## 8) Refinement Opportunities

- Shared adapter utilities (HTTP client, retry policy, streaming decoder).
- Feature‑flagged metrics enrichment and structured logging helpers.
