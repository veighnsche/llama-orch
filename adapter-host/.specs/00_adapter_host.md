# adapter-host — Component Spec (in-process)

Status: Stable (draft)
Owner: @llama-orch-maintainers
Date: 2025-09-19
Conformance language: RFC‑2119
Applies to: `adapter-host/`

## 0) Purpose & Scope

Provide an in-process registry and facade for `WorkerAdapter` implementations used by `orchestratord`. Centralizes `submit`/`cancel` routing, correlation ID propagation, retries/backoff via `worker-adapters/http-util`, and capability snapshots without adding network hops.

## 1) Normative API & Semantics

- [AH-1001] Registry API MUST support bind/rebind of `(pool_id, replica_id) -> Adapter` on preload/reload and during drain/reload cycles. Rebind MUST be atomic from the caller perspective.
- [AH-1002] Facade MUST expose:
  - `submit(pool_id, TaskRequest) -> Stream<TokenEvent>`
  - `cancel(pool_id, task_id) -> Result<()>`
  - `health(pool_id) -> WorkerHealth`
  - `props(pool_id) -> WorkerProps`
- [AH-1003] `submit` MUST propagate `X-Correlation-Id` (or generate a UUIDv4) to downstream adapters and log contexts.
- [AH-1004] `submit`/`cancel` MUST be race-free with respect to stream termination: no tokens may be emitted after a successful cancel acknowledgment window.
- [AH-1005] Errors returned by adapters MUST map to the shared `WorkerError` taxonomy and include advisory fields (`retriable`, `retry_after_ms?`) when available from upstream or policy.
- [AH-1006] The facade SHOULD provide optional per-pool overrides for timeouts and retry policy that wrap adapter calls via `http-util`.
- [AH-1007] Optional pinning: when `TaskRequest` includes a valid `pool_id` (pin override) and policy allows, the host MUST route to the specified pool and skip placement. Otherwise, normal placement applies. Invalid pins MUST be rejected deterministically with `INVALID_PARAMS` (see root specs for override semantics).

## 2) Observability & Narration

- [AH-2001] Logs MUST include standard fields (`job_id`, `session_id`, `engine`, `engine_version`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out`, `decode_time_ms`) where applicable.
- [AH-2002] Emit short human‑readable narration at `submit`, `stream start/end`, `cancel`, and error points; do not leak secrets; use redaction helpers from `http-util`.
- [AH-2003] Metrics SHOULD include counters for requests, cancels, retries, and histograms for first‑token/decode latencies if measured at this layer.

## 3) Security

- [AH-3001] Secrets (Authorization headers, API keys) MUST NOT be logged; all error logs MUST pass through redaction helpers.
- [AH-3002] Network egress policy (if any at this layer) MUST be respected; otherwise defer to adapter policy.

## 4) Testing Expectations

- Unit: registry bind/rebind atomicity, routing correctness, correlation ID propagation, cancel race‑free semantics.
- Integration: end‑to‑end streaming via a stub adapter verifying ordering (`started → token* → end`), redaction, and error taxonomy mapping passthrough.

## 5) Traceability

- Root specs: `/.specs/35-worker-adapters.md`, `/.specs/20-orchestratord.md` (§SSE)
- Related crates: `worker-adapters/http-util/.specs/00_http_util.md`

## Refinement Opportunities

- Circuit breaker policy and capability cache integration.
- Per-pool timeouts and micro-batch hints surfaced to orchestrator.
