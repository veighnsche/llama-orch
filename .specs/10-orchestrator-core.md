# Orchestrator Core SPEC — Queue, Scheduling, Determinism (v1.0)

Status: Stable (draft)
Applies to: `orchestrator-core/`
Conformance language: RFC‑2119 (MUST/SHOULD/MAY)

## 0) Scope & Versioning

This SPEC covers the core queueing, placement, and determinism invariants implemented by `orchestrator-core`. Requirements are versioned as `OC-CORE-1xxx`.

## 1) Queue & Admission

- [OC-CORE-1001] Each Pool MUST expose a bounded FIFO queue per priority class.
- [OC-CORE-1002] Admission MUST reject when the queue is full according to configured policy (reject/drop-lru/shed-low-priority).
- [OC-CORE-1003] Enqueue MUST be O(1) amortized and MUST preserve request arrival order within the same priority.
- [OC-CORE-1004] Dequeue MUST prefer higher priority and MUST respect FIFO order within a priority class.
- [OC-CORE-1005] Cancellation MUST remove the task from the queue or mark the slot so it is not dispatched.

See: ../orchestrator-core/src/lib.rs, ../orchestrator-core/tests/props_queue.rs

## 2) Scheduling & Placement

- [OC-CORE-1010] Scheduler MUST dispatch only to Ready replicas.
- [OC-CORE-1011] Placement MUST respect device masks; cross‑mask spillover MUST NOT occur.
- [OC-CORE-1012] Least‑loaded placement MUST be used across replicas of the same replica set.
- [OC-CORE-1013] Session affinity SHOULD keep a session on its last good replica when possible.

## 3) Capacity & Guardrails

- [OC-CORE-1020] Context length MUST be ≤ model limit; otherwise reject before enqueue.
- [OC-CORE-1021] Token budget (prompt + generation) MUST be validated pre‑admission.
- [OC-CORE-1022] Watchdog MUST abort stuck Jobs with configurable wall/idle timeouts.
- [OC-CORE-1023] When per‑session budgets (token/time/cost) are configured, admission and/or scheduling MUST enforce remaining budget and reject infeasible requests with a typed error.
- [OC-CORE-1024] Budget accounting SHOULD be surfaced to clients via SSE `metrics` frames and/or response headers.

## 4) Determinism

- [OC-CORE-1030] Within a replica set, identical {prompt, parameters, seed, sampler_profile_version, engine_version, model_digest} MUST yield identical token streams.
- [OC-CORE-1031] Replica sets MUST pin engine_version and sampler_profile_version; mixed replicas MUST NOT share a set.
- [OC-CORE-1032] Determinism MUST NOT be assumed across engine/model updates.

## 5) Observability

- [OC-CORE-1040] Logs MUST include job_id, session_id, engine, pool_id, replica_id, model_id, quant, ctx, kv_warmth, queue_time_ms, decode_time_ms.
- [OC-CORE-1041] Metrics MUST include queue depth, reject/drop counts, first-token/decode latency, GPU/VRAM utilization, KV pressure, preload outcomes.

## 6) Traceability Map

- Requirements → Code: ../orchestrator-core/src/, Tests: ../orchestrator-core/tests/props_queue.rs
- Contracts: ../contracts/openapi/*.yaml (control/data via `orchestratord`)

## 7) Proof Hooks

- Property tests (to be created/enabled): ../orchestrator-core/tests/props_queue.rs
- Metrics contract: ../ci/metrics.lint.json, ../test-harness/metrics-contract/tests/metrics_lint.rs
