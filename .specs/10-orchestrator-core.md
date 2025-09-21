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
- [OC-CORE-1012] Placement MUST use least‑loaded selection with VRAM awareness across replicas of the same replica set: prefer the replica with the most free VRAM, then fewest active slots; tie‑break deterministically (e.g., by `replica_id`).
- [OC-CORE-1013] Session affinity SHOULD keep a session on its last good replica when possible.

### 2A) Pin Override (optional feature)

- [OC-CORE-1018] When a `TaskRequest` includes a valid pin override (e.g., `pool_id`), and policy allows pinning, the scheduler MUST route the request to the specified pool and skip normal candidate selection across other pools.
- [OC-CORE-1019] If the specified pool is not Ready or the override is invalid (unknown pool, policy disabled), admission MUST return a deterministic typed error (e.g., `INVALID_PARAMS` or `POOL_UNREADY`), and MUST NOT silently fall back to automatic placement.

### 2A) Data Types — Canonical (authoritative)

`ModelRequirements` (canonical; referenced from wiring specs):

```
ModelRequirements {
  model_id: string,            // catalog id or ref-derived id
  model_digest: Option<string>,// content digest when available
  ctx_max: int,                // maximum context length supported by the model artifacts
  quant: Option<string>,       // normalized quantization tag (e.g., Q4_K_M), if applicable
  streaming: bool,             // supports token streaming
  extensions: Vec<string>,     // e.g., "speculative_decode", "mmproj"
}
```

- [OC-CORE-1014] `ModelRequirements` MUST be derivable from catalog metadata plus adapter/engine capability metadata. Missing fields (e.g., `quant`) MAY remain `None` if not observable; callers MUST NOT guess.
- [OC-CORE-1015] `ctx_max` MUST be the effective user-visible limit considering tokenizer/template overhead.

`PlacementInput` (scheduler view of a replica):

```
PlacementInput {
  replica_id: string,
  pool_id: string,
  engine: string,
  engine_version: string,
  free_vram_mb: int,
  active_slots: int,
  ctx_max_supported: int,
  features: Vec<string>, // extensions/features enabled on this replica
}
```

- [OC-CORE-1016] Feasibility MUST require `ctx_max_supported >= ModelRequirements.ctx_max` and feature subset satisfaction (when `extensions` are required). Engine/model mismatches MUST be rejected pre-dispatch.
- [OC-CORE-1017] Deterministic tie-break mapping: selection ordering MUST be defined as a tuple sort `(free_vram_mb desc, active_slots asc, replica_id asc)`.

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

## Refinement Opportunities

- Incorporate predicted decode time and KV pressure into scheduling signals while preserving determinism.
- Expose scheduler decision reasons in trace logs to aid performance tuning.
- Explore per-pool weighting or simple admission throttles tied to GPU class (e.g., 3090 vs 3060) while keeping deterministic tie‑breakers.
