# orchestrator-core — Component Specification (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## 0) Purpose & Scope

`orchestrator-core` contains engine-agnostic orchestration primitives: bounded priority queues with admission policies, placement/scoring hooks, and determinism/observability invariants. It has no HTTP or adapter I/O.

In scope:
- Queue invariants and admission behavior (Reject, Drop‑LRU) with priorities (Interactive, Batch).
- Placement inputs/outputs and tie‑breaker policy (compatibility predicate; performance‑aware scoring; deterministic fallback).
- Determinism and observability contracts for consumers to emit.

Out of scope:
- HTTP endpoints; adapter RPCs; process supervision.

## 1) Normative Requirements (RFC‑2119)

- [ORCH‑3400] The crate MUST expose queue types with the following semantics:
  - Priorities: `Interactive`, `Batch`.
  - Policies: `Reject`, `DropLru`.
  - FIFO MUST hold within a priority class.
  - With `Reject`, enqueues on full MUST fail with a typed error.
  - With `DropLru`, the oldest `Batch` item SHOULD be dropped first; if none, drop oldest across priorities.
- [ORCH‑3401] The queue API MUST include `enqueue(id, prio)`, `cancel(id)`, `len()`, `capacity()`, and a read‑only snapshot of each priority.
- [ORCH‑3402] `cancel(id)` MUST remove the first occurrence across priorities, returning whether removal occurred.
- [ORCH‑3403] The crate MUST NOT perform network I/O or spawn processes.
- [ORCH‑3404] Placement policy (planning): the crate MUST define data shapes for input and output and SHOULD provide a reference implementation:
  - `PlacementInput { pools: Vec<PoolSnapshot>, job: JobSpec }`.
  - Feasibility MUST be checked first via a compatibility predicate (engine/model/device), rejecting pools that fail min VRAM, compute capability, quantization, or required extensions.
  - Primary scoring SHOULD minimize `predicted_end_ms = admission_latency + first_token_ms + decode_ms(tokens_out, perf_tokens_per_s)`.
  - Tie‑breakers MUST apply only when primary scores are equal, in this order: KV/session affinity; least loaded; highest residual VRAM (`vram_free − est_kv_bytes`); higher `perf_tokens_per_s`; stable lexicographic `pool_id`.
- [ORCH‑3405] Determinism: consumers MUST be able to emit logs with fields `{seed, sampler_profile_version, engine_version, model_digest}` and MUST NOT change replica mid‑stream unless requested.
- [ORCH‑3406] Observability: consumers SHOULD emit counters for `tasks_enqueued_total`, `tasks_started_total`, `tasks_canceled_total`, `tasks_rejected_total`, a gauge for `queue_depth`, and optional histograms for `admission_latency_ms`.

## 2) Data Types & Semantics (planning)

```rust
pub enum Priority { Interactive, Batch }
pub enum Policy { Reject, DropLru }

pub struct PoolSnapshot {
    pub id: String,
    pub engine: String,
    pub slots_total: i32,
    pub slots_free: i32,
    pub vram_total_bytes: i64,
    pub vram_free_bytes: i64,
    pub compute_capability: Option<String>,
    pub perf_tokens_per_s: Option<f64>,
    pub first_token_ms: Option<f64>,
}

pub struct JobSpec {
    pub priority: Priority,
    pub expected_tokens: Option<i32>,
    pub engine: String,
    pub model_id: String,
    pub required_ctx: i32,
    pub est_kv_bytes: Option<i64>,
}

pub enum PlacementDecision { Assigned { pool_id: String }, NoCapacity }
```

## 3) Interfaces & Contracts

- The queue API is stable within the repo pre‑1.0; name/semantics MUST not change without updating all call sites in the same PR.
- Placement input/output shapes MAY evolve; changes MUST be traced in specs and tests.

## 4) Observability

- Required log fields for downstream emission: `job_id`, `session_id`, `engine`, `engine_version`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out`, `decode_time_ms`.

## 5) Security

- No secrets; no network; pure logic only.

## 6) Testing & Proof Bundle

- Unit/property tests MUST cover: bounded queues, policies, FIFO within class, cancel semantics.
- Placement tests SHOULD cover: compatibility predicate edges; tie‑breakers; deterministic selection.
- Determinism tests SHOULD exist in the determinism suite (cross‑crate) and MUST NOT duplicate sampler/engine internals here.

## 7) Open Questions

- Should placement live here or as a separate policy crate to reduce churn?
- How much of the predicted_end_ms should be learned online vs static hints from pools?

## 8) Refinement Opportunities

- Budget‑aware scoring and SLO fairness weights.
- Session KV reuse affinity for latency wins.
- Advisory scheduling traces for debugging.
