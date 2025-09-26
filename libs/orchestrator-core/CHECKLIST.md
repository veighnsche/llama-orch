# orchestrator-core — Production Readiness Checklist

This checklist captures all work required to ship `orchestrator-core` as a production‑ready component. It focuses on engine‑agnostic scheduling primitives, model/device feasibility, deterministic placement, and clear contracts with `orchestratord` and `pool-managerd`.

## Scope & References

- Specs: `.specs/00_llama-orch.md` (§2.3 Placement, §2.5 Data plane, §2.8 Observability, §2.10 Resilience)
- Metrics: `.specs/71-metrics-contract.md`, `.specs/metrics/otel-prom.md`
- CLI expectations: `cli/llama-orch-cli/feature-requirements.md` (FR‑DP, FR‑MA, FR‑DV)

## Compatibility & Placement

- [ ] Compatibility predicate (model‑aware feasibility)
  - [ ] Reject pools that don’t satisfy model+engine requirements.
  - [ ] Inputs: `min_vram_bytes`, `quantization`, `min_compute_capability`, `required_extensions`.
  - [ ] Account for `required_ctx` → `est_kv_bytes` (ctx length and model dims).
  - [ ] Return explicit reasons: `Incompatible(Model|Engine|DeviceMask)`.
- [ ] VRAM‑aware, deterministic selection
  - [ ] Primary scoring compares predicted_end_ms; tie‑breakers apply only when scores are equal.
  - [ ] Recommended tie‑breakers (in order):
    1) Session/KV affinity (prefer pools with warm KV for the session)
    2) Least loaded (fewer active leases / higher `slots_free`)
    3) Highest residual VRAM headroom: `(vram_free_bytes − est_kv_bytes_for_job)`
    4) Higher steady‑state throughput (`perf_tokens_per_s`)
    5) Stable lexicographic fallback (`pool_id` or stable UUID)
  - [ ] Consume `slots_free`, `vram_free_bytes`, `engine`, `perf_tokens_per_s`, `kv_warmth` signals from pool snapshots.
- [ ] Performance‑aware scoring (GPU‑aware)
  - [ ] Use `perf_tokens_per_s` and `first_token_ms` to minimize `predicted_end_ms`.
  - [ ] Include admission latency estimate in scoring.
- [ ] Device masks & hetero GPU constraints
  - [ ] Ensure device subsets are feasible for the model (quant/size/split requirements).
  - [ ] Propagate `NoCapacity` when all feasible pools are exhausted.

## Admission & Backpressure

- [ ] Backpressure policy
  - [ ] `admission_policy(queue_state, capacity) -> Accept | Reject { policy_label, retry_after_ms }`.
  - [ ] Map policy to queue behavior: `Reject` vs `Drop‑LRU`.
- [ ] Priority & fairness
  - [ ] Maintain FIFO within a priority class.
  - [ ] (Optional) weights for fairness/SLOs, behind config gate.

## Queue Invariants & API

- [ ] Queue facade around `InMemoryQueue`
  - [ ] Enqueue with priority class; capacity check before insert.
  - [ ] Cancel must be race‑free; removing pending items deterministically.
- [ ] Isolation semantics
  - [ ] Separate Interactive/Batch queues; `Drop‑LRU` drops oldest batch first.

## Budgets, Sessions & Forecasting

- [ ] Budget awareness (optional for v1)
  - [ ] Accept token/time/cost budgets and return advisory `predicted_start_ms`.
- [ ] Session reuse & KV hints
  - [ ] Honor `KVHint::Reuse`; surface `kv_warmth` for `metrics` SSE frames via `orchestratord`.

## Determinism Guarantees

- [ ] Determinism by default
  - [ ] Honor `{seed, sampler_profile_version, engine_version, model_digest}`.
  - [ ] Ensure placement does not violate per‑replica determinism during a stream.
- [ ] Tests
  - [ ] Byte‑exact stream tests for fixed seed on the same replica (with harness support).

## Observability & Metrics

- [ ] Metrics hooks
  - [ ] Counters: `tasks_enqueued_total`, `tasks_started_total`, `tasks_canceled_total`, `tasks_rejected_total`.
  - [ ] Gauges: `queue_depth`.
  - [ ] Histograms: optional `admission_latency_ms`; feed `latency_first_token_ms`, `latency_decode_ms` via `orchestratord`.
- [ ] Logs and traces
  - [ ] Emit traceable fields: `job_id`, `session_id`, `engine`, `engine_version`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out`, `decode_time_ms`.
  - [ ] (Optional) scheduling trace for placement decisions.
- [ ] SSE metrics support (via `orchestratord`)
  - [ ] Provide values that can be surfaced in `metrics` frames: `queue_depth`, advisory `on_time_probability`, `kv_warmth`.

## Data Structures & Interfaces

- [ ] Placement IO types
  - [ ] `PlacementInput { pools: Vec<PoolSnapshot>, job: JobSpec }`.
  - [ ] `PlacementDecision { Assigned { pool_id } | NoCapacity { reason } }`.
- [ ] `PoolSnapshot` fields
  - [ ] `id`, `engine`, `slots_total`, `slots_free`.
  - [ ] `vram_total_bytes`, `vram_free_bytes`, `compute_capability`.
  - [ ] `perf_tokens_per_s`, `first_token_ms`, `supports.quantizations`.
- [ ] `JobSpec` fields
  - [ ] `priority`, `expected_tokens`, `engine`, `model_id`, `required_ctx`.
  - [ ] `est_kv_bytes`, `requirements.{min_vram_bytes, min_compute_cap, quantization}`.

## Failure Handling & Resilience

- [ ] Snapshot robustness
  - [ ] Tolerate empty/stale pool snapshots; prefer `NoCapacity` over panics.
- [ ] Draining awareness
  - [ ] Treat draining pools as ineligible for new placements.
- [ ] Retry strategies
  - [ ] Minimal retries around snapshot fetch if applicable (caller‑driven in `orchestratord`).

## Contracts & CDC Alignment

- [ ] Spec compliance
  - [ ] `.specs/00_llama-orch.md` §2.3 placement rules; §2.8 observability.
- [ ] Stable API to `orchestratord`
  - [ ] Keep placement API stable; version when changing data shapes.

## Testing Strategy

- [ ] Unit tests
  - [ ] Deterministic tie‑breakers; compatibility matrix; backpressure policies.
- [ ] Property tests
  - [ ] Queue never goes negative; FIFO preserved within class.
- [ ] Integration tests
  - [ ] Mock pool snapshots → expected placement; expected metrics/logging hooks invoked.

## Performance & Memory

- [ ] Efficiency
  - [ ] O(P) selection/scoring over pools; avoid quadratic behavior.
  - [ ] No excessive heap churn in hot path; reuse buffers where reasonable.
- [ ] KV estimation helpers
  - [ ] Optional helpers for `est_kv_bytes` from model dims and ctx length (or consume from adapters).

## Configuration & Feature Toggles

- [ ] Tuning knobs
  - [ ] Backpressure thresholds, priority weights, scoring weights.
- [ ] Feature gates
  - [ ] Ability to disable performance‑aware scoring for minimal configs.

## Documentation

- [ ] Code docs
  - [ ] Document placement policy, compatibility predicate, scoring function.
- [ ] Developer README
  - [ ] Add High/Mid/Low behavior summary consistent with repo style.
- [ ] Traceability
  - [ ] Cross‑reference ORCH‑3xxx requirement IDs in modules and tests.

## Release Gating Criteria

- [ ] All unit/property/integration tests green (`cargo test -p orchestrator-core`).
- [ ] Clippy/fmt clean: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`.
- [ ] Metrics lint green (names/labels as per `.specs/metrics/otel-prom.md`).
- [ ] Verified through provider/BDD paths when integrated with `orchestratord`.
- [ ] Documentation complete and up to date.

---

## Appendix: Data Type Sketches (for discussion)

```rust
pub struct PoolSnapshot {
    pub id: String,
    pub engine: String,
    pub slots_total: i32,
    pub slots_free: i32,
    pub vram_total_bytes: i64,
    pub vram_free_bytes: i64,
    pub compute_capability: Option<String>, // e.g., "8.6"
    pub perf_tokens_per_s: Option<f64>,     // baseline for model size/quant
    pub first_token_ms: Option<f64>,        // baseline
    pub supports: Option<PoolSupports>,     // adapter-published caps
}

pub struct PoolSupports {
    pub quantizations: Vec<String>,         // e.g., ["Q4_K_M", "BF16"]
    pub min_compute_cap: Option<String>,
}

pub struct JobSpec {
    pub priority: Priority,
    pub expected_tokens: Option<i32>,
    pub engine: String,
    pub model_id: String,
    pub required_ctx: i32,
    pub est_kv_bytes: Option<i64>,          // derived or provided
    pub requirements: Option<ModelRequirements>,
}

pub struct ModelRequirements {
    pub min_vram_bytes: Option<i64>,
    pub quantization: Option<String>,
    pub min_compute_cap: Option<String>,
    pub required_extensions: Vec<String>,
}

pub enum PlacementDecision {
    Assigned { pool_id: String },
    NoCapacity,
}
```
