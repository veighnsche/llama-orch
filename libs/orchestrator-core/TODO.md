# orchestrator-core — TODO

Focus: queue invariants, placement hooks, determinism and scheduling primitives. This crate should remain engine-agnostic and expose clean traits for `orchestratord` to drive admission/dispatch.

Spec references
- `.specs/00_llama-orch.md` §2.3 Placement, §2.5 Data plane, §2.8 Observability, §2.10 Resilience & Recovery
- `.specs/71-metrics-contract.md`

High (must)
- [ ] **Placement hook (VRAM-aware, deterministic)**
  - Define `PlacementInput { pools: Vec<PoolSnapshot>, job: JobSpec } -> PlacementDecision`.
  - Deterministic tie-breakers: `least_loaded -> highest_vram_free -> lexicographic pool_id`.
  - Accept `slots_free`, `vram_free_bytes`, `engine` filters.
- [ ] **Compatibility predicate (model-aware)**
  - Feasibility check before scoring: pool must satisfy engine+model requirements.
  - Inputs from catalog/adapters: `min_vram_bytes`, `quantization`, `requires_compute_cap >= X`, `requires_extensions`.
  - Reject with `NoCapacity{ reason: Incompatible(Model|Engine|DeviceMask) }` if predicate fails.
- [ ] **Performance-aware scoring (GPU-aware)**
  - Consume `pools[*].perf.tokens_per_s` (per engine/model size/quant) and estimate `predicted_decode_ms` = `tokens_out / tokens_per_s * 1000`.
  - Include `first_token_ms` baseline when available.
  - Score by `predicted_end_ms` (admission latency + first-token + decode), prefer smaller.
- [ ] **Backpressure policy**
  - Implement `admission_policy(queue_state, capacity) -> Accept|Reject{policy_label, retry_after_ms}`.
  - Map to queue behavior (Reject/Drop-LRU) and produce headers via `orchestratord`.
- [ ] **Queue API**
  - Wrap `InMemoryQueue` into an admission façade that records metrics and supports cancel.
  - Enforce FIFO within class; property tests already present.
- [ ] **Metrics hooks**
  - Counters: enqueued/started/canceled/rejected.
  - Gauges: queue_depth.
  - Histograms: optional `admission_latency_ms` (TBD).

Mid (should)
- [ ] **Budget awareness**
  - Accept optional per-session budgets and surface advisory plan fields (predictive `predicted_start_ms`).
- [ ] **SLO-aware scoring**
  - Pluggable policy scoring for future tuning (latency targets, fairness weights).
- [ ] **Test scaffolding**
  - Property tests for placement determinism and backpressure stability.
  - Compatibility matrix tests (model vs device capabilities, quantization, ctx length to KV memory sizing).

Low (nice-to-have)
- [ ] **Scheduling traces**
  - Return a small trace log for decisions to aid observability.

Data types (sketch)
```rust
pub struct PoolSnapshot {
    pub id: String,
    pub engine: String,
    pub slots_total: i32,
    pub slots_free: i32,
    pub vram_total_bytes: i64,
    pub vram_free_bytes: i64,
    pub compute_capability: Option<String>, // e.g., "8.6"
    pub perf_tokens_per_s: Option<f64>,     // baseline for typical model size/quant
    pub supports: Option<PoolSupports>,     // adapter-published capabilities
}

pub struct JobSpec {
    pub priority: Priority,
    pub expected_tokens: Option<i32>,
    pub engine: String,
    pub model_id: String,
    pub required_ctx: i32,
    pub est_kv_bytes: Option<i64>, // derived from ctx/model dims
    pub requirements: Option<ModelRequirements>,
}

pub enum PlacementDecision {
    Assigned { pool_id: String },
    NoCapacity,
}

pub struct PoolSupports {
    pub quantizations: Vec<String>, // e.g., ["Q4_K_M", "BF16"]
    pub min_compute_cap: Option<String>,
}

pub struct ModelRequirements {
    pub min_vram_bytes: Option<i64>,
    pub quantization: Option<String>,
    pub min_compute_cap: Option<String>,
}
```

Proof tasks
- [ ] Unit tests: deterministic tie-breakers.
- [ ] Property tests: never assign to negative capacity.
- [ ] Integration with `orchestratord`: mock pool snapshots and assert decisions.
