# pool-managerd — TODO

Focus: pool lifecycle, health/readiness, preload (models+engines), device masks and heterogeneous GPU awareness, and supervision. Must integrate with `model-provisioner` and `engine-provisioner`.

Spec references
- `.specs/00_llama-orch.md` §2.6 Catalog, §2.9 Capabilities, §2.10 Resilience & Recovery, §2.12 Engine Provisioning
- `.specs/71-metrics-contract.md`

High (must)
- [ ] **Preload path (Ready gating)**
  - Call `model-provisioner::ensure_present` for the configured `model_ref`.
  - Prepare engine via `engine-provisioner::prepare()` (new API) and only flip `ready=true` after success.
- [ ] **Health reporting**
  - Track `live`, `ready`, `last_error`, `last_heartbeat_ms`, `version` (`engine_version`).
  - Expose device summaries: `vram_total_bytes`, `vram_free_bytes`, `slots_total/free`.
- [ ] **Device masks & hetero GPUs**
  - Represent device sets and simple `tensor_split` where applicable; publish in health and registry.
- [ ] **Registry integration**
  - Keep `registry::Registry` in sync with health/version/leases; add fields for capacity/VRAM.

Mid (should)
- [ ] **Supervisor**
  - Own engine process lifecycle (spawn/monitor/restart/backoff) using `PreparedEngine` metadata.
  - Respect draining state; stop accepting new leases when draining.
- [ ] **Backoff & drain**
  - Tunable backoff policy for restarts; graceful drain on reload.
- [ ] **Metrics**
  - Emit pool-level metrics for readiness transitions, restarts, and capacity changes.

Low (nice-to-have)
- [ ] **KV warm-up hooks**
  - Trigger adapter-specific warm-ups (prime tokenizers, preallocate KV) to reduce first-token latency.

Data types (sketch)
```rust
pub struct PoolConfig {
    pub id: String,
    pub engine: String,
    pub model_ref: String,
    pub device_mask: Vec<i32>,
}

pub struct HealthStatus {
    pub live: bool,
    pub ready: bool,
}

pub struct DeviceSnapshot {
    pub id: i32,
    pub vram_total_bytes: i64,
    pub vram_free_bytes: i64,
}

pub struct PoolSummary {
    pub engine_version: Option<String>,
    pub devices: Vec<DeviceSnapshot>,
    pub slots_total: i32,
    pub slots_free: i32,
}
```

Proof tasks
- [ ] Unit tests: registry updates, lease accounting, health transitions.
- [ ] Integration: preload success gates `ready=true`; drain/reload cycles.
