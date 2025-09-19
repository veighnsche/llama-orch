# pool-managerd — Production Readiness Checklist

This checklist covers everything required to ship `pool-managerd` as a production‑ready process manager for engine pools. It owns device discovery and health, preload and readiness gating, supervised engine lifecycle, and live capacity reporting for placement.

## Scope & References

- Specs: `.specs/00_llama-orch.md` (§2.6 Catalog, §2.9 Capabilities, §2.10 Resilience & Recovery, §2.12 Engine Provisioning)
- Metrics: `.specs/71-metrics-contract.md`, `.specs/metrics/otel-prom.md`
- CLI expectations: `cli/llama-orch-cli/feature-requirements.md` (FR‑CP: control, FR‑DV: discovery)

## Responsibilities (at a glance)

- [ ] Discover GPUs/devices; compute capability, VRAM totals/free, device masks, optional MIG awareness.
- [ ] Preload model + prepare engine via provisioners; gate `ready=true` only after success.
- [ ] Supervise engine processes/containers with backoff; publish health/metrics; handle drain/reload.
- [ ] Publish pool snapshots/registry entries to be consumed by `orchestrator-core` placement.

## Device Discovery & Snapshots

- [ ] GPU inventory
  - [ ] Enumerate devices (NVIDIA).
  - [ ] Collect: `device_id`, `compute_capability`, `vram_total_bytes`, `vram_free_bytes`.
  - [ ] Optional: MIG partitions; present a stable mask per pool.
- [ ] Pool snapshot schema
  - [ ] Summarize devices into `PoolSummary` with `slots_total/free`, `engine_version`, device list.
  - [ ] Report steady‑state perf hints: `perf_tokens_per_s`, `first_token_ms` (see Perf section).
  - [ ] Surface supported quantizations/features from adapters (if available).

## Preload & Readiness Gating

- [ ] Model staging (via model‑provisioner)
  - [ ] Call `model-provisioner::ensure_present(model_ref)`; no package installs here.
  - [ ] Verify digest when provided; log advisory otherwise.
- [ ] Engine preparation (via engine‑provisioner)
  - [ ] Add `prepare(pool) -> PreparedEngine { bin/entry, args, env, ports, workdir }` API; no process spawn here yet.
  - [ ] Honor device masks and runtime flags required by the engine.
- [ ] Readiness transitions
  - [ ] `live=true` when the manager is up; `ready=true` only after model staged + engine prepared + healthy endpoint.
  - [ ] Write `engine_version` and `model_id` to the registry on success.

## Supervision & Lifecycle

- [ ] Process/container manager
  - [ ] Start/stop engine using `PreparedEngine` (binary) or container (prefer `podman`, fallback `docker`).
  - [ ] Health checks: poll engine HTTP/gRPC endpoints; configurable intervals and timeouts.
  - [ ] Backoff policy: exponential with jitter; max backoff cap; reset on stable run.
- [ ] Draining & reload
  - [ ] On drain: stop accepting new leases; wait for in‑flight or force stop on deadline.
  - [ ] On reload: drain, stage new model, restart engine, run health checks, flip `ready=true`.
  - [ ] Publish `last_error` and `version` changes to the registry.

## Registry & Contracts

- [ ] Update `pool-managerd::registry::Registry`
  - [ ] Track: `live`, `ready`, `last_error`, `last_heartbeat_ms`, `engine_version`, `active_leases`.
  - [ ] Add: `slots_total`, `slots_free`, `vram_total_bytes`, `vram_free_bytes`, `compute_capability`, `device_mask`.
  - [ ] Optional: `perf_tokens_per_s`, `first_token_ms`, `supports.quantizations`.
- [ ] Health readout consumed by `orchestratord`
  - [ ] Ensure `GET /v1/pools/{id}/health` can reflect the above.

## Capacity & Leases

- [ ] Lease accounting
  - [ ] Increment/decrement atomically; never negative (tests exist).
  - [ ] Export `slots_total/free` based on engine concurrency; configurable.
- [ ] Draining awareness
  - [ ] Set a `draining` flag; refuse new work; allow in‑flight to drain until deadline.

## Performance Baselines (Hints for Placement)

- [ ] Benchmark and publish
  - [ ] `tokens_per_s` steady‑state for common model sizes/quant (e.g., 7B Q4, 7B BF16) per device mask.
  - [ ] `first_token_ms` baseline.
  - [ ] Record per `engine_version`/`model_id`/`device_mask` when possible; keep cardinality bounded.
- [ ] Rolling update
  - [ ] Optionally update perf hints after warm‑up; smooth using EWMA.

## Metrics & Logs

- [ ] Prometheus metrics (names per `.specs/metrics/otel-prom.md`)
  - [ ] Pool readiness transitions, restarts, backoff counts, health probe failures.
  - [ ] Capacity gauges: `slots_*`, `vram_*` (if exported), `active_leases`.
  - [ ] Backpressure interactions (admission is upstream; expose pool local signals that affect it).
- [ ] Logs (structured JSON)
  - [ ] Include: `pool_id`, `engine`, `engine_version`, `device_mask`, `model_id`, `restart_count`, `backoff_ms`, `last_error`.
  - [ ] Correlate with `X-Correlation-Id` where applicable.

## Security & Policy

- [ ] Privilege model
  - [ ] Run engines as non‑root user; drop capabilities; isolate workdir.
  - [ ] Container runtime hardening (rootless podman; constrained seccomp/apparmor if available).
- [ ] Policy gates
  - [ ] No package installs here; rely on provisioners and configuration.
  - [ ] Restrict network egress where possible; only engine endpoints.

## Configuration & Feature Toggles

- [ ] Device masks
  - [ ] Configurable masks per pool; validate mask against discovered devices.
- [ ] Concurrency/slots
  - [ ] Config for `slots_total`; derive from engine when possible.
- [ ] Health checks
  - [ ] Intervals/timeouts; fail‑open/fail‑fast modes for dev vs prod.
- [ ] Backoff policy
  - [ ] Initial, factor, max; jitter enabled.

## Failure Handling & Resilience

- [ ] Robust to partial failures
  - [ ] Engine crash restarts with backoff; publish `last_error`.
  - [ ] Health probe failures do not crash manager; only flip `ready` and escalate after threshold.
  - [ ] Detect `CUDA`/driver errors; switch `ready=false` and suggest remediation.

## Testing Strategy

- [ ] Unit tests
  - [ ] Registry updates (health, version, heartbeat) and leases (never negative).
  - [ ] Device mask validation.
- [ ] Integration tests
  - [ ] Preload gates readiness (model + prepare + healthy check → `ready=true`).
  - [ ] Drain/reload cycles with deadlines.
  - [ ] Supervision/backoff: simulated crash loops.
- [ ] BDD/provider wiring
  - [ ] `orchestratord` endpoints (`/pools/{id}/health`, drain, reload) reflect pool-managerd state.

## Contracts & CDC Alignment

- [ ] Align with control OpenAPI
  - [ ] `GET /v1/pools/{id}/health` returns fields needed by CLI and placement.
  - [ ] Drain/reload semantics: deadlines honored; errors surfaced.
- [ ] Capability reporting
  - [ ] Ensure enough data is available for `GET /v1/capabilities` to be truthful (per engine/version).

## Performance & Memory

- [ ] Low overhead supervision
  - [ ] Avoid busy‑polling; async IO.
  - [ ] Bound memory growth of logs and transient buffers.

## Release Gating Criteria

- [ ] All unit/integration tests green (pool-managerd crate).
- [ ] Clippy/fmt clean: `cargo fmt --all -- --check`, `cargo clippy --all-targets --all-features -- -D warnings`.
- [ ] Metrics lint green for any emitted metrics; linkcheck passes.
- [ ] Verified reload/drain with real engine in home profile (or CI equivalent).
- [ ] Documentation complete and current (`README.md`, `CHECKLIST.md`).

---

## Appendix: Data Shapes (sketch)

```rust
pub struct DeviceSnapshot {
    pub id: i32,
    pub compute_capability: Option<String>,
    pub vram_total_bytes: i64,
    pub vram_free_bytes: i64,
}

pub struct PoolSummary {
    pub engine: String,
    pub engine_version: Option<String>,
    pub model_id: Option<String>,
    pub device_mask: Vec<i32>,
    pub devices: Vec<DeviceSnapshot>,
    pub slots_total: i32,
    pub slots_free: i32,
    pub perf_tokens_per_s: Option<f64>,
    pub first_token_ms: Option<f64>,
}
```
