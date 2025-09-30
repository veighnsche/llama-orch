# pool-managerd Stub Analysis

**Date:** 2025-09-30  
**Status:** Investigation complete; TODOs added to all stub files

## Summary

The `pool-managerd` crate contains a mix of **implemented**, **partially implemented**, and **stub** modules. This analysis traces each file to its spec requirements, identifies what's using it, and documents next steps.

---

## File Status Matrix

| File | Status | Spec IDs | Used By | Action |
|------|--------|----------|---------|--------|
| `src/health.rs` | ✅ **IMPLEMENTED** | ORCH-3002 | registry.rs, orchestratord, BDD tests | Keep as-is; optional expansion |
| `src/preflight.rs` | ✅ **IMPLEMENTED** | ORCH-1102, ORCH-3202 | engine-provisioner | Expand for more tools (git, cmake, nvcc) |
| `src/registry.rs` | 🟡 **PARTIAL** | ORCH-3002, 3010, 3027, 3028, 3038 | orchestratord, BDD tests | Add VRAM fields, wire supervision/drain |
| `src/registry/entry.rs` | ✅ **IMPLEMENTED** | (internal) | registry.rs | Add VRAM/compute_capability fields |
| `src/registry/types.rs` | ✅ **IMPLEMENTED** | (internal) | registry.rs | Add VRAM fields if needed |
| `src/registry/snapshot.rs` | ✅ **IMPLEMENTED** | (internal) | registry.rs | Populate VRAM/perf_hints from discovery |
| `src/backoff.rs` | ❌ **STUB** | ORCH-3038, 3040 | (future supervision module) | Implement exponential backoff + jitter |
| `src/devicemasks.rs` | ❌ **STUB** | ORCH-3010, 3011, 3052 | (future placement module) | Implement mask parsing/validation |
| `src/drain.rs` | ❌ **STUB** | ORCH-3031, 3038 | (future orchestratord control API) | Implement drain/reload lifecycle |
| `src/hetero_split.rs` | ❌ **STUB** | ORCH-3052 | (future preload module) | Implement tensor split planner |
| `src/leases.rs` | ⚠️ **REDUNDANT?** | ORCH-3004, 3010 | (registry already has lease counters) | **Decision needed: delete or expand** |
| `src/preload.rs` | ❌ **STUB** | ORCH-3002, 3003 | (future pool initialization) | Implement preload orchestration |
| `src/main.rs` | ❌ **STUB** | ORCH-3004..3045 | (standalone daemon or embedded?) | **Decision needed: implement or delete** |

---

## Detailed Findings

### ✅ Implemented & Active

#### `src/health.rs`

- **Status:** Fully implemented, actively used
- **Spec:** ORCH-3002 (live/ready states)
- **Used by:** `registry.rs`, `orchestratord/api/control.rs`, `test-harness/bdd`
- **Tests:** Unit tests (OC-POOL-3001), BDD tests
- **TODO:** Optional expansion for `last_check_at`, `consecutive_failures`

#### `src/preflight.rs`

- **Status:** Fully implemented with tests
- **Spec:** ORCH-1102 (GPU-only), ORCH-3202 (preflight checks)
- **Used by:** `engine-provisioner` (before build/start)
- **Tests:** Unit tests (lines 68-98)
- **TODO:** Expand for git, cmake, make, nvcc, podman/docker, huggingface-cli, aws, oras (ORCH-3203)

#### `src/registry.rs` + submodules

- **Status:** Partially implemented (Owner E tasks complete)
- **Spec:** ORCH-3002, 3010, 3027, 3028, 3038
- **Completed:**
  - Health/version/heartbeat/last_error getters/setters
  - Lease counters (allocate/release, never negative)
  - `register_ready_from_handoff()` API
  - `set_engine_meta()` for engine metadata
  - Draining flag and lease refusal
  - Snapshots export for placement
- **Used by:** `orchestratord/src/state.rs:6`, `test-harness/bdd`
- **Tests:** Unit tests (OC-POOL-3001, 3007, 3101-3109)
- **TODO:**
  - Add VRAM fields (vram_total_bytes, vram_free_bytes, compute_capability)
  - Add perf hints (tokens_per_s, first_token_ms)
  - Wire to supervision module (backoff.rs)
  - Wire to drain/reload module (drain.rs)

---

### ❌ Stubs Awaiting Implementation

#### `src/backoff.rs`

- **Spec:** ORCH-3038 (restart with backoff), ORCH-3040 (circuit breakers)
- **Checklist:** "Backoff policy: exponential with jitter; max backoff cap; reset on stable run"
- **Usage:** Called by supervision module (not yet implemented) on engine crash/health failure
- **Expected API:**
  - `BackoffPolicy::new(initial_ms, max_ms, jitter_factor)`
  - `BackoffPolicy::next_delay(&mut self) -> Duration`
  - `BackoffPolicy::reset(&mut self)`
  - Optional: circuit breaker state machine (open/half-open/closed)
- **Tests:** BDD step "restart storms are bounded by circuit breaker" (pool_manager.rs:48)

#### `src/devicemasks.rs`

- **Spec:** ORCH-3010, 3011 (scheduler respects masks), ORCH-3052 (no spillover)
- **Checklist:** "Optional: MIG partitions; present a stable mask per pool"
- **Usage:** Called by placement/scheduler to ensure jobs only run on allowed GPUs
- **Expected API:**
  - `DeviceMask::parse(mask_str: &str) -> Result<Self>`
  - `DeviceMask::validate_against_discovered(devices: &[DeviceSnapshot])`
  - `DeviceMask::to_cuda_visible_devices(&self) -> String`
- **Tests:** BDD step "placement respects device masks; no cross-mask spillover occurs" (pool_manager.rs:56)
- **Integration:** Used by registry.rs (device_mask field) and future placement module

#### `src/drain.rs`

- **Spec:** ORCH-3031, 3038 (atomic drain/reload; reversible on failure)
- **Checklist:** "Draining & reload: stop accepting new leases; wait for in-flight or force stop on deadline"
- **Usage:** Called by orchestratord control API (POST /v1/pools/{id}/drain, /reload)
- **Expected API:**
  - `DrainRequest::new(pool_id, deadline_ms)`
  - `ReloadRequest::new(pool_id, new_model_ref, new_engine_version)`
  - `execute_drain(req, registry) -> Result<DrainOutcome>`
  - `execute_reload(req, registry, provisioner) -> Result<ReloadOutcome>`
- **Integration:** Calls `registry.set_draining(true)`, waits for `active_leases → 0`, stops engine
- **Reload flow:** drain → stage new model → restart engine → health check → flip ready=true or rollback
- **Tests:** Integration test for drain/reload cycles with deadlines (CHECKLIST.md)

#### `src/hetero_split.rs`

- **Spec:** ORCH-3052 (tensor splits opt-in; respect smallest GPU's VRAM)
- **Checklist:** "Optional: MIG partitions; present a stable mask per pool"
- **Usage:** Called during pool initialization when multiple GPUs with different VRAM are configured
- **Expected API:**
  - `SplitPlan::compute(devices: &[DeviceSnapshot], model_size_bytes: u64)`
  - `SplitPlan::ratios(&self) -> &[f32]` (per-GPU split ratios, sum to 1.0)
  - `SplitPlan::validate(&self)` (ensure smallest GPU can hold its share)
- **Integration:** Used by preload module to configure engine with --tensor-split flags (llama.cpp)
- **Tests:** BDD step "per-GPU resident KV is capped for smallest GPU" (pool_manager.rs:64)
- **Note:** Default assumes no split (single GPU); this is opt-in only

#### `src/preload.rs`

- **Spec:** ORCH-3002 (preload at startup, ready only after success), ORCH-3003 (fail fast on insufficient VRAM/RAM)
- **Checklist:** "Model staging (via model-provisioner) + Engine preparation (via engine-provisioner) → ready=true only after healthy endpoint"
- **Usage:** Called during pool initialization before marking ready=true
- **Expected API:**
  - `PreloadOutcome::execute(pool_id, model_ref, device_mask, registry) -> Result<Self>`
  - Calls `model-provisioner::ensure_present(model_ref)` to stage model
  - Calls `engine-provisioner::prepare(pool)` to get PreparedEngine
  - Starts engine process/container and waits for health check
  - On success: `registry.register_ready_from_handoff(pool_id, handoff)` → ready=true
  - On failure: `registry.set_last_error(pool_id, err)` → ready=false
- **Integration:** Wired by orchestratord bootstrap or pool-managerd daemon (not yet implemented)
- **Tests:** BDD steps "pool is Unready due to preload failure" (pool_manager.rs:5-12)
- **Tests:** Integration test "preload gates readiness" (CHECKLIST.md)

---

### ⚠️ Redundant or Decision Needed

#### `src/leases.rs` — **DELETE** (observability does not require per-lease tracking)

**Current Implementation:**

- Stub module with `#[allow(dead_code)]` markers
- Contains only type definitions: `LeaseId(String)` and `Lease` (both empty)
- No actual implementation or usage anywhere in codebase

**Registry Already Provides:**

- `allocate_lease(pool_id) -> i32` — increments active_leases counter (registry.rs:408-429)
- `release_lease(pool_id) -> i32` — decrements active_leases counter, never goes negative (registry.rs:431-450)
- `get_active_leases(pool_id) -> i32` — queries current count (registry.rs:452-454)
- Draining support: when `draining=true`, allocate_lease refuses new leases (registry.rs:425-427)
- Full test coverage: OC-POOL-3007 validates leases never go negative (registry.rs:483-494)

**Spec Requirements (ORCH-3004, 3010):**

- ORCH-3004: "Each pool MUST expose a bounded FIFO queue" → handled by orchestrator-core, not pool-managerd
- ORCH-3010: "Scheduler MUST only dispatch to Ready replicas" → registry.get_health() + allocate_lease() provide this
- No spec requirement for per-lease metadata tracking

**Usage Analysis:**

- **orchestratord** uses registry lease counters directly (no imports of leases.rs)
- **test-harness/bdd** uses registry lease counters directly (no imports of leases.rs)
- **No other crate** imports or references `pool_managerd::leases`

**Human Narration Observability Analysis (ORCH-3300, 3312):**

From `.specs/proposals/batch_1/2025-09-19-human-narration-logging.md` and `.specs/00_llama-orch.md §2.8.1`:

- **ORCH-3300:** "Significant events/spans SHOULD attach a short narration string alongside structured fields"
- **ORCH-3312:** "The `human` narration text MUST be natural-language and human-friendly. It MUST NOT primarily consist of opaque identifiers (UUIDs, hashes). Keep raw identifiers in structured fields (e.g., `job_id`, `session_id`, `pool_id`)"
- **Emission points (normative):** admission decision, placement decision, stream start, stream end, cancel path

**Question: Is tracking which job_id holds which lease interesting for observability?**

**Answer: NO — lease allocation is already observable via narration at admission/placement points**

**Why per-lease tracking is NOT needed:**

1. ✅ **Admission narration** (ORCH-3300 emission point) already logs: `"Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'"` with structured fields `job_id`, `pool_id`, `queue_position`
2. ✅ **Placement narration** (ORCH-3300 emission point) will log: `"Dispatched job {job_id} to pool 'default' replica 'r0' (slot 1/4 allocated)"` with structured fields `job_id`, `pool_id`, `replica_id`, `slot_allocated`
3. ✅ **Stream end narration** (ORCH-3300 emission point) logs: `"Completed job {job_id} on pool 'default': 456 tokens in 9876 ms"` with structured fields `job_id`, `tokens_out`, `decode_time_ms`
4. ✅ **Cancel narration** (ORCH-3300 emission point) logs: `"Canceled job {job_id} on pool 'default' (released slot)"` with structured field `job_id`

**What narration provides without per-lease tracking:**

- Human-readable story: admission → placement → stream → completion/cancel
- Structured fields for correlation: `job_id`, `pool_id`, `replica_id`, `session_id`
- Metrics: `active_leases` gauge, `tasks_started_total`, `tasks_completed_total`, `tokens_out_total`
- Proof bundles: SSE transcripts with correlation IDs (ORCH-3300)

**When Would Expansion Be Needed?**
Only if we need (NONE in current spec):

1. **Per-lease audit trail** — track which job_id holds which lease, when allocated, for debugging/observability → **NOT NEEDED:** narration at emission points provides this
2. **Capacity estimation** — track tokens_in/out, duration per lease to predict future capacity needs → **NOT NEEDED:** metrics already track `tokens_out_total`, `latency_decode_ms` histograms
3. **Lease expiry/timeouts** — automatic lease release after deadline (not in current spec)
4. **Multi-slot leases** — single job reserves multiple slots (not in current spec)

**Current Architecture Decision (from .plan/30_pool_managerd.md):**

- Lines 49-74 propose future modularization: `pool-domain` (types) → `pool-services` (registry/leases) → `pool-api` → `pool-managerd` (bin)
- But explicitly states: "Rollout: Stage 7.1: enforce in-crate modules... Extract `pool-domain` first **if compile times grow**"
- Current compile times are fine; no extraction needed yet

**Recommendation: DELETE `src/leases.rs`**

**Rationale:**

1. ✅ Registry counters fulfill all current spec requirements
2. ✅ No codebase usage or imports
3. ✅ No failing tests that need this module
4. ✅ YAGNI principle: don't implement until spec/tests require it
5. ✅ If per-lease metadata is needed later, can be added to PoolEntry or a new module with clear requirements

**Alternative (if keeping):**
If you want to preserve the stub for future expansion, add this to the module:

```rust
// TODO: This module is currently unused. Registry.allocate_lease/release_lease provide
// sufficient lease tracking for current specs. Expand this module only when:
// - Spec requires per-lease audit trail (job_id, allocated_at, tokens_in/out)
// - Spec requires lease expiry/timeouts
// - Spec requires multi-slot leases
// Until then, consider deleting to reduce maintenance burden.
```

---

#### `src/main.rs` — **KEEP STUB** (CLOUD_PROFILE.md suggests standalone daemon for multi-tenant)

**Current Architecture (Home Profile):**

- **orchestratord embeds pool-managerd as a library** (bin/orchestratord/src/state.rs:6):

  ```rust
  use pool_managerd::registry::Registry as PoolRegistry;
  pub struct AppState {
      pub pool_manager: Arc<Mutex<PoolRegistry>>,
      // ...
  }
  ```

- orchestratord owns the lifecycle: creates Registry, updates it from handoff files, queries it for health/placement
- No separate pool-managerd process exists or is referenced in deployment docs
- **On-demand engine startup:** Engines/models are started **only when needed**, not preloaded (clarified by user)

**CLOUD_PROFILE.md Analysis (.specs/proposals/CLOUD_PROFILE.md):**

**Key Requirements for Cloud Deployment:**

- **CLOUD-200:** Public HTTPS endpoint behind gateway/WAF (multi-tenant)
- **CLOUD-210:** Private endpoint per customer (dedicated/single-tenant)
- **CLOUD-300:** Cloud VMs or k8s nodes with NVIDIA GPUs
- **CLOUD-402:** Multi-tenant logs/metrics MUST carry `tenant_id`
- **CLOUD-810:** Preload required models before marking pool Ready (dedicated baseline)
- **CLOUD-1100-1103:** Kubernetes deployment with Helm chart, NVIDIA device plugin, HPA/KEDA autoscaling

**Would standalone pool-managerd daemon help with CLOUD profile?**

**Answer: YES — standalone daemon enables better cloud deployment patterns**

**Why standalone daemon makes sense for CLOUD:**

1. ✅ **Separation of concerns:** orchestratord handles HTTP/auth/quotas/metering; pool-managerd handles GPU lifecycle/supervision
2. ✅ **Multi-tenant isolation:** Different pools can run in different namespaces/VPCs with separate pool-managerd instances
3. ✅ **Kubernetes-native:** pool-managerd as DaemonSet on GPU nodes; orchestratord as Deployment behind LB
4. ✅ **Autoscaling:** HPA/KEDA can scale orchestratord pods independently of GPU pool supervision
5. ✅ **Fault isolation:** pool-managerd crash on one GPU node doesn't affect orchestratord or other pools
6. ✅ **On-demand startup:** pool-managerd can start engines on first request, not preload (aligns with user's clarification)
7. ✅ **Control API:** orchestratord queries pool-managerd via HTTP/gRPC for health/placement decisions

**Architecture Evolution Path:**

**Phase 1 — Home Profile (Current):**

- orchestratord embeds pool-managerd as library
- Single binary, single workstation
- On-demand engine startup when first request arrives

**Phase 2 — Cloud Profile (Future):**

- pool-managerd as standalone daemon (DaemonSet on k8s GPU nodes)
- orchestratord queries pool-managerd via HTTP control API
- Multi-tenant: separate pool-managerd instances per namespace/VPC
- On-demand engine startup with preload option (CLOUD-810 for dedicated)

**Standalone Daemon Responsibilities (Cloud Profile):**

1. **GPU Discovery** — enumerate NVIDIA devices, collect compute_capability, VRAM totals/free
2. **On-Demand Startup** — start engines when first request arrives (home profile) OR preload (CLOUD-810 dedicated)
3. **Engine Supervision** — spawn/monitor engine processes, restart with backoff on crash
4. **Health Monitoring** — periodic health checks, update registry, detect driver errors
5. **Control API** — HTTP endpoints for orchestratord to query/update registry (drain/reload, health, placement)

**Current Evidence for Future Standalone:**

1. **CLOUD_PROFILE.md lines 180-187** — Kubernetes deployment with DaemonSet pattern
2. **CLOUD_PROFILE.md line 810** — "Preload required models before marking pool Ready" (dedicated baseline)
3. **.plan/30_pool_managerd.md:67-74** — proposes future extraction "if compile times grow" OR deployment model changes
4. **CLOUD_PROFILE.md line 267** — "Cloud = presets + DevOps + quotas + metering shim. Most changes are config + small hooks, not deep rewrites."

**Recommendation: KEEP `src/main.rs` STUB**

**Rationale:**

1. ✅ CLOUD_PROFILE.md suggests standalone daemon for multi-tenant/k8s deployment
2. ✅ Stub preserves future migration path without premature implementation
3. ✅ Home profile continues to embed as library (no change needed now)
4. ✅ On-demand engine startup works in both architectures
5. ✅ Modular structure (registry.rs, backoff.rs, drain.rs) makes extraction easy when needed

**Updated Stub Comment (add to src/main.rs):**

```rust
//! pool-managerd daemon entrypoint.
//!
//! STATUS: STUB (Home profile embeds as library; Cloud profile will use standalone daemon)
//!
//! Home Profile (Current):
//! - orchestratord embeds pool-managerd::registry as library (bin/orchestratord/src/state.rs:6)
//! - Single binary, single workstation, on-demand engine startup
//!
//! Cloud Profile (Future, per .specs/proposals/CLOUD_PROFILE.md):
//! - pool-managerd as standalone daemon (DaemonSet on k8s GPU nodes)
//! - orchestratord queries via HTTP control API for health/placement
//! - Multi-tenant: separate instances per namespace/VPC
//! - On-demand startup (home) OR preload (CLOUD-810 dedicated baseline)
//!
//! Responsibilities (when implemented):
//! 1. GPU Discovery — enumerate NVIDIA devices, collect compute_capability, VRAM
//! 2. On-Demand Startup — start engines when needed (or preload for dedicated)
//! 3. Engine Supervision — spawn/monitor processes, restart with backoff on crash
//! 4. Health Monitoring — periodic checks, update registry, detect driver errors
//! 5. Control API — HTTP endpoints for orchestratord (drain/reload, health, placement)
//!
//! Implementation trigger: When implementing CLOUD_PROFILE.md (k8s deployment)
//! Until then: Home profile continues to embed as library (no changes needed)

fn main() {
    println!("pool-managerd stub");
    eprintln!("Home profile: orchestratord embeds pool-managerd as library");
    eprintln!("Cloud profile: implement standalone daemon per CLOUD_PROFILE.md");
}
```

**Implementation Path (when Cloud Profile is needed):**

1. Implement HTTP control API in pool-managerd (health, drain/reload, placement queries)
2. Implement supervision module (calls backoff.rs, monitors engine health)
3. Implement on-demand startup orchestration (calls preload.rs when first request arrives)
4. Add Kubernetes manifests (DaemonSet for pool-managerd, Deployment for orchestratord)
5. Update orchestratord to query pool-managerd via HTTP instead of embedded Registry
6. Keep backward compat: Home profile can still embed as library via feature flag

**Why This Approach Works:**

- ✅ No premature optimization: Home profile stays simple (embedded library)
- ✅ Clear migration path: CLOUD_PROFILE.md provides requirements when needed
- ✅ On-demand startup: Works in both architectures (embedded or standalone)
- ✅ Modular: Current structure (registry.rs, backoff.rs, drain.rs) makes extraction easy
- ✅ Binary vs library: Standalone daemon is a binary; embedded is a library (both valid)

---

**Summary of Recommendations:**

| File | Recommendation | Rationale |
|------|---------------|-----------|
| `src/leases.rs` | **DELETE** | Narration at emission points provides observability; registry counters sufficient; no usage; YAGNI |
| `src/main.rs` | **KEEP STUB** | CLOUD_PROFILE.md suggests standalone daemon for k8s/multi-tenant; preserves migration path; home profile continues embedded |

**Alignment:**

- ✅ **Home profile (current):** orchestratord embeds pool-managerd as library (lightweight, single binary)
- ✅ **Cloud profile (future):** pool-managerd as standalone daemon (k8s DaemonSet, multi-tenant isolation)
- ✅ **On-demand startup:** Works in both architectures (start engines when needed, not preloaded)
- ✅ **Observability:** Human narration at admission/placement/stream emission points provides lease tracking without per-lease module
- ✅ **YAGNI:** Don't implement standalone daemon until CLOUD_PROFILE.md is needed; stub preserves path

---

## Spec Traceability

### Specs Fulfilled

- ✅ **ORCH-3002:** Preload/Ready lifecycle (registry + health.rs)
- ✅ **ORCH-3010:** Dispatch to Ready replicas (registry.get_health, allocate_lease)
- ✅ **ORCH-3027:** Logs include pool_id, engine_version, etc. (registry fields exist)
- ✅ **ORCH-3028:** Metrics for queue depth, active_leases (registry exposes data)
- ✅ **ORCH-1102:** GPU-only enforcement (preflight.rs)
- ✅ **ORCH-3202:** Preflight checks (preflight.rs)

### Specs Pending Implementation

- ❌ **ORCH-3004:** Bounded FIFO queue (orchestrator-core, not pool-managerd)
- ❌ **ORCH-3005:** Full-queue policy (orchestrator-core, not pool-managerd)
- ❌ **ORCH-3011:** Respect device masks (devicemasks.rs stub)
- ❌ **ORCH-3031:** Atomic drain/reload (drain.rs stub)
- ❌ **ORCH-3038:** Driver/CUDA errors → restart with backoff (backoff.rs stub)
- ❌ **ORCH-3040:** Circuit breakers (backoff.rs stub)
- ❌ **ORCH-3052:** Heterogeneous tensor splits (hetero_split.rs stub)
- ❌ **ORCH-3003:** Fail fast on insufficient VRAM/RAM (preload.rs stub)

---

## BDD Test Coverage

### Implemented Steps (test-harness/bdd/src/steps/pool_manager.rs)

- ✅ `given_pool_unready_due_to_preload_failure` (lines 5-12) → uses registry.set_health, set_last_error
- ✅ `then_pool_readiness_false_last_error_present` (lines 15-26) → queries registry
- ✅ `given_driver_error_occurs` (lines 28-35) → uses registry.set_health, set_last_error
- ✅ `then_pool_unready_and_restarts_with_backoff` (lines 37-46) → queries registry

### Stub Steps (awaiting implementation)

- ❌ `then_restart_storms_bounded_by_circuit_breaker` (line 48) → needs backoff.rs
- ❌ `given_device_masks_configured` (line 52) → needs devicemasks.rs
- ❌ `then_placement_respects_device_masks_no_spill` (line 56) → needs devicemasks.rs + placement
- ❌ `given_heterogeneous_split_ratios_configured` (line 60) → needs hetero_split.rs
- ❌ `then_per_gpu_kv_capped_smallest_gpu` (line 64) → needs hetero_split.rs

---

## Recommendations

### Immediate Actions

1. **Keep as-is:**
   - `health.rs` (fully implemented, actively used)
   - `preflight.rs` (fully implemented, expand later)
   - `registry.rs` + submodules (partially implemented, expand VRAM fields)

2. **Implement next (priority order):**
   - `preload.rs` → gates readiness (ORCH-3002, 3003)
   - `backoff.rs` → restart supervision (ORCH-3038, 3040)
   - `drain.rs` → atomic reload (ORCH-3031, 3038)
   - `devicemasks.rs` → placement enforcement (ORCH-3010, 3011, 3052)
   - `hetero_split.rs` → multi-GPU support (ORCH-3052)

3. **Decision needed:**
   - `leases.rs` → delete if registry counters are sufficient, else expand for per-lease metadata
   - `main.rs` → delete if orchestratord embeds pool-managerd, else implement full daemon

### Long-term Roadmap

- Add VRAM tracking to registry (vram_total_bytes, vram_free_bytes, compute_capability)
- Add perf hints to registry (tokens_per_s, first_token_ms)
- Implement supervision module (calls backoff.rs, monitors engine health)
- Implement placement module (calls devicemasks.rs, consumes registry snapshots)
- Wire drain/reload to orchestratord control API
- Expand preflight.rs for all required tools (git, cmake, nvcc, podman, huggingface-cli, etc.)

---

## References

- Spec: `.specs/00_llama-orch.md` (ORCH-3xxx requirements)
- Checklist: `libs/pool-managerd/CHECKLIST.md` (production readiness)
- Owner tasks: `TODO_OWNERS_MVP_pt3.md` (Owner E completed, Owner F pending)
- BDD tests: `test-harness/bdd/src/steps/pool_manager.rs`
- Integration: `bin/orchestratord/src/state.rs:6` (embeds Registry)

---

**Next Steps:**

1. Review this analysis with the team
2. Decide on `leases.rs` and `main.rs` (delete or implement)
3. Prioritize stub implementations based on roadmap gates
4. Update `TODO.md` and `TODO_OWNERS_MVP_pt3.md` with findings
