# pool-managerd Responsibility Audit & Cloud Profile Readiness

**Date:** 2025-09-30  
**Purpose:** Verify pool-managerd responsibilities are correct, not duplicated, and future-proof for CLOUD_PROFILE.md

---

## Executive Summary

**CRITICAL FINDINGS:**

1. ❌ **MAJOR OVERLAP:** engine-provisioner already does GPU discovery, engine startup, supervision, and health monitoring
2. ❌ **pool-managerd has NO unique responsibilities** — everything is already done by engine-provisioner or orchestratord
3. ❌ **BDD setup is BROKEN:** pool-managerd/bdd has empty World, no step definitions, no features
4. ✅ **Registry is the ONLY useful part** — health/lease tracking used by orchestratord

**RECOMMENDATION:** pool-managerd should be **DELETED or RADICALLY SIMPLIFIED**

---

## Responsibility Analysis

### Claimed Responsibilities (from src/main.rs)

| # | Responsibility | Current Owner | Overlap? |
|---|----------------|---------------|----------|
| 1 | GPU Discovery | **engine-provisioner** | ❌ DUPLICATE |
| 2 | On-Demand Startup | **engine-provisioner** | ❌ DUPLICATE |
| 3 | Engine Supervision | **engine-provisioner** | ❌ DUPLICATE |
| 4 | Health Monitoring | **engine-provisioner** | ❌ DUPLICATE |
| 5 | Control API | **orchestratord** | ❌ DUPLICATE |

---

## Detailed Overlap Analysis

### 1. GPU Discovery — ALREADY DONE by engine-provisioner

**engine-provisioner does:**
- CUDA toolkit detection (`libs/provisioners/engine-provisioner/src/providers/llamacpp/preflight.rs`)
- NVIDIA driver detection (nvcc, nvidia-smi)
- GPU capability checks
- Device enumeration via CUDA flags

**Evidence:**
```rust
// libs/provisioners/engine-provisioner/src/providers/llamacpp/preflight.rs
pub fn preflight_tools(prov: &cfg::ProvisioningConfig, src: &cfg::Source) -> Result<()> {
    // Checks for git, cmake, make, nvcc, nvidia-smi
    // Optionally installs via pacman on Arch
}

// libs/provisioners/engine-provisioner/src/providers/llamacpp/toolchain.rs
pub fn discover_cuda_root() -> Option<PathBuf> {
    // Discovers CUDA installation
}
```

**pool-managerd has:**
- `src/preflight.rs` — CUDA detection (cuda_available, assert_gpu_only)
- `src/devicemasks.rs` — stub (no implementation)

**Verdict:** ❌ **DUPLICATE** — engine-provisioner already does this

---

### 2. On-Demand Startup — ALREADY DONE by engine-provisioner

**engine-provisioner does:**
- Clones/builds llama.cpp from source
- Calls model-provisioner to fetch GGUF
- Spawns llama-server process
- Writes handoff JSON with URL/port
- Writes PID file for supervision

**Evidence:**
```rust
// libs/provisioners/engine-provisioner/src/providers/llamacpp/mod.rs:200-280
impl EngineProvisioner for LlamaCppSourceProvisioner {
    fn ensure(&self, pool: &cfg::PoolConfig) -> Result<()> {
        // 1. Preflight tools
        // 2. Clone/update repo
        // 3. CMake configure + build
        // 4. Fetch model via model-provisioner
        // 5. Spawn llama-server
        // 6. Write handoff JSON
        // 7. Write PID file
    }
}
```

**pool-managerd has:**
- `src/preload.rs` — stub (no implementation)

**Verdict:** ❌ **DUPLICATE** — engine-provisioner already does this

---

### 3. Engine Supervision — ALREADY DONE by engine-provisioner

**engine-provisioner does:**
- Spawns engine process
- Writes PID file
- Provides `stop_pool(pool_id)` to kill process (SIGTERM → SIGKILL)
- Has restart-on-crash test (`tests/restart_on_crash.rs`)

**Evidence:**
```rust
// libs/provisioners/engine-provisioner/src/lib.rs:9-37
pub fn stop_pool(pool_id: &str) -> Result<()> {
    // Reads PID file
    // Sends SIGTERM
    // Waits 5 seconds for graceful shutdown
    // Sends SIGKILL if still alive
    // Removes PID file
}
```

**pool-managerd has:**
- `src/backoff.rs` — stub (BackoffPolicy struct, no implementation)

**Verdict:** ❌ **DUPLICATE** — engine-provisioner already does this (though backoff logic could be added)

---

### 4. Health Monitoring — ALREADY DONE by engine-provisioner

**engine-provisioner does:**
- Waits for engine health endpoint (`wait_for_health` in util.rs)
- Polls HTTP endpoint until ready
- Writes handoff JSON only after health check passes

**Evidence:**
```rust
// libs/provisioners/engine-provisioner/src/util.rs
pub fn wait_for_health(url: &str, timeout_s: u64) -> Result<()> {
    // Polls health endpoint until ready or timeout
}

// libs/provisioners/engine-provisioner/src/providers/llamacpp/mod.rs:270
wait_for_health(&health_url, 30)?;
```

**pool-managerd has:**
- `src/health.rs` — HealthStatus struct (used by registry)
- No health monitoring logic

**Verdict:** ⚠️ **PARTIAL OVERLAP** — engine-provisioner does initial health check; pool-managerd registry stores health state

---

### 5. Control API — ALREADY DONE by orchestratord

**orchestratord does:**
- `GET /v1/pools/{id}/health` — queries registry (bin/orchestratord/src/api/control.rs)
- Drain/reload endpoints (planned)
- Embeds pool-managerd::registry directly

**Evidence:**
```rust
// bin/orchestratord/src/state.rs:6
use pool_managerd::registry::Registry as PoolRegistry;
pub struct AppState {
    pub pool_manager: Arc<Mutex<PoolRegistry>>,
}

// bin/orchestratord/src/api/control.rs:26-35
pub async fn pool_health(state: &AppState, id: String) -> Result<impl IntoResponse> {
    let reg = state.pool_manager.lock().expect("pool_manager lock");
    let h = reg.get_health(&id).unwrap_or(...);
}
```

**pool-managerd has:**
- `src/drain.rs` — stub (DrainRequest/ReloadRequest, no implementation)
- No HTTP server

**Verdict:** ❌ **DUPLICATE** — orchestratord already provides control API

---

## What pool-managerd ACTUALLY Provides

### ✅ Registry (USEFUL)

**Used by:**
- orchestratord (state.rs:6) — embeds Registry
- test-harness/bdd (pool_manager.rs:3) — imports HealthStatus

**Provides:**
- Health tracking (live/ready)
- Lease counters (allocate/release, never negative)
- Heartbeat tracking
- Version/engine metadata
- Draining flag
- Snapshots for placement

**Verdict:** ✅ **KEEP** — this is the only unique, actively used part

---

### ❌ Everything Else (STUBS)

| Module | Status | Verdict |
|--------|--------|---------|
| `backoff.rs` | Stub (BackoffPolicy struct, no logic) | ❌ DELETE or move to engine-provisioner |
| `devicemasks.rs` | Stub (DeviceMask struct, no logic) | ❌ DELETE or move to placement |
| `drain.rs` | Stub (DrainRequest/ReloadRequest, no logic) | ❌ DELETE or move to orchestratord |
| `hetero_split.rs` | Stub (SplitPlan struct, no logic) | ❌ DELETE or move to placement |
| `leases.rs` | Stub (Lease/LeaseId, no logic, no usage) | ❌ DELETE (already decided) |
| `preload.rs` | Stub (PreloadOutcome, no logic) | ❌ DELETE (engine-provisioner does this) |
| `preflight.rs` | Implemented (CUDA detection) | ⚠️ DUPLICATE (engine-provisioner has same) |
| `main.rs` | Stub daemon entrypoint | ⚠️ KEEP STUB (cloud profile may need) |

---

## BDD Setup Analysis

### pool-managerd/bdd Status: ❌ **BROKEN**

**What exists:**
- `bdd/Cargo.toml` — defines `bdd-runner` binary
- `bdd/src/main.rs` — cucumber runner (looks for `tests/features`)
- `bdd/src/steps/world.rs` — **EMPTY World struct** (no state!)
- `bdd/src/steps/mod.rs` — only exports world
- `bdd/tests/features/` — **EMPTY directory** (no .feature files!)

**What's missing:**
- ❌ No .feature files
- ❌ No step definitions (world.rs is empty)
- ❌ No state (World has no fields)
- ❌ No tests

**Comparison with test-harness/bdd:**
- test-harness/bdd has **World with AppState** (includes pool_manager: Arc<Mutex<Registry>>)
- test-harness/bdd has **pool_manager.rs step definitions** (6 steps using pool-managerd::health::HealthStatus)
- test-harness/bdd has **.feature files** in tests/features/

**Verdict:** ❌ **pool-managerd/bdd is USELESS** — all BDD tests are in test-harness/bdd

---

## Cloud Profile Readiness Analysis

### Question: Is pool-managerd future-proof for CLOUD_PROFILE.md?

**Answer: NO — current design has fundamental issues**

### Issues:

1. **Responsibility overlap:** engine-provisioner already does everything pool-managerd claims to do
2. **No unique value:** Only Registry is useful; everything else is stubs or duplicates
3. **Architectural confusion:** Is pool-managerd a daemon or a library?
   - Home profile: embedded library (just Registry)
   - Cloud profile: standalone daemon (but what would it do that engine-provisioner doesn't?)

### Cloud Profile Requirements (from CLOUD_PROFILE.md):

| Requirement | Current Owner | pool-managerd Needed? |
|-------------|---------------|----------------------|
| CLOUD-300: GPU nodes | engine-provisioner | ❌ NO |
| CLOUD-810: Preload models | engine-provisioner | ❌ NO |
| CLOUD-1100: k8s DaemonSet | ? | ⚠️ MAYBE |
| CLOUD-402: Multi-tenant isolation | orchestratord | ❌ NO |

### Cloud Profile Architecture Options:

**Option A: Delete pool-managerd, use engine-provisioner as DaemonSet**
- ✅ engine-provisioner already does GPU discovery, startup, supervision
- ✅ No code duplication
- ✅ Simpler architecture
- ❌ Loses Registry abstraction (but orchestratord could embed it directly)

**Option B: Keep pool-managerd as thin wrapper around engine-provisioner**
- ✅ Preserves Registry abstraction
- ✅ Could add supervision/backoff logic on top of engine-provisioner
- ❌ Adds extra layer of indirection
- ❌ Unclear value vs. just using engine-provisioner

**Option C: Radically simplify pool-managerd to ONLY Registry**
- ✅ Clear responsibility: health/lease tracking
- ✅ No overlap with engine-provisioner
- ✅ Works for both home (embedded) and cloud (standalone HTTP API)
- ✅ Aligns with current usage (orchestratord only uses Registry)

---

## Recommendations

### Immediate Actions

1. **DELETE pool-managerd/bdd** — it's broken and unused; all BDD tests are in test-harness/bdd
2. **DELETE src/leases.rs** — already decided (no usage, narration provides observability)
3. **DECIDE on pool-managerd scope:**
   - **Option 1 (RECOMMENDED):** Simplify to ONLY Registry + health/lease tracking
   - **Option 2:** Delete entirely, move Registry to orchestratord
   - **Option 3:** Keep stubs but document they're for future cloud profile

### Scope Clarification Needed

**What should pool-managerd actually do?**

**Current reality:**
- ✅ Registry (health/lease tracking) — actively used
- ❌ Everything else — stubs or duplicates of engine-provisioner

**Proposed scope (Option 1 — Simplify):**
```
pool-managerd/
├── src/
│   ├── lib.rs          # pub mod registry, pub mod health
│   ├── registry.rs     # ✅ KEEP (actively used)
│   ├── health.rs       # ✅ KEEP (used by registry)
│   ├── main.rs         # ⚠️ KEEP STUB (cloud profile may need HTTP API)
│   ├── backoff.rs      # ❌ DELETE (move to engine-provisioner if needed)
│   ├── devicemasks.rs  # ❌ DELETE (move to placement if needed)
│   ├── drain.rs        # ❌ DELETE (move to orchestratord if needed)
│   ├── hetero_split.rs # ❌ DELETE (move to placement if needed)
│   ├── leases.rs       # ❌ DELETE (already decided)
│   ├── preload.rs      # ❌ DELETE (engine-provisioner does this)
│   └── preflight.rs    # ❌ DELETE (engine-provisioner has same)
└── bdd/                # ❌ DELETE (broken, unused)
```

**Rationale:**
- pool-managerd = **state tracker** (registry)
- engine-provisioner = **lifecycle manager** (startup, supervision, health checks)
- orchestratord = **control plane** (HTTP API, placement, admission)

**Cloud profile:**
- pool-managerd as standalone daemon = HTTP API wrapping Registry
- engine-provisioner as DaemonSet = manages engine processes on GPU nodes
- orchestratord as Deployment = queries pool-managerd HTTP API for placement decisions

---

## Questions for Decision

1. **Should pool-managerd exist at all?**
   - If YES: What is its unique responsibility vs. engine-provisioner?
   - If NO: Move Registry to orchestratord?

2. **Should pool-managerd be a daemon in cloud profile?**
   - If YES: What does it do that engine-provisioner doesn't?
   - If NO: Why keep src/main.rs stub?

3. **Should supervision/backoff logic live in pool-managerd or engine-provisioner?**
   - Current: engine-provisioner has stop_pool() and restart-on-crash test
   - Future: Who owns restart-with-backoff logic?

4. **Should BDD tests for pool manager stay in test-harness/bdd or move to pool-managerd/bdd?**
   - Current: test-harness/bdd has all pool_manager step definitions
   - pool-managerd/bdd is broken and empty

---

## Conclusion

**pool-managerd is NOT future-proof for cloud profile in its current form.**

**Core issue:** Massive responsibility overlap with engine-provisioner. Almost everything pool-managerd claims to do is already done by engine-provisioner.

**Path forward:** Clarify scope and delete/move modules to their correct homes.
