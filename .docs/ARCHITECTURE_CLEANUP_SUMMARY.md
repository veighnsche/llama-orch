# Architecture Cleanup Summary

**Date**: 2025-10-03 01:16  
**Status**: ✅ Complete

---

## What Was Done

Executed massive architecture cleanup to align with corrected trio-binary model where:
- **Orchestrator** = THE BRAIN (all intelligence)
- **Pool Manager** = STATE REPORTER + WORKER FACTORY (no decisions)
- **Worker** = DUMB EXECUTOR (one model, simple execution)

---

## Crates Deleted

### ❌ Wrong-Layer Crates Removed

1. **`bin/worker-orcd-crates/vram-residency/`** (entire directory + BDD)
   - Mixed pool manager and worker concerns
   - Replaced by `gpu-inventory` + `vram-policy`

2. **`bin/worker-orcd-crates/scheduler/`**
   - Workers don't schedule (orchestrator does)

3. **`bin/pool-managerd-crates/router/`**
   - Pool manager doesn't route (orchestrator does)

4. **`bin/pool-managerd-crates/model-eviction/`**
   - Pool manager doesn't decide eviction (orchestrator does)

---

## Crates Created

### ✅ New Crates with Correct Boundaries

1. **`bin/pool-managerd-crates/gpu-inventory/`**
   - **Purpose**: Multi-GPU VRAM capacity tracking
   - **API**: `can_fit_model()`, `register_worker()`, `available_vram()`
   - **Tests**: 3 unit tests included
   - **Status**: ✅ Compiles

2. **`bin/worker-orcd-crates/vram-policy/`**
   - **Purpose**: Single-model VRAM-only enforcement
   - **API**: `enforce_vram_only()`, `load_model_to_vram()`, `verify_vram_residency()`
   - **Tests**: 2 unit tests included
   - **Status**: ✅ Compiles

### ✅ Stubs Created (Dependencies)

3. **`bin/orchestratord-crates/node-registry/`**
   - Stub for `service-registry` dependency
   - Status: ✅ Compiles

4. **`bin/pool-managerd-crates/pool-registry/`**
   - Stub (needs rename to `worker-registry`)
   - Status: ✅ Compiles

5. **`bin/pool-managerd-crates/node-registration-client/`**
   - Stub for node registration
   - Status: ✅ Compiles

---

## Files Modified

### Cargo.toml (Workspace)
```diff
# Pool Manager Crates
+ "bin/pool-managerd-crates/gpu-inventory",
- "bin/pool-managerd-crates/model-eviction",
- "bin/pool-managerd-crates/router",

# Worker Crates
+ "bin/worker-orcd-crates/vram-policy",
- "bin/worker-orcd-crates/vram-residency",
- "bin/worker-orcd-crates/vram-residency/bdd",
- "bin/worker-orcd-crates/scheduler",
```

### bin/worker-orcd/Cargo.toml
```diff
- vram-residency = { path = "../worker-orcd-crates/vram-residency" }
+ vram-policy = { path = "../worker-orcd-crates/vram-policy" }
- scheduler = { path = "../worker-orcd-crates/scheduler" }
```

---

## Documentation Created

### ✅ Authoritative Architecture Doc

**`.docs/ARCHITECTURE_TRIO_CORRECTED.md`** (500+ lines)
- Clear boundaries for all three binaries
- Data flow examples
- Terminology dictionary
- Crate responsibility mapping
- **This is now the single source of truth**

### ✅ Cleanup Logs

**`.docs/ARCHITECTURE_CLEANUP_LOG.md`**
- Complete change inventory
- Migration status
- Documentation debt tracker

**`.docs/ARCHITECTURE_CLEANUP_SUMMARY.md`** (this file)
- Quick reference for what changed

---

## Verification

```fish
# New crates compile
cargo check -p gpu-inventory
cargo check -p vram-policy
cargo check -p pool-registry

# Workspace state
✅ 4 crates deleted
✅ 2 new crates created (gpu-inventory, vram-policy)
✅ 3 stubs created (node-registry, pool-registry, node-registration-client)
✅ Workspace Cargo.toml updated
✅ worker-orcd dependencies fixed
```

---

## Architecture Summary

### Orchestrator (THE BRAIN)
**Crates**: orchestrator-core, placement, streaming, task-cancellation, catalog-core, backpressure  
**Does**: Queue, schedule, route, place, evict, catalog, stream to clients  
**Does NOT**: Execute inference, manage GPUs, start workers

### Pool Manager (STATE REPORTER)
**Crates**: gpu-inventory ✨NEW, worker-registry, model-cache, lifecycle, health-monitor, api  
**Does**: Track VRAM, track workers, spawn workers, stage models in RAM, report state  
**Does NOT**: Make decisions, route, schedule, evict

### Worker (DUMB EXECUTOR)
**Crates**: vram-policy ✨NEW, model-loader, api, capability-matcher  
**Does**: Load ONE model, execute inference, enforce VRAM-only, stream results  
**Does NOT**: Load multiple models, schedule, place

---

## Data Flow (Correct)

```
1. Orchestrator asks Pool Manager: "Can you fit llama-13b (26GB)?"
2. Pool Manager (gpu-inventory): GPU 1 has 20GB free → "Yes"
3. Orchestrator: "Start worker for llama-13b on GPU 1"
4. Pool Manager: Spawns worker, worker loads to VRAM
5. Worker: Calls back "Ready, using 26GB"
6. Pool Manager: Updates gpu-inventory
7. Orchestrator: Queries state, sees worker
8. Orchestrator: Connects DIRECTLY to worker for inference
9. Worker: Streams SSE directly to orchestrator
```

**Key insight**: Pool manager is NOT in the data path!

---

## Known Documentation Debt

### Critical (Needs Update)
- `.specs/00_llama-orch.md` — References "engine provisioning"
- `.specs/20-orchestratord.md` — References "adapters"
- `.specs/30-pool-managerd.md` — Says pool manager manages "pools"
- `bin/orchestratord/README.md` — Architecture diagram shows adapters
- `bin/pool-managerd/README.md` — Wrong responsibilities

### Medium
- `.docs/ARCHITECTURE_CHANGE_PLAN.md` — Pool manager too smart
- `.docs/WORKER_READINESS_CALLBACK_DESIGN.md` — Minor fixes
- All crate READMEs — Check terminology

### Terminology Cleanup
- "Pool" → "Worker"
- "Engine" → "Worker"
- "Adapter" → "Direct connection"
- "Engine provisioning" → "Worker spawning"

---

## Status

- ✅ **Crate cleanup**: Complete
- ✅ **New crates**: Created and compiling
- ✅ **Workspace**: Updated
- ✅ **Dependencies**: Fixed
- ✅ **Architecture doc**: Authoritative reference created
- ⏳ **Specs**: Need updates (documented in cleanup log)
- ⏳ **READMEs**: Need updates (documented in cleanup log)

---

## Next Actions

1. **Update specs** (`.specs/00_llama-orch.md`, `20-orchestratord.md`, `30-pool-managerd.md`)
2. **Update READMEs** (orchestratord, pool-managerd, worker-orcd)
3. **Rename** `pool-registry` → `worker-registry`
4. **Add warnings** to outdated docs pointing to ARCHITECTURE_TRIO_CORRECTED.md

---

## Reference

**Authoritative Architecture**: `.docs/ARCHITECTURE_TRIO_CORRECTED.md`  
**Complete Change Log**: `.docs/ARCHITECTURE_CLEANUP_LOG.md`  
**This Summary**: `.docs/ARCHITECTURE_CLEANUP_SUMMARY.md`

---

**Cleanup executed by**: Cascade  
**Date**: 2025-10-03 01:16 UTC+02:00  
**Time taken**: ~15 minutes  
**Lines of code changed**: ~1000+ (mostly deletions)  
**Status**: ✅ **COMPLETE**
