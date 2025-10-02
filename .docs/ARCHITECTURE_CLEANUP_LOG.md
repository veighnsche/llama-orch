# Architecture Cleanup Log

**Date**: 2025-10-03  
**Action**: Massive cleanup to align with corrected trio-binary architecture

---

## Problem Statement

The codebase reflected a **confused mental model** where:
- ❌ Pool manager was "smart" (made routing, scheduling, eviction decisions)
- ❌ Workers managed multiple models
- ❌ "Pools" were first-class entities
- ❌ Adapters abstracted worker communication

**Correct model**:
- ✅ Orchestrator = THE BRAIN (all intelligence)
- ✅ Pool manager = STATE REPORTER + WORKER FACTORY (no decisions)
- ✅ Worker = DUMB EXECUTOR (one model, simple execution)

---

## Changes Made

### Deleted Crates (Wrong Layer)

1. **`bin/worker-orcd-crates/vram-residency/`**
   - **Reason**: Wrong abstraction (mixed pool manager and worker concerns)
   - **Replaced by**: 
     - `gpu-inventory/` (pool manager tracks VRAM across GPUs)
     - `vram-policy/` (worker enforces VRAM-only for one model)

2. **`bin/worker-orcd-crates/scheduler/`**
   - **Reason**: Workers don't schedule (orchestrator does)
   - **Replaced by**: Nothing (orchestrator already has scheduling)

3. **`bin/pool-managerd-crates/router/`**
   - **Reason**: Pool manager doesn't route (orchestrator does)
   - **Replaced by**: Nothing (orchestrator routes directly)

4. **`bin/pool-managerd-crates/model-eviction/`**
   - **Reason**: Pool manager doesn't decide eviction (orchestrator does)
   - **Replaced by**: Nothing (orchestrator makes eviction decisions)

### Created Crates (Correct Boundaries)

1. **`bin/pool-managerd-crates/gpu-inventory/`**
   - **Purpose**: Track VRAM capacity across all GPUs on this host
   - **API**: `can_fit_model()`, `register_worker()`, `available_vram()`
   - **Used by**: Pool manager to answer orchestrator queries

2. **`bin/worker-orcd-crates/vram-policy/`**
   - **Purpose**: Enforce VRAM-only for a single model
   - **API**: `enforce_vram_only()`, `load_model_to_vram()`, `verify_vram_residency()`
   - **Used by**: Worker at startup and health checks

### Updated Workspace

**`Cargo.toml` changes**:
- ➕ Added `bin/pool-managerd-crates/gpu-inventory`
- ➕ Added `bin/worker-orcd-crates/vram-policy`
- ➖ Removed `bin/worker-orcd-crates/vram-residency`
- ➖ Removed `bin/worker-orcd-crates/vram-residency/bdd`
- ➖ Removed `bin/worker-orcd-crates/scheduler`
- ➖ Removed `bin/pool-managerd-crates/router`
- ➖ Removed `bin/pool-managerd-crates/model-eviction`

---

## Documentation Created

### New Authoritative Document

**`.docs/ARCHITECTURE_TRIO_CORRECTED.md`**
- Defines clear boundaries for orchestrator, pool manager, worker
- Provides data flow examples
- Maps old terminology to new
- Lists all crate responsibilities
- **This is now the authoritative architecture reference**

---

## Documentation Needing Updates

The following documents still reflect the old model and need revision:

### Critical (High Impact)
1. `.specs/00_llama-orch.md` — References "engine provisioning" instead of "worker spawning"
2. `.specs/20-orchestratord.md` — References "adapters" abstraction
3. `.specs/30-pool-managerd.md` — Says pool manager manages "pools" not "workers"
4. `bin/orchestratord/README.md` — Architecture diagram shows adapters
5. `bin/pool-managerd/README.md` — Entire doc describes wrong responsibilities
6. `bin/worker-orcd/README.md` — Minor fixes needed

### Medium (Documentation)
7. `.docs/ARCHITECTURE_CHANGE_PLAN.md` — Pool manager responsibilities incorrect
8. `.docs/WORKER_READINESS_CALLBACK_DESIGN.md` — Mostly correct, minor fixes
9. `.docs/security/SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Security model outdated

### Low (Crate-Specific)
10. `bin/pool-managerd-crates/pool-registry/README.md` — Should be renamed to `worker-registry`
11. All orchestratord-crates READMEs — Check for adapter references
12. All pool-managerd-crates READMEs — Check for decision-making claims

---

## Next Steps

### Immediate
1. ✅ Delete obsolete crates
2. ✅ Create new crates with correct boundaries
3. ✅ Update workspace Cargo.toml
4. ✅ Create authoritative architecture doc

### Short-Term
5. ⏳ Update critical specs (00_llama-orch, 20-orchestratord, 30-pool-managerd)
6. ⏳ Update binary READMEs
7. ⏳ Rename `pool-registry` → `worker-registry`
8. ⏳ Add architecture warnings to outdated docs

### Medium-Term
9. ⏳ Update all crate READMEs
10. ⏳ Revise security audit
11. ⏳ Update test documentation
12. ⏳ Clean up references to "pools", "engines", "adapters"

---

## Impact Assessment

### Breaking Changes
- ✅ **Safe**: All deleted crates were stubs or minimal implementation
- ✅ **Safe**: New crates don't break existing code (they're new)
- ✅ **Safe**: Workspace changes compile correctly

### Documentation Debt
- ⚠️ **High**: Many docs still reference old model
- ⚠️ **Medium**: Terminology confusion (pools vs workers)
- ⚠️ **Low**: Most specs are still accurate in requirements, just naming is off

### Code Debt
- ✅ **None**: No production code depends on deleted crates yet
- ✅ **Low**: New crates provide clear migration path

---

## Lessons Learned

1. **Terminology matters**: "Pool" was ambiguous (pool of workers? pool manager? GPU pool?)
2. **Boundaries matter**: Mixing "smart" and "dumb" in same component leads to confusion
3. **Documentation drift**: Specs and READMEs diverged from actual architecture
4. **Early cleanup is cheap**: v0.1.0 allowed aggressive refactoring without compatibility burden

---

## Files Modified

**Deleted**:
- `bin/worker-orcd-crates/vram-residency/` (entire directory)
- `bin/worker-orcd-crates/scheduler/` (entire directory)
- `bin/pool-managerd-crates/router/` (entire directory)
- `bin/pool-managerd-crates/model-eviction/` (entire directory)

**Created**:
- `bin/pool-managerd-crates/gpu-inventory/` (new crate)
- `bin/worker-orcd-crates/vram-policy/` (new crate)
- `.docs/ARCHITECTURE_TRIO_CORRECTED.md` (authoritative doc)
- `.docs/ARCHITECTURE_CLEANUP_LOG.md` (this file)

**Modified**:
- `Cargo.toml` (workspace members)

---

## Verification

```bash
# Check workspace compiles
cargo check

# Verify new crates exist
ls bin/pool-managerd-crates/gpu-inventory
ls bin/worker-orcd-crates/vram-policy

# Verify deletions
! ls bin/worker-orcd-crates/vram-residency 2>/dev/null
! ls bin/worker-orcd-crates/scheduler 2>/dev/null
! ls bin/pool-managerd-crates/router 2>/dev/null
! ls bin/pool-managerd-crates/model-eviction 2>/dev/null
```

---

**Cleanup Status**: ✅ Complete (crates and workspace)  
**Documentation Status**: ⏳ In Progress (many docs need updates)  
**Authoritative Reference**: `.docs/ARCHITECTURE_TRIO_CORRECTED.md`
