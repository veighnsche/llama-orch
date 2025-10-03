# Binary Specs Recreation Summary

**Date**: 2025-10-03 01:29  
**Action**: Recreated all binary specifications aligned with corrected architecture

---

## What Was Done

After you deleted all `.specs/` folders for the three binaries, I recreated comprehensive specifications aligned with the corrected trio-binary architecture.

---

## Specs Created

### ✅ Orchestratord Spec

**File**: `bin/orchestratord/.specs/00_orchestratord.md`  
**Requirements**: ORCH-1001 to ORCH-1172  
**Length**: ~350 lines

**Sections**:
1. Core Responsibilities (Intelligence Boundary)
2. Queue & Admission
3. Scheduling
4. Placement (Worker Selection)
5. Worker Startup Decisions
6. Eviction Decisions
7. Routing (Direct Worker Connection)
8. Streaming (SSE to Clients)
9. Catalog Management
10. Observability
11. HTTP API
12. Pool Manager Communication
13. Error Handling
14. Security
15. Configuration
16. Traceability
17. Refinement Opportunities

**Key Points**:
- ✅ Orchestrator is THE BRAIN (all intelligence)
- ✅ Makes ALL decisions (placement, scheduling, routing, eviction)
- ✅ Connects directly to workers (no proxy through pool manager)
- ✅ Manages catalog
- ✅ Streams SSE to clients

---

### ✅ Pool Managerd Spec

**File**: `bin/pool-managerd/.specs/00_pool-managerd.md`  
**Requirements**: POOL-2001 to POOL-2122  
**Length**: ~320 lines

**Sections**:
1. Core Responsibilities (State Reporter & Worker Factory)
2. GPU Inventory (VRAM tracking)
3. Worker Registry
4. Model Cache (RAM Staging)
5. Worker Lifecycle (Start/Stop)
6. Health Monitoring
7. Internal API
8. Observability
9. Configuration
10. Error Handling
11. Security
12. GPU Requirements
13. Multi-Node Support (Optional)
14. Traceability
15. Refinement Opportunities

**Key Points**:
- ✅ Pool manager is STATE REPORTER (no intelligence)
- ✅ Tracks GPU VRAM across all local GPUs
- ✅ Maintains worker registry
- ✅ Spawns/kills workers when commanded by orchestratord
- ✅ Pre-stages models in RAM for fast worker startup
- ✅ Reports state to orchestratord on demand

---

### ✅ Worker-orcd Spec

**File**: `bin/worker-orcd/.specs/00_worker-orcd.md`  
**Requirements**: WORK-3001 to WORK-3132  
**Length**: ~330 lines

**Sections**:
1. Core Responsibilities (Dumb Executor)
2. Startup & Initialization
3. VRAM Policy (VRAM-only enforcement)
4. Model Loading
5. HTTP API
6. SSE Streaming
7. Inference Execution
8. Resource Limits
9. Observability
10. Error Taxonomy
11. Shutdown
12. Configuration
13. Security
14. GPU Requirements
15. Traceability
16. Refinement Opportunities

**Key Points**:
- ✅ Worker is DUMB EXECUTOR (no intelligence)
- ✅ Loads ONE model for lifetime
- ✅ Enforces VRAM-only policy (no RAM fallback)
- ✅ Executes inference requests from orchestratord
- ✅ Streams SSE directly to orchestratord
- ✅ Calls back to pool manager when ready

---

## Additional Documentation

### ✅ Specs Index

**File**: `.docs/SPECS_INDEX.md`

Provides:
- Overview of all three specs
- Architecture reference
- Data flow examples
- Requirement ID ranges
- Cross-cutting concerns
- Testing requirements

---

## Architecture Alignment

All specs are aligned with:

**Authoritative Reference**: `.docs/ARCHITECTURE_TRIO_CORRECTED.md`

**Key Principles**:
1. **Orchestrator** = ALL intelligence (planning, scheduling, routing, placement, eviction)
2. **Pool Manager** = State reporter + worker factory (NO decisions)
3. **Worker** = Dumb executor (one model, simple execution)

---

## Requirement ID Scheme

| Binary | Range | Count | Example |
|--------|-------|-------|---------|
| Orchestratord | ORCH-1xxx | ~35 | ORCH-1001, ORCH-1050, ORCH-1120 |
| Pool Managerd | POOL-2xxx | ~30 | POOL-2001, POOL-2040, POOL-2090 |
| Worker-orcd | WORK-3xxx | ~30 | WORK-3001, WORK-3060, WORK-3120 |

**Total**: ~95 normative requirements across all three binaries

---

## Data Flow Summary

### Job Execution Flow
```
Client → Orchestrator (enqueue)
  ↓
Orchestrator (schedule)
  ↓
Orchestrator queries Pool Manager: "What workers do you have?"
  ↓
Pool Manager: "Here's GPU state + worker list"
  ↓
Orchestrator (placement): Select worker-abc
  ↓
Orchestrator → Worker-abc (direct): POST /execute
  ↓
Worker-abc → Orchestrator (direct): SSE stream
  ↓
Orchestrator → Client: SSE relay
```

**Key insight**: Pool manager is NOT in the data path!

---

## Key Differences from Old Model

| Aspect | OLD (Wrong) | NEW (Correct) |
|--------|-------------|---------------|
| **Intelligence** | Pool manager makes decisions | Orchestrator makes ALL decisions |
| **Routing** | Through pool manager | Direct to workers |
| **Worker Lifecycle** | Pool manager decides when to start | Orchestrator commands when to start |
| **Eviction** | Pool manager decides | Orchestrator decides, pool manager executes |
| **Model Management** | Worker loads multiple models | Worker loads ONE model for lifetime |
| **Placement** | Pool manager does placement | Orchestrator does placement |

---

## Verification

```bash
# Check specs exist
ls bin/orchestratord/.specs/00_orchestratord.md  ✅
ls bin/pool-managerd/.specs/00_pool-managerd.md  ✅
ls bin/worker-orcd/.specs/00_worker-orcd.md      ✅

# Check index
ls .docs/SPECS_INDEX.md                          ✅

# Check architecture reference
ls .docs/ARCHITECTURE_TRIO_CORRECTED.md          ✅
```

---

## Next Steps

### Immediate
1. ✅ Specs created
2. ⏳ Review specs for completeness
3. ⏳ Align existing code with specs
4. ⏳ Update other docs to reference these specs

### Short-Term
5. ⏳ Generate test catalog from specs
6. ⏳ Implement missing requirements
7. ⏳ Add BDD tests per spec requirements
8. ⏳ Update OpenAPI to match specs

### Medium-Term
9. ⏳ Full implementation of all three binaries
10. ⏳ Integration testing across binaries
11. ⏳ Performance testing
12. ⏳ Production hardening

---

## Related Documents

**Architecture**:
- `.docs/ARCHITECTURE_TRIO_CORRECTED.md` — Authoritative architecture
- `.docs/ARCHITECTURE_CLEANUP_LOG.md` — What changed
- `.docs/ARCHITECTURE_CLEANUP_SUMMARY.md` — Quick reference

**Specifications**:
- `bin/orchestratord/.specs/00_orchestratord.md`
- `bin/pool-managerd/.specs/00_pool-managerd.md`
- `bin/worker-orcd/.specs/00_worker-orcd.md`
- `.docs/SPECS_INDEX.md` — This index

**Parent Spec**:
- `.specs/00_llama-orch.md` — System-wide requirements

---

## Statistics

**Files Created**: 4
- 3 binary specs
- 1 index document

**Total Lines**: ~1,000 lines of normative requirements

**Requirement IDs**: ~95 normative requirements

**Time**: ~30 minutes

**Status**: ✅ **COMPLETE**

---

**Created by**: Cascade  
**Date**: 2025-10-03 01:29 UTC+02:00  
**Quality**: RFC-2119 conformant, aligned with architecture
