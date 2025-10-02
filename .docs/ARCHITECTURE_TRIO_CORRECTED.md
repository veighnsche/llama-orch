# Corrected Trio-Binary Architecture

**Date**: 2025-10-03  
**Status**: Authoritative  
**Supersedes**: All prior architecture documents with conflicting boundaries

---

## Executive Summary

The llama-orch system consists of three binaries with **clearly defined boundaries**:

1. **Orchestrator** = THE BRAIN (all intelligence)
2. **Pool Manager** = STATE REPORTER + WORKER FACTORY (no decisions)
3. **Worker** = DUMB EXECUTOR (one model, simple execution)

---

## Architectural Boundaries

### 1. Orchestrator (THE ONLY BRAIN)

**Binary**: `bin/orchestratord`  
**Runs on**: Control plane (no GPU required)

**Responsibilities**:
- ✅ **ALL planning**: Queue management, job prioritization
- ✅ **ALL scheduling**: Which worker executes which job
- ✅ **ALL placement**: Which GPU should get a new worker
- ✅ **ALL routing**: Route tasks to specific workers
- ✅ **Catalog management**: Model registry, verification, lifecycle
- ✅ **Client-facing API**: HTTP endpoints, SSE streaming
- ✅ **Eviction decisions**: When to unload models from VRAM
- ✅ **Load balancing**: Distribute work optimally

**Key Crates**:
- `orchestrator-core` — Queue, admission logic
- `placement` — Worker selection algorithms
- `streaming` — SSE to clients
- `task-cancellation` — Job cancellation
- `catalog-core` — Model catalog
- `backpressure` — 429 responses

**Does NOT**:
- ❌ Execute inference
- ❌ Manage GPU hardware
- ❌ Start/stop worker processes

---

### 2. Pool Manager (STATE REPORTER + WORKER FACTORY)

**Binary**: `bin/pool-managerd`  
**Runs on**: GPU machine (one per physical host)

**Responsibilities**:
- ✅ **GPU inventory**: Track VRAM capacity across all local GPUs
- ✅ **Worker registry**: Track running workers (id, model, VRAM, URI, status)
- ✅ **Worker lifecycle**: Start/stop workers when orchestrator commands
- ✅ **Model cache**: Pre-stage models in RAM for fast worker startup
- ✅ **Health monitoring**: Check worker heartbeats, report failures
- ✅ **State reporting**: Provide snapshots to orchestrator

**Key Crates**:
- `gpu-inventory` — Track VRAM per GPU
- `worker-registry` — Track running workers (RENAMED from pool-registry)
- `model-cache` — Download models, stage in RAM
- `lifecycle` — Spawn/kill worker processes
- `health-monitor` — Worker health checks
- `api` — Internal API for orchestrator

**Does NOT**:
- ❌ Make placement decisions (orchestrator decides)
- ❌ Route tasks (orchestrator routes directly to workers)
- ❌ Schedule jobs (orchestrator schedules)
- ❌ Decide which models to evict (orchestrator decides)
- ❌ Stream SSE to clients (workers stream directly)

**Data Provided to Orchestrator**:
```json
{
  "gpus": [
    {"id": 0, "total_vram": 24GB, "available_vram": 8GB},
    {"id": 1, "total_vram": 24GB, "available_vram": 20GB}
  ],
  "workers": [
    {
      "id": "worker-abc",
      "model": "llama-7b",
      "gpu": 0,
      "vram_used": 16GB,
      "uri": "http://localhost:8001",
      "status": "ready"
    }
  ]
}
```

---

### 3. Worker (DUMB EXECUTOR)

**Binary**: `bin/worker-orcd`  
**Runs on**: GPU machine (multiple per GPU possible)

**Responsibilities**:
- ✅ **Load ONE model**: At startup, load assigned model to VRAM
- ✅ **Execute inference**: Run jobs when orchestrator connects
- ✅ **VRAM enforcement**: Ensure model stays in VRAM (no RAM fallback)
- ✅ **Report status**: Tell pool manager VRAM usage, readiness
- ✅ **Stream results**: SSE directly to orchestrator

**Key Crates**:
- `vram-policy` — VRAM-only enforcement
- `model-loader` — Load model to VRAM
- `api` — HTTP endpoints for orchestrator
- `capability-matcher` — Report capabilities

**Does NOT**:
- ❌ Load multiple models (tied to ONE model for lifetime)
- ❌ Make scheduling decisions (orchestrator schedules)
- ❌ Manage other workers
- ❌ Make placement decisions

**Lifecycle**:
1. Pool manager spawns: `worker-orcd --model llama-7b.gguf --gpu-device 0`
2. Worker loads model to VRAM
3. Worker calls back to pool manager: `POST /v2/internal/workers/ready`
4. Pool manager updates registry
5. Orchestrator queries pool manager, sees worker
6. Orchestrator connects DIRECTLY to worker URI for inference
7. Worker streams SSE directly back to orchestrator

---

## Deleted Concepts (Wrong Layer)

### ❌ Removed from Pool Manager
- `model-eviction/` — Orchestrator decides eviction
- `router/` — Orchestrator does routing

### ❌ Removed from Worker
- `scheduler/` — Orchestrator does scheduling
- `vram-residency/` — Wrong abstraction (split into gpu-inventory + vram-policy)

### ❌ Removed Concepts
- "Pools" as first-class entities — Workers exist, not pools
- "Adapters" — Orchestrator connects directly to workers
- "Engines" — Workers ARE the engine

---

## Data Flow Examples

### Example 1: Start New Worker

```
1. Orchestrator: "I need llama-13b (26GB)"
   ↓
2. Orchestrator queries Pool Manager:
   GET /v2/state
   Response: GPU 1 has 20GB free
   ↓
3. Orchestrator commands:
   POST /v2/workers/start
   { "model": "llama-13b", "gpu": 1 }
   ↓
4. Pool Manager:
   - Checks model-cache (stage to RAM if needed)
   - Spawns: worker-orcd --model llama-13b.gguf --gpu-device 1
   ↓
5. Worker:
   - Loads model to VRAM (26GB)
   - Calls: POST /v2/internal/workers/ready
   ↓
6. Pool Manager:
   - Updates gpu-inventory: GPU 1 now 46GB used
   - Updates worker-registry: worker-xyz ready
   ↓
7. Orchestrator queries again, sees new worker available
```

### Example 2: Execute Inference

```
1. Client → Orchestrator:
   POST /v2/tasks { "model": "llama-7b", "prompt": "..." }
   ↓
2. Orchestrator (placement):
   - Queries pool manager state
   - Selects worker-abc (least loaded)
   ↓
3. Orchestrator → Worker (DIRECT):
   POST http://localhost:8001/execute { "prompt": "..." }
   ↓
4. Worker → Orchestrator (DIRECT):
   SSE stream: token → token → end
   ↓
5. Orchestrator → Client:
   SSE stream: token → token → end
```

**Note**: Pool manager is NOT in the data path!

### Example 3: Evict Model

```
1. Orchestrator decision: "VRAM is full, evict llama-7b"
   ↓
2. Orchestrator → Pool Manager:
   POST /v2/workers/stop { "worker_id": "worker-abc" }
   ↓
3. Pool Manager:
   - Drains worker (finish active jobs)
   - Kills worker process
   - Updates gpu-inventory: GPU 0 freed 16GB
   - Updates worker-registry: remove worker-abc
```

---

## Terminology Dictionary

| OLD (Wrong) | NEW (Correct) | Meaning |
|-------------|---------------|---------|
| Pool | Worker | A running inference process |
| Engine | Worker | Same |
| Pool Manager provisions engines | Pool Manager spawns workers | Process creation |
| Pool has lifecycle | Worker has lifecycle | States belong to workers |
| Adapters | Direct connection | No abstraction layer |
| Pool Manager routes | Orchestrator routes | Routing is intelligence |
| Pool Manager schedules | Orchestrator schedules | Scheduling is intelligence |
| Pool Manager decides eviction | Orchestrator decides eviction | Eviction is intelligence |

---

## Crate Mapping

### Pool Manager Crates
| Crate | Purpose |
|-------|---------|
| `gpu-inventory` | Track VRAM per GPU |
| `worker-registry` | Track running workers |
| `model-cache` | Download/stage models in RAM |
| `lifecycle` | Start/stop workers |
| `health-monitor` | Worker health checks |
| `api` | Internal API |

### Worker Crates
| Crate | Purpose |
|-------|---------|
| `vram-policy` | VRAM-only enforcement |
| `model-loader` | Load model to VRAM |
| `api` | HTTP endpoints |
| `capability-matcher` | Report capabilities |

### Orchestrator Crates
| Crate | Purpose |
|-------|---------|
| `orchestrator-core` | Queue, admission |
| `placement` | Worker selection |
| `streaming` | SSE to clients |
| `task-cancellation` | Job cancellation |
| `catalog-core` | Model catalog |

---

## Migration from Old Docs

**All documentation referencing:**
- "Pool Manager makes decisions" → INCORRECT
- "Pool Manager routes tasks" → INCORRECT
- "Pool lifecycle states" → Should be "Worker lifecycle states"
- "Adapters" → Should be "direct worker connection"
- "Engine provisioning" → Should be "worker spawning"

**Refer to this document as authoritative.**

---

## Status

- **Date**: 2025-10-03
- **Version**: 1.0
- **Supersedes**: 
  - `.docs/ARCHITECTURE_CHANGE_PLAN.md` (boundaries incorrect)
  - `.docs/WORKER_READINESS_CALLBACK_DESIGN.md` (mostly correct)
  - All READMEs with conflicting boundaries

---

**End of Authoritative Architecture**
