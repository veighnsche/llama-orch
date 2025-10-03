# Binary Specifications Index

**Date**: 2025-10-03  
**Status**: Active

---

## Overview

This document indexes all binary specifications aligned with the corrected trio-binary architecture.

---

## Architecture Reference

**Authoritative**: `.docs/ARCHITECTURE_TRIO_CORRECTED.md`

**Summary**:
- **Orchestrator** = THE BRAIN (all intelligence)
- **Pool Manager** = STATE REPORTER + WORKER FACTORY (no decisions)
- **Worker** = DUMB EXECUTOR (one model, simple execution)

---

## Binary Specifications

### 1. Orchestratord (THE BRAIN)

**Spec**: `bin/orchestratord/.specs/00_orchestratord.md`  
**Requirement Range**: ORCH-1xxx

**Responsibilities**:
- Queue management & admission
- Job scheduling
- Worker placement
- Routing (direct worker connection)
- SSE streaming to clients
- Catalog management
- Eviction decisions
- All intelligence

**Key Crates**:
- `orchestrator-core` — Queue, admission
- `placement` — Worker selection
- `streaming` — SSE relay
- `task-cancellation` — Job cancellation
- `catalog-core` — Model catalog
- `backpressure` — 429 responses

**API Endpoints**:
- `POST /v2/tasks` — Submit job
- `GET /v2/tasks/{id}/events` — SSE stream
- `POST /v2/tasks/{id}/cancel` — Cancel job
- `GET /v2/state` — Query pool managers
- `POST /v2/workers/start` — Command worker start
- `POST /v2/workers/stop` — Command worker stop
- `POST /v2/catalog/models` — Catalog operations

---

### 2. Pool Managerd (STATE REPORTER)

**Spec**: `bin/pool-managerd/.specs/00_pool-managerd.md`  
**Requirement Range**: POOL-2xxx

**Responsibilities**:
- GPU VRAM inventory tracking
- Worker registry maintenance
- Worker process lifecycle (spawn/kill)
- Model cache (download, RAM staging)
- Health monitoring
- State reporting to orchestratord
- NO decisions

**Key Crates**:
- `gpu-inventory` — VRAM tracking per GPU
- `worker-registry` — Running workers (rename from pool-registry)
- `model-cache` — Download/stage models
- `lifecycle` — Process management
- `health-monitor` — Worker health checks
- `api` — Internal API

**API Endpoints**:
- `GET /v2/state` — Report GPU & worker state
- `POST /v2/workers/start` — Start worker (commanded by orchestratord)
- `POST /v2/workers/stop` — Stop worker (commanded by orchestratord)
- `POST /v2/internal/workers/ready` — Worker callback

---

### 3. Worker-orcd (DUMB EXECUTOR)

**Spec**: `bin/worker-orcd/.specs/00_worker-orcd.md`  
**Requirement Range**: WORK-3xxx

**Responsibilities**:
- Load ONE model to VRAM at startup
- Execute inference requests
- VRAM-only enforcement
- SSE streaming to orchestratord
- Health reporting
- NO decisions

**Key Crates**:
- `vram-policy` — VRAM-only enforcement
- `model-loader` — Load model to VRAM
- `api` — HTTP endpoints
- `capability-matcher` — Report capabilities

**API Endpoints**:
- `POST /execute` — Execute inference (called by orchestratord)
- `GET /health` — Health check
- `GET /metrics` — Prometheus metrics
- `POST /shutdown` — Graceful shutdown

---

## Data Flow Examples

### Example 1: Client Submits Job

```
1. Client → Orchestrator:
   POST /v2/tasks { "model": "llama-7b", "prompt": "..." }

2. Orchestrator (admission):
   - Validate request
   - Enqueue to priority queue

3. Orchestrator (scheduling):
   - Pop job from queue
   - Query pool managers: GET /v2/state

4. Orchestrator (placement):
   - Select worker-abc (least loaded)
   - Or decide to start new worker

5. Orchestrator → Worker (direct):
   POST http://localhost:8001/execute { "prompt": "..." }

6. Worker → Orchestrator (SSE):
   started → token → token → ... → end

7. Orchestrator → Client (SSE):
   Relay stream + add metadata
```

### Example 2: Start New Worker

```
1. Orchestrator: "Need llama-13b, no worker available"

2. Orchestrator → Pool Manager:
   POST /v2/workers/start { "model": "llama-13b", "gpu_id": 1 }

3. Pool Manager:
   - Check GPU 1 has 20GB free (via gpu-inventory)
   - Download model if needed (via model-cache)
   - Spawn: worker-orcd --model llama-13b.gguf --gpu-device 1

4. Worker:
   - Load model to VRAM (26GB)
   - POST http://localhost:9200/v2/internal/workers/ready

5. Pool Manager:
   - Update worker-registry: worker-xyz ready
   - Update gpu-inventory: GPU 1 allocated += 26GB

6. Orchestrator:
   - Query /v2/state again
   - See new worker available
   - Route job to worker-xyz
```

### Example 3: Evict Worker

```
1. Orchestrator: "VRAM full, need space, evict worker-abc"

2. Orchestrator → Pool Manager:
   POST /v2/workers/stop { "worker_id": "worker-abc" }

3. Pool Manager:
   - Send SIGTERM to worker-abc
   - Wait 30s for graceful shutdown
   - Kill process if needed
   - Update worker-registry: remove worker-abc
   - Update gpu-inventory: GPU 0 allocated -= 16GB

4. Orchestrator:
   - Query /v2/state
   - See GPU 0 now has 16GB free
   - Can start new worker if needed
```

---

## Requirement ID Ranges

| Binary | Range | Example |
|--------|-------|---------|
| Orchestratord | ORCH-1xxx | ORCH-1001, ORCH-1050 |
| Pool Managerd | POOL-2xxx | POOL-2001, POOL-2040 |
| Worker-orcd | WORK-3xxx | WORK-3001, WORK-3060 |

---

## Cross-Cutting Concerns

### Shared Specs
- `.specs/00_llama-orch.md` — System-wide requirements
- `.specs/00_proof-bundle.md` — Testing standard
- `.specs/11_min_auth_hooks.md` — Authentication (optional)
- `.specs/metrics/otel-prom.md` — Observability

### Shared Crates
- `auth-min` — Bearer token authentication
- `narration-core` — Human-readable logging
- `audit-logging` — Security audit trail
- `secrets-management` — Secret handling
- `input-validation` — Input validation
- `gpu-info` — GPU discovery

---

## Testing

Each binary MUST have:
- **Unit tests**: `tests/*.rs`
- **BDD tests**: `bdd/tests/features/*.feature`
- **Integration tests**: Test against real/mock dependencies

Tests MUST emit proof bundles to `.proof_bundle/` per `.specs/00_proof-bundle.md`.

---

## Status

- ✅ Orchestratord spec: Complete (ORCH-1xxx)
- ✅ Pool Managerd spec: Complete (POOL-2xxx)
- ✅ Worker-orcd spec: Complete (WORK-3xxx)
- ⏳ Implementation: In progress

---

## Related Documents

- **Architecture**: `.docs/ARCHITECTURE_TRIO_CORRECTED.md`
- **Cleanup Log**: `.docs/ARCHITECTURE_CLEANUP_LOG.md`
- **Parent Spec**: `.specs/00_llama-orch.md`
- **Test Catalog**: `.docs/spec-derived-test-catalog.md`

---

**Maintained by**: llama-orch team  
**Last Updated**: 2025-10-03
