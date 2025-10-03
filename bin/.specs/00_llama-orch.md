# Llama-Orch SPEC — System Architecture (SYS-0xxx)

**Status**: Draft  
**Version**: 0.1.0  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Executive Summary

### Purpose

Llama-Orch is a **deterministic, VRAM-only, multi-node GPU orchestration system** for large language model inference. It provides guaranteed reproducibility, EU-native compliance, and enterprise-grade orchestration across distributed GPU resources.

**Core Value Propositions:**
1. **Determinism Guarantee**: Same seed → Same output (every time, provably)
2. **VRAM-Only Policy**: Model fully resident in GPU VRAM (no RAM fallback)
3. **Multi-Node Orchestration**: Distribute models across GPU clusters
4. **EU Compliance**: GDPR-native, EU-only data residency
5. **Marketplace Ready**: Enable GPU provider ecosystem

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATORD (The Brain - All Intelligent Decisions)           │
│                                                                   │
│ • Admission, Queue, Scheduling, Worker Selection                 │
│ • Eviction, Retry, Timeout, Cancellation Policies               │
│ • Client-facing API (Platform & Agentic)                        │
│ • SSE Streaming Relay                                           │
└────────────────────┬────────────────────────────────────────────┘
                     │ Commands (HTTP)
                     ├──────────────┬──────────────┬──────────────
                     ↓              ↓              ↓
┌────────────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ POOL-MANAGERD         │ │ POOL-MANAGERD   │ │ POOL-MANAGERD   │
│ (Control Plane)       │ │ (Control Plane) │ │ (Control Plane) │
│                       │ │                 │ │                 │
│ • GPU Inventory       │ │                 │ │                 │
│ • Capability Match    │ │                 │ │                 │
│ • Model Cache         │ │                 │ │                 │
│ • Worker Lifecycle    │ │                 │ │                 │
│ • Operational Cleanup │ │                 │ │                 │
│ • Report State Up     │ │                 │ │                 │
└─────┬──────────────────┘ └─────┬───────────┘ └─────┬───────────┘
      │ Spawns                   │                   │
      ├──────┬──────             ├──────             ├──────
      ↓      ↓      ↓            ↓      ↓            ↓      ↓
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│WORKER│ │WORKER│ │WORKER│ │WORKER│ │WORKER│ │WORKER│ │WORKER│
│      │ │      │ │      │ │      │ │      │ │      │ │      │
│GPU 0 │ │GPU 1 │ │GPU 2 │ │GPU 0 │ │GPU 1 │ │GPU 0 │ │GPU 1 │
│VRAM  │ │VRAM  │ │VRAM  │ │VRAM  │ │VRAM  │ │VRAM  │ │VRAM  │
│Alloc │ │Alloc │ │Alloc │ │Alloc │ │Alloc │ │Alloc │ │Alloc │
└──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
  Pool 1 GPUs           Pool 2 GPUs           Pool 3 GPUs
```

**Decision Hierarchy:**
```
Orchestratord (Brain) → Makes ALL intelligent decisions
    ↓ commands
Pool Manager (Levers) → Executes commands, reports state
    ↓ spawns
Worker (Executor) → Loads model, executes inference
```

---

## 1. System-Level Requirements

### [SYS-0001] Intelligence Boundary
The system MUST centralize ALL intelligent decisions in orchestratord. Pool managers and workers MUST be dumb executors.

**Intelligent decisions** (orchestratord only):
- Admission (accept/reject requests)
- Scheduling (which job next)
- Worker selection (where to run)
- Eviction (which worker to stop)
- Retry (whether to retry failures)
- Timeout enforcement
- Cancellation propagation

**Dumb execution** (pool managers and workers):
- Execute commands received
- Report facts and state
- Operational cleanup (no policy decisions)

### [SYS-0002] VRAM-Only Policy
The system MUST enforce VRAM-only policy: model weights, KV cache, activations, and intermediate tensors MUST reside entirely in GPU VRAM.

**Prohibited:**
- ❌ RAM fallback
- ❌ Unified memory (CUDA UMA)
- ❌ Zero-copy mode
- ❌ CPU inference fallback
- ❌ Disk swapping

### [SYS-0003] Determinism Guarantee
The system MUST guarantee deterministic inference: same model + same seed + same prompt → same output.

**Requirements:**
- Sealed VRAM shards (worker-orcd)
- Pinned engine versions
- Deterministic sampling
- No non-deterministic operations

### [SYS-0004] Multi-Node Support
The system MUST support distributed deployment across multiple GPU nodes.

**Deployment modes:**
- Single node, single GPU (M0)
- Single node, multi GPU (M1)
- Multi node, multi GPU (M2+)

### [SYS-0005] Process Isolation
Workers MUST run in separate processes from pool managers.

**Why**: CUDA allocations are per-process. Workers need self-contained VRAM ownership within their CUDA context.

---

## 2. Component Architecture

### [SYS-0010] Component Separation
The system MUST implement three separate binaries: orchestratord, pool-managerd, and worker-orcd. Each component MUST communicate via HTTP APIs only.

### [SYS-0011] No Direct Worker Communication
Clients MUST NOT communicate directly with workers. All client requests MUST go through orchestratord.

---

### 2.1 Orchestratord (The Brain)

**Binary**: `bin/orchestratord/`  
**Port**: 8080 (default)  
**Role**: Centralized intelligence for all decisions

### [SYS-0020] Orchestrator Intelligence
Orchestratord MUST implement ALL intelligent decision-making:
- MUST validate and admit requests before enqueue
- MUST manage bounded FIFO queue with Interactive/Batch priorities
- MUST select next job and target worker (combined scheduling decision)
- MUST command pool managers to start/stop workers
- MUST route inference requests to selected workers
- MUST relay SSE streams from workers to clients
- MUST enforce timeout limits on jobs
- MUST propagate cancellation requests to workers

### [SYS-0021] Orchestrator Statelessness
Orchestratord SHOULD be stateless (all state derived from pool manager queries). This enables orchestrator restart without state loss.

**Responsibilities:**
- Accept client inference requests
- Validate and admit requests (admission)
- Enqueue jobs (bounded FIFO with priorities)
- Schedule jobs (select which job + which worker)
- Command pool managers (start/stop workers)
- Route requests to workers
- Relay SSE streams to clients
- Enforce timeouts and cancellation
- Track metrics and observability

**Crates:**
- `scheduling` — Admission, queue, job tracking, worker selection, eviction
- `platform-api` — Marketplace federation facade
- `agentic-api` — Standard/home orchestrator API
- `pool-registry` — Track pool managers and state
- `streaming` — SSE relay with metadata
- `task-cancellation` — Cancellation propagation
- `job-timeout` — Timeout enforcement
- `backpressure` — Queue backpressure handling

**Specs**: `bin/orchestratord/.specs/00_orchestratord.md`

**Requirements**:
- Orchestratord SHOULD be stateless; on restart it MUST reconstruct state from pool-managerd heartbeats and registries without manual intervention.
- All intelligent decisions (admission, scheduling, eviction, retry, timeout, cancellation) MUST occur in orchestratord and MUST NOT be delegated.
- SSE relay to clients MUST preserve worker event order and MUST add orchestrator metadata without altering payload semantics.

---

### 2.2 Pool-Managerd (Control Plane with All Levers)

**Binary**: `bin/pool-managerd/`  
**Port**: 9200 (default)  
**Role**: Local agent on GPU nodes, executes orchestrator commands

### [SYS-0030] Pool Manager Execution
Pool-managerd MUST execute orchestrator commands without making policy decisions:
- MUST query GPU state via NVML (read-only)
- MUST download and cache models as commanded
- MUST validate model-GPU compatibility before worker start (preflight)
- MUST spawn worker processes as commanded
- MUST update VRAM accounting when workers start/stop
- MUST send periodic heartbeats to orchestratord (default 15s interval)
- MUST perform operational cleanup on worker failures (no retry decisions)

### [SYS-0031] Pool Manager State Reporting
Pool-managerd MUST report facts, not decisions:
- MUST report GPU VRAM state (total, available, allocated)
- MUST report worker state (running, ready, failed)
- MUST report failures with context (exit code, error message)
- MUST NOT decide to retry or failover

**Responsibilities:**
- Track GPU state system-wide (NVML)
- Download and stage models
- Validate model compatibility (preflight)
- Spawn and monitor worker processes
- Update VRAM accounting
- Report state to orchestratord (heartbeat)
- Operational cleanup (error-ops)

**Crates:**
- `gpu-inventory` — NVML FFI for GPU/VRAM tracking
- `capability-matcher` — Preflight model compatibility validation
- `model-cache` — Model storage
- `model-provisioner` — Model download orchestration
- `model-catalog` — Model metadata registry
- `worker-lifecycle` — Worker process spawning and monitoring
- `control-api` — HTTP API for orchestrator commands
- `error-ops` — Operational cleanup (not policy)
- `pool-registration-client` — Register with orchestrator/platform

**FFI Boundary**: Uses **NVML** (read-only GPU queries), NOT CUDA

**Requirements**:
- Pool-managerd MUST perform preflight validation (GPU VRAM headroom, capability match) before spawning workers.
- Heartbeats MUST include GPU VRAM totals/allocated and worker states; missed heartbeats beyond timeout MUST mark the pool unavailable.
- Pool-managerd MUST NOT perform placement or retry decisions; it MUST execute orchestrator commands and report facts only.
- NVML MUST be used for GPU queries; CUDA allocations MUST NOT be performed by pool-managerd.

**Specs**: `bin/pool-managerd/.specs/00_pool-managerd.md`

---

### 2.3 Worker-Orcd (Self-Contained Executor)

**Binary**: `bin/worker-orcd/`  
**Port**: Dynamic (assigned by pool manager)  
**Role**: Execute inference on single GPU (or multi-GPU for large models)

### [SYS-0040] Worker Self-Containment
Worker-orcd MUST operate as a self-contained process:
- MUST load exactly ONE model at startup (from disk to VRAM)
- MUST own VRAM allocation within its CUDA context
- MUST allocate all model resources in VRAM only (no RAM fallback)
- MUST execute inference requests received via HTTP
- MUST stream results via SSE (token-by-token)
- MUST monitor VRAM residency (self-health checks)
- MUST report actual VRAM usage to pool manager on ready

### [SYS-0041] Worker Isolation
Each worker MUST run in a separate OS process. Workers MUST NOT share VRAM or CUDA contexts.

**Responsibilities:**
- Load ONE model from disk → VRAM (at startup)
- Own VRAM allocation within process (CUDA context)
- Execute inference requests
- Stream token-by-token results (SSE)
- Monitor own health (VRAM residency)
- Report actual VRAM usage to pool manager (callback)

**Crates:**
- **Integrated binary** — All functionality integrated into single binary due to CUDA context requirements
- **CUDA modules** — `src/cuda/` (context, model, inference, health) - process-level operations
- **HTTP handlers** — `src/http/` (execute, health endpoints)
- **Lifecycle** — `src/startup.rs` (initialization and callbacks)

**Binary structure:**
- `src/cuda/` — CUDA FFI layer (memory allocation, kernels, enforcement) - process-level
- `src/main.rs` — Worker entry point and orchestration

**Specs**: `bin/worker-orcd/.specs/00_worker-orcd.md`

**Requirements**:
- Worker-orcd MUST load exactly one model at startup and MUST own all VRAM allocations within its process CUDA context.
- All model weights, KV cache, and activations MUST reside in VRAM (no RAM/unified memory fallback).
- HTTP server MUST start before the ready callback is sent; SSE streams MUST emit `started` → `token*` → `metrics*` → terminal (`end`/`error`).
- Ready callback to pool-managerd MUST include `worker_id`, `model_ref`, `vram_bytes`, and `uri`.

---
## 3. Data Flow & Interactions

### 3.1 Job Submission Flow

```
1. Client → Orchestrator
   POST /v2/tasks
   { model, prompt, max_tokens, seed }

2. Orchestrator: Admission
   - Validate model exists
   - Check context length
   - Check token budget

3. Orchestrator: Enqueue
   - Add to queue (Interactive or Batch priority)
   - Return 202 Accepted + job_id

4. Orchestrator: Schedule
   - Dequeue job (Interactive first, then Batch)
   - Query pool managers for state
   - Select best worker (least-loaded, most-vram-free, round-robin)

5. Orchestrator: Dispatch
   - POST {worker_uri}/execute
   - Establish SSE stream

6. Worker: Execute
   - Load prompt
   - Run inference
   - Stream tokens via SSE

7. Orchestrator: Relay
   - Relay SSE events to client
   - Add orchestrator metadata

8. Client: Receive
   - Consume SSE stream
   - Get tokens in real-time
```

**Requirements**:
- Admission MUST validate model presence, context length, and token budgets before enqueue.
- Queueing MUST respect priority classes; rejection behavior MUST follow configured policy (e.g., 429 for reject).
- Scheduling MUST select a worker based on current pool states; stale states SHOULD be refreshed if older than the heartbeat interval.
- SSE relays MUST preserve event ordering and MUST propagate correlation IDs.

### 3.2 Worker Startup Flow

```
1. Orchestrator decides: "Need worker for model X on GPU 0"

2. Orchestrator → Pool Manager
   POST /v2/workers/start
   { model_ref: "hf:author/repo@rev::file=models/llama-7b.Q4_K_M.gguf", gpu_id: 0 }

3. Pool Manager: Preflight Validation
   - gpu-inventory: "Does GPU 0 have 16GB free?" → Yes
   - capability-matcher: "Is model compatible?" → Yes

4. Pool Manager: Spawn Worker
   worker-orcd \
     --worker-id worker-abc \
     --model /models/llama-7b.gguf \
     --gpu-device 0 \
     --port 8001 \
     --callback-url http://pool:9200/v2/internal/workers/ready

5. Worker: Initialize
   - cuda: Enforce VRAM-only, allocate VRAM
   - model-lifecycle: Load model to VRAM
   - inference-api: Start HTTP server
   - health-monitor: Start self-monitoring

6. Worker → Pool Manager (callback)
   POST /v2/internal/workers/ready
   {
     worker_id: "worker-abc",
     model_ref: "llama-7b",
     vram_bytes: 17000000000,
     uri: "http://localhost:8001"
   }

7. Pool Manager: Update State
   - gpu-inventory: Update allocated_vram
   - worker-lifecycle: Mark worker as ready

8. Pool Manager → Orchestrator (heartbeat)
   POST /v2/pools/{id}/heartbeat
   { gpus: [...], workers: [...] }

9. Orchestrator: Worker Available
   - pool-registry: Update pool state
   - scheduling: Worker now available for jobs
```

**Requirements**:
- Preflight in pool-managerd MUST validate GPU free VRAM and model compatibility before spawning the worker.
- Worker-orcd MUST start HTTP server before issuing ready callback (server-first), and MUST include `model_ref` and `vram_bytes`.
- Pool-managerd MUST update VRAM accounting on ready and MUST mark worker ready atomically with state update.
- Orchestratord SHOULD avoid scheduling until a worker is marked ready via heartbeat; optimistic dispatch MAY be used only if configured.

### 3.3 Worker Failure Flow

```
1. Worker crashes (process exits)

2. Pool Manager detects (process monitoring)

3. Pool Manager: Operational Cleanup (error-ops)
   - Remove from worker registry
   - Release VRAM accounting (gpu-inventory)
   - Kill zombie processes
   - Close file handles

4. Pool Manager → Orchestrator
   POST /v2/internal/workers/failed
   {
     worker_id: "worker-abc",
     exit_code: -11,
     vram_released: 17000000000
   }

5. Orchestrator: Handle Failure
   - pool-registry: Mark worker offline
   - scheduling: Decide whether to retry
     • If retry policy allows → command new worker start
     • If not → fail pending jobs
```

**Requirements**:
- Pool-managerd MUST perform operational cleanup on crash (remove worker, release VRAM accounting, kill zombies) and MUST report failure with exit code and context.
- Orchestratord MUST mark the worker offline and MUST decide retry vs. fail per configured policy; retries MUST apply backoff and MUST not violate idempotency.
- If the failed worker held a scheduled or running job, orchestrator SHOULD requeue or reassign only if policy permits and inputs are available; otherwise it MUST fail the job with a stable error code.
- Metrics and logs SHOULD record failure reason, retry attempt, and outcome; proof bundles MAY capture a failure timeline for reproduction.

---

## 4. Key Architectural Principles

### [SYS-0050] Smart vs Dumb Architecture
The system MUST enforce a strict smart/dumb boundary:

**Smart components** (orchestratord only):
- MUST make ALL policy decisions (admission, scheduling, eviction, retry, timeout)
- MUST use configured policies (no hardcoded decisions)
- MUST decide actions based on aggregated state

**Dumb components** (pool manager, worker):
- MUST execute commands received without interpretation
- MUST report facts and state without filtering
- MUST NOT make policy decisions
- MUST perform operational cleanup only (not recovery decisions)

### [SYS-0051] FFI Boundaries
The system MUST enforce strict FFI boundaries:

**Pool manager** (NVML only):
- MUST use NVML for read-only GPU queries
- MUST query system-wide state (all GPUs)
- MUST NOT allocate VRAM or use CUDA
- MUST NOT perform compute operations

**Worker** (CUDA only):
- MUST use CUDA Runtime API for VRAM allocation
- MUST allocate VRAM within its process CUDA context
- MUST use CUDA for compute operations
- MUST own VRAM lifecycle (allocate → use → free)

**Rationale**: Pool manager monitors system-wide GPU state. Worker manages per-process VRAM within its CUDA context. These are orthogonal concerns with different FFI layers.

### [SYS-0052] Process Isolation
Workers MUST run in separate processes:

**Requirements**:
- Each worker MUST have its own OS process
- Each worker MUST have its own CUDA context
- Workers MUST NOT share VRAM pointers
- Worker MUST own complete VRAM lifecycle (allocate → use → free)

**Rationale**: CUDA VRAM allocations are per-process. Workers need isolated VRAM ownership.

**Testing benefit**: Enables standalone worker testing (`worker-orcd --model X --gpu 0`).

**Communication**: Components MUST communicate via HTTP APIs only.

### [SYS-0053] State Propagation
The system MUST implement unidirectional state flow:

**Upward state flow** (MUST report facts):
```
Worker → Pool Manager → Orchestrator
  (VRAM usage)  (GPU state)  (Cluster state)
```
- Worker MUST report VRAM usage to pool manager
- Pool manager MUST aggregate GPU state and report to orchestrator
- Orchestrator MUST query pool managers for scheduling decisions

**Downward command flow** (MUST execute commands):
```
Orchestrator → Pool Manager → Worker
  (decisions)   (execution)   (inference)
```
- Orchestrator MUST send commands (start worker, dispatch job)
- Pool manager MUST execute commands (spawn worker)
- Worker MUST execute inference requests

---

## 5. Deployment Modes

### 5.1 Home Mode (M0)

**Single node, single GPU**

```
[Orchestrator] (localhost:8080)
      ↓
[Pool Manager] (localhost:9200)
      ↓
[Worker-1] GPU 0 (localhost:8001)
```

**Use case**: Development, home lab, single user

**Specs**: M0 milestone requirements

**Requirements**:
- Orchestratord and pool-managerd MUST bind to loopback by default and SHOULD require explicit config to bind non-loopback.
- No authentication is REQUIRED for localhost; if exposed beyond localhost, bearer auth SHOULD be enabled (see §8.1).
- A single worker per GPU MUST be enforced (batch=1) and VRAM-only policy MUST be enabled.
- Configuration SHOULD be minimal; sensible defaults MUST be provided for ports and timeouts.

---

### 5.2 Multi-GPU Mode (M1)

**Single node, multiple GPUs**

```
[Orchestrator] (localhost:8080)
      ↓
[Pool Manager] (localhost:9200)
      ├─→ [Worker-1] GPU 0 (localhost:8001)
      ├─→ [Worker-2] GPU 1 (localhost:8002)
      └─→ [Worker-3] GPU 2 (localhost:8003)
```

**Use case**: Single powerful machine with 2-4 GPUs

**Features**:
- Tensor parallelism (split large models across GPUs)
- Multiple models loaded simultaneously

**Requirements**:
- Orchestratord MUST remain single-instance and stateless; restarts MUST NOT corrupt in-flight state beyond reconnect semantics.
- Pool-managerd MUST manage multiple workers mapped 1:1 to GPUs; tensor-parallel workers MUST coordinate within node only (no cross-node for M1).
- VRAM accounting MUST handle per-GPU allocation and reserved headroom; preflight MUST fail when insufficient free VRAM.
- Cancellation and timeouts MUST apply per-job; shared GPU allocations across tensor-parallel shards MUST be released on job termination.

---

### 5.3 Multi-Node Mode (M2+)

**Multiple nodes, multiple GPUs each**

```
[Orchestrator] (orchestrator.local:8080)
      ├─→ [Pool Manager 1] (node1:9200)
      │        ├─→ [Worker-1] GPU 0
      │        └─→ [Worker-2] GPU 1
      ├─→ [Pool Manager 2] (node2:9200)
      │        ├─→ [Worker-3] GPU 0
      │        └─→ [Worker-4] GPU 1
      └─→ [Pool Manager 3] (node3:9200)
               ├─→ [Worker-5] GPU 0
               └─→ [Worker-6] GPU 1
```

**Use case**: Enterprise, data center, GPU cluster

**Features**:
- Cluster-wide orchestration
- Load balancing across nodes
- High availability

**Requirements**:
- Orchestratord MUST manage multiple pool-managerd across nodes and MUST make all placement decisions centrally (no nested orchestrators).
- Heartbeat aggregation MUST tolerate partial failures; pools missing heartbeats beyond timeout MUST be excluded from scheduling.
- Network communication between nodes SHOULD use TLS or a trusted network; service discovery MAY be static or registry-based.
- Scheduling SHOULD prefer locality and capacity signals (e.g., free VRAM); policy MUST be observable via metrics and logs.

---

### 5.4 Platform Mode (Marketplace)

**Federation across provider orchestrators**

```
[Platform Orchestrator] (api.yourplatform.com:443)
      ├─→ [Provider A Orchestrator] (provider-a.internal:8080)
      │        └─→ Provider A's pools/workers
      ├─→ [Provider B Orchestrator] (provider-b.internal:8080)
      │        └─→ Provider B's pools/workers
      └─→ [Provider C Orchestrator] (provider-c.internal:8080)
               └─→ Provider C's pools/workers
```

**Use case**: GPU marketplace, provider ecosystem

**Features**:
- Provider registration
- Federated routing (NOT nesting)
- Billing and usage tracking
- Multi-tenancy and quotas

**Key distinction**: Platform orchestrator is a **smart router**, not a nested orchestrator. Provider orchestrators make their own placement decisions.

**Business doc**: `.docs/.business/monetization.md`

**Requirements**:
- Platform orchestrator MUST authenticate providers and MUST maintain a registry of provider endpoints and capabilities.
- Routing MUST be federated (provider orchestrators make placement decisions); nested scheduling MUST NOT occur.
- Tenancy context MUST be preserved across federation and included in billing records.
- Provider-level SLIs/labels (availability, capacity) SHOULD inform routing; fallbacks MAY be used when signals are missing.

---

## 6. API Contracts

### 6.0 Model Reference Format (Canonical)

**Accepted forms**:
- Client input (`model` in Agentic API) MAY be one of:
  - `hf:{org}/{repo}` or `hf:{org}/{repo}@{rev}` or `hf:{org}/{repo}@{rev}::file={path}`
  - `file:/abs/path/to/model.gguf`
  - Alias without a scheme (e.g., `llama-7b`)

**Resolution rules**:
- Orchestratord (catalog) MUST resolve aliases to a canonical `model_ref` prior to scheduling.
- Orchestratord SHOULD pin `@rev` to an immutable commit SHA and SHOULD pin a concrete artifact via `::file=...` for determinism.
- Pool-managerd MUST receive a normalized `model_ref` that starts with `hf:` or `file:` and MUST NOT perform alias resolution.
- The model-provisioner in pool-managerd MUST support:
  - `hf:` — Downloading the specified artifact from Hugging Face.
  - `file:` — Treating the path as a local file.
- Other schemes (e.g., `https:`, `s3:`) MAY be added in future milestones and are NOT REQUIRED for M0.

**Worker contract**:
- Worker-orcd MUST be provided a concrete filesystem path to the model artifact at startup.
- Worker ready callbacks MUST include the resolved `model_ref` for traceability.

### 6.1 Client → Orchestrator (Agentic API)

**Task submission**:
```
POST /v2/tasks
{
  "session_id": "sess-abc",
  "model": "llama-3.1-8b",
  "prompt": "Hello world",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42,
  "priority": "interactive"
}

Response (202 Accepted):
{
  "job_id": "job-xyz",
  "status": "queued",
  "queue_position": 2,
  "events_url": "/v2/tasks/job-xyz/events"
}
```

**Streaming**:
```
GET /v2/tasks/{job_id}/events (SSE)

Events:
- queued → started → token* → metrics* → end
- error (if failure or cancellation)
```

**Requirements**:
- Requests MUST include `model`, `prompt`, and `max_tokens`; `seed` SHOULD be provided for determinism, otherwise orchestrator MAY supply one and MUST record it.
- `priority` MUST be one of `interactive` or `batch`; unknown values MUST be rejected with 400.
- Orchestratord MUST validate model existence and token budgets at admission; invalid requests MUST return 4xx with a stable error code.
- When the queue is full and policy=`reject`, orchestrator MUST return 429 with a Retry-After where possible; for `drop-lru`, orchestrator MUST emit metrics indicating eviction.
- SSE endpoint MUST be idempotent to reconnects for the same `job_id` and SHOULD resume streaming from the last sent offset when supported; otherwise it MAY restart from the latest available position.

**Specs**: `bin/orchestratord-crates/agentic-api/.specs/00_agentic_api.md`

---

### 6.2 Orchestrator ↔ Pool Manager

**Pool registration**:
```
POST /v2/pools/register
{
  "pool_id": "pool-1",
  "endpoint": "http://192.168.1.100:9200",
  "gpus": [...]
}
```

**Heartbeat**:
```
POST /v2/pools/{id}/heartbeat
{
  "pool_id": "pool-1",
  "gpus": [...],
  "workers": [...]
}
```

**Requirements**:
- Pool registration MUST be authenticated (see §8.1) and MUST include `pool_id`, `endpoint`, and initial GPU inventory.
- Heartbeats MUST be sent at the configured interval and MUST include current GPU VRAM state and worker states; missing heartbeats beyond timeout MUST mark the pool unavailable.
- Orchestratord MUST validate `pool_id` uniqueness and SHOULD reject re-registration attempts that conflict without explicit takeover semantics.
- Worker start commands MUST be accepted only after preflight checks (VRAM availability and compatibility) have passed in pool-managerd.

**Worker start command**:
```
POST /v2/workers/start
{
  "model_ref": "hf:author/repo@rev::file=models/model.Q4_K_M.gguf",
  "gpu_id": 0
}
```

**Specs**: `bin/pool-managerd-crates/control-api/.specs/00_control_api.md`
  (lifecycle commands and state), and `bin/pool-managerd/.specs/00_pool-managerd.md` §13 (multi-pool registration & heartbeat)

---

### 6.3 Pool Manager ↔ Worker

**Worker ready callback**:
```
POST /v2/internal/workers/ready
{
  "worker_id": "worker-abc",
  "model_ref": "llama-7b",
  "vram_bytes": 17000000000,
  "uri": "http://localhost:8001"
}
```

**Specs**: `bin/pool-managerd-crates/worker-lifecycle/.specs/00_worker_lifecycle.md`

---

### 6.4 Orchestrator → Worker (Direct)

**Inference execution**:
```
POST {worker_uri}/execute
{
  "job_id": "job-xyz",
  "prompt": "Hello world",
  "max_tokens": 100,
  "seed": 42
}

Response: SSE stream
- started
- token* (multiple)
- metrics* (periodic)
- end
- error (on failure or cancellation)
```

**Cancellation**:
```
POST {worker_uri}/cancel
{
  "job_id": "job-xyz"
}
```

Cancellation semantics:
- Idempotent: repeated cancels for the same `job_id` MUST be safe and return the same terminal outcome.
- Prompt propagation: orchestrator MUST issue cancel immediately on client request or stream disconnect.
- Worker behavior: upon cancel, worker MUST stop decoding promptly, free resources, and emit SSE `error` with a stable code `CANCELLED`.
- Acknowledgement: worker SHOULD return HTTP 202 for `POST /cancel` if cancellation has been accepted.
- Deadline: orchestrator SHOULD enforce a cancellation deadline (default 5s) after which it treats the job as cancelled and closes client SSE.

**Requirements**:
- SSE stream event order MUST be: `started` → zero or more `token` → zero or more `metrics` (interleaved as needed) → terminal `end` or `error`.
- Exactly one terminal event MUST be emitted per job (`end` on success, `error` on failure/cancel).
- Workers MUST include stable error codes; cancellation MUST use `CANCELLED`.
- Orchestrator MUST propagate `X-Correlation-Id` to worker requests; workers SHOULD echo it in SSE metadata.
- Orchestrator SHOULD enforce request timeouts per job policy and close SSE cleanly on timeout with a terminal event.

**Specs**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

## 7. Quality Attributes

### 7.1 Determinism

**Normative requirement**:
- For a fixed model, parameters, and seed, the system MUST produce the same output for the same prompt across runs and nodes.

**Enforcement**:
- Worker-orcd MUST allocate and keep all model weights, KV cache, and activations in VRAM only.
- Engine versions and kernel parameters MUST be pinned and recorded per job.
- Sampling MUST be deterministic for identical inputs and seeds; non-deterministic operations MUST be disabled or replaced.
- SSE event ordering MUST be stable; token emission for a given job and seed MUST be identical.

**Recording**:
- Proof bundles SHOULD include seed, model_ref (including pinned @rev and artifact), engine version, and device info sufficient to reproduce runs.

**Verification**:
- Property tests MUST verify determinism across repeated runs and across workers/nodes.
- See `.docs/test-case-discovery-method.md` for verification approach.

---

### 7.2 Performance

**Latency targets (measurement points)**:
- Queue admission SHOULD complete within 10ms measured from HTTP receive to enqueue decision.
- Scheduling decision SHOULD complete within 50ms measured from job-ready to dispatch command issued.
- Worker startup SHOULD complete within 60s measured from start command to ready callback receipt.
- First token latency SHOULD be under 100ms measured from worker execute accept to first SSE `token` event.
- Token generation rate SHOULD be within 20–100 tokens/sec depending on model; deviations MAY be acceptable with justification in metrics.

**Throughput and limits**:
- Queue capacity MUST be configurable; default SHOULD be 100 jobs.
- Worker concurrency for M0 MUST be 1 job per worker (batch=1).
- Multi-job batching MAY be enabled in M1+; when enabled, batching policies MUST be documented and observable via metrics.

---

### 7.3 Reliability

**Availability**:
- The service SHOULD achieve 99.9% uptime (3 nines) for orchestrator in supported deployments.
- On worker failures, the system SHOULD degrade gracefully by failing affected jobs or rerouting per policy without impacting unrelated jobs.
- Automatic retry with backoff SHOULD be applied where policies permit; retries MUST NOT violate idempotency or determinism guarantees.

**Fault tolerance**:
- Pool-managerd MUST detect worker process exits and perform operational cleanup (state removal, VRAM accounting release) promptly.
- Orchestratord MUST detect missed heartbeats from pools within a configured timeout and mark them unavailable for scheduling.
- Retry policies MUST be configurable; defaults SHOULD be conservative to avoid thundering herds.

**Observability**:
- Components MUST emit structured JSON logs and Prometheus metrics sufficient to diagnose failures and capacity issues.
- Key reliability events (crash, retry, timeout, cancel) SHOULD have stable event codes in logs.
- Test artifacts and incident analyses MAY include proof bundles to capture timelines and seeds for reproduction.

---

### 7.4 Scalability

**Horizontal scaling**:
- Orchestratord MUST support registration of multiple pool-managerd instances to scale across GPU nodes.
- Pool-managerd MUST support multiple workers per node to scale with additional GPUs.
- In platform mode, a platform orchestrator MAY federate across multiple provider orchestrators; routing policies SHOULD avoid nested placement decisions.

**Scaling properties**:
- Orchestrator’s scheduling and registry operations SHOULD remain O(log N) or better with respect to total workers.
- Metrics cardinality SHOULD be bounded when scaling; high-cardinality labels MUST be avoided or sampled.

**Configuration**:
- Scaling parameters (max pools, max workers per pool, heartbeat fan-in) SHOULD be configurable and documented.

**Limits (M0)**:
- Single orchestrator instance
- Multiple pool managers supported
- Workers scale per GPU

**Limits (M1+)**:
- Orchestrator HA/clustering: future
- Current: single orchestrator, stateless (can restart)

---

## 8. Security & Compliance

### 8.1 Authentication

**Home mode (M0)**:
- Orchestratord and pool-managerd MUST accept requests from localhost without authentication by default.
- Orchestratord and pool-managerd MUST bind to loopback-only by default in home mode. Exposing to non-loopback interfaces MUST require an explicit configuration change.
- When exposed beyond localhost in home mode, requests SHOULD be authenticated using a bearer token.

**Platform mode (M2+/Platform)**:
- All client requests to orchestratord MUST be authenticated using HTTP bearer tokens.
- Orchestratord MUST validate bearer tokens on every request and MUST reject unauthenticated or invalid tokens with HTTP 401/403.
- Pool-managerd MUST authenticate registration and heartbeat requests to orchestratord using a configured shared secret or bearer token.
- Inter-service calls from orchestratord to pool-managerd MAY use mTLS or bearer tokens; if bearer tokens are used, pool-managerd MUST validate them.

**Future provisions**:
- The system MAY add OAuth2/OpenID Connect and API key authentication in future milestones; these are NOT REQUIRED for M0.
- If OAuth2/OIDC is configured, orchestratord SHOULD validate audience/scope claims and SHOULD enforce token expiry.

---

### 8.2 EU Compliance (GDPR)

**Data residency**:
- MUST: Pool-managerd instances serving production workloads MUST be located within EU regions only.
- MUST: Orchestratord MUST refuse to schedule work to pools that are not explicitly marked as EU-resident.
- MUST: Worker-orcd MUST NOT transmit model inputs/outputs to endpoints outside the EU.

**Geo-verification**:
- SHOULD: Provider registration MUST include asserted region metadata and evidence (e.g., provider-declared ISO 3166-1 country code and hosting provider region ID).
- SHOULD: Orchestratord SHOULD verify region metadata against an allow-list and SHOULD reject pools with ambiguous or missing region information.

**Data handling**:
- MUST: Logs MUST NOT include raw prompts, tokens, or PII unless explicitly configured for debugging; when enabled, such logs MUST be redacted or truncated per policy.
- MAY: Proof bundles MAY include hashed or redacted payloads but MUST include correlation identifiers sufficient for auditing.

**Audit trail**:
- MUST: All requests MUST be logged with correlation IDs propagated end-to-end.
- SHOULD: Access to audit logs SHOULD be role-restricted and retention policies SHOULD comply with GDPR data minimization.

---
**Compliance docs**: `.docs/.business/monetization.md`

---

### 8.3 Multi-Tenancy (Platform Mode)

**Isolation guarantees**:
- Orchestratord MUST enforce tenant-level isolation of compute and data paths.
- Pool-managerd MUST ensure workers for different tenants do not share processes or VRAM allocations.
- Logs and metrics MUST NOT expose cross-tenant identifiers except in aggregated, non-identifying form.

**Quotas and limits**:
- Orchestratord MUST support per-tenant quotas (e.g., VRAM ceilings, max concurrent jobs, token budgets) and MUST reject or queue work when quotas are exceeded.
- Submission rate limiting SHOULD be enforced per tenant on task endpoints.

**Authorization**:
- All client API requests in platform mode MUST include a tenant identifier bound to the authenticated principal.
- Orchestratord MUST authorize every request against the tenant’s entitlements before admission.
- Inter-service requests (orchestrator ↔ pool-managerd) SHOULD carry tenant context for auditing; pool-managerd MAY validate tenant context when relevant.

**Data separation**:
- Shared model caches MAY be used across tenants; tenant runtime data (inputs/outputs/proofs) MUST be segregated with per-tenant namespaces.
- Proof bundles MUST include correlation and tenant identifiers and SHOULD avoid raw prompts unless debugging with redaction is explicitly enabled.

**Observability and billing**:
- Metrics emitted by components SHOULD include `tenant_id` where policy permits; systems MAY hash or omit this label to reduce privacy risk.
- Usage accounting MUST be recorded per tenant for tokens generated, inference duration, and (where available) VRAM occupancy-time.

**Specs**: `bin/orchestratord-crates/platform-api/.specs/00_platform_api.md`

---

## 9. Development Workflow

### 9.1 Spec-Driven Development

**Process**: Spec → Contract → Tests → Code

**Workflow**:
1. Write spec (RFC-2119 normative requirements)
2. Define contracts (API, data structures)
3. Write tests (BDD, property tests, unit tests)
4. Implement code (guided by specs and tests)

**Docs**: `README_LLM.md`, `.docs/workflow.md`

---

### 9.2 Testing Strategy

**Test types (scope and requirements)**:
- Unit tests MUST exist per crate to cover core functions and error paths.
- Integration tests SHOULD validate cross-crate interactions and HTTP boundaries.
- Contract tests MUST verify API conformance against the OpenAPI/contracts with stable IDs.
- Property tests MUST enforce critical invariants (e.g., determinism, queue bounds, idempotent cancellation).
- BDD tests SHOULD derive from spec scenarios with stable identifiers and traceability to requirements.

**Artifacts and determinism**:
- All automated test runs SHOULD produce proof bundles under `<crate>/.proof_bundle/<type>/<run_id>/`.
- Proof bundles MUST include an autogenerated header and MUST respect `LLORCH_RUN_ID` and `LLORCH_PROOF_DIR` when set.
- Tests that rely on randomness MUST seed RNGs explicitly and record seeds in proof bundles.

**References**:
- Proof bundle standard: `.specs/00_proof-bundle.md`

---

### 9.3 CI/CD Pipeline

**Gates (must-pass checks)**:
- Stage 0 (Spec hygiene) MUST pass link checks and ID stability; specs MUST use RFC‑2119 terms.
- Stage 1 (Code quality) MUST pass `fmt` and `clippy` with no new warnings in changed code.
- Stage 2 (Tests) MUST pass unit, integration, and property tests; flaky tests MUST be quarantined or fixed before merge.
- Stage 3 (Contract compliance) MUST validate APIs against OpenAPI/contracts; breaking changes MUST be versioned and documented.
- Stage 4 (Determinism suite) MUST verify determinism properties on supported targets.
- Stage 5 (Metrics emission) SHOULD validate presence and type/label cardinality of required metrics; missing optional metrics MAY be tolerated with waivers.
- Stage 6 (E2E acceptance) SHOULD pass scenario-based BDD tests for targeted milestones before release.

**Planning references**:
- Roadmap documents SHOULD be maintained at `TODO.md` and `.plan/00_meta_plan.md`.

---

## 10. Metrics & Observability

### 10.1 Metrics Contract

**Orchestrator metrics**:
- `orchd_queue_depth{priority}`
- `orchd_tasks_enqueued_total{outcome}`
- `orchd_tasks_dispatched_total{worker_id, outcome}`
- `orchd_scheduling_latency_ms`

**Pool metrics**:
- `pool_mgr_gpu_vram_total_bytes{gpu_id}`
- `pool_mgr_gpu_vram_allocated_bytes{gpu_id}`
- `pool_mgr_workers_total{status}`
- `pool_mgr_worker_starts_total{outcome}`

**Worker metrics**:
- `worker_inference_duration_ms`
- `worker_tokens_generated_total`
- `worker_vram_bytes{worker_id}`

**Requirements**:
- Metric names, types, and labels MUST conform to `bin/.specs/71_metrics_contract.md`.
- Required metrics MUST be emitted at INFO-level operation; optional metrics MAY be omitted but SHOULD have waivers referenced in CI (Stage 5).
- Label cardinality MUST be bounded; high-cardinality labels (e.g., `job_id`) MUST NOT be used. `worker_id` MAY be included where documented.
- All metrics MUST include `component` and SHOULD include `pool_id`/`worker_id` where applicable.
- Metric units MUST be encoded in the name (e.g., `_ms`, `_bytes`, `_total`).

**Spec**: `.specs/71_metrics_contract.md`

---

### 10.2 Logging

**Format**:
- All components MUST emit JSON structured logs.
- Log schemas SHOULD be stable and versioned to avoid breaking ingestion.

**Levels**:
- Components MUST support standard levels: ERROR, WARN, INFO, DEBUG, TRACE.
- Default level SHOULD be INFO in production; DEBUG/TRACE MAY be enabled temporarily for diagnostics.

**Correlation**:
- `X-Correlation-Id` MUST be accepted from clients and propagated across orchestrator → pool-managerd → worker calls.
- If the header is absent, orchestrator MUST generate a new correlation ID and propagate it downstream.

**Content**:
- Logs MUST include component, timestamp, level, correlation_id, and stable event codes for key actions (admission, schedule, dispatch, execute, cancel).
- Logs SHOULD avoid raw prompts/tokens unless explicitly enabled; when enabled, content MUST be redacted or truncated per policy.

**Narration**:
- Human-readable narration fields MAY be included for developer ergonomics but MUST NOT replace structured fields.

---

## 11. Configuration

### 11.1 Orchestrator Config

```yaml
orchestratord:
  bind: "0.0.0.0:8080"
  mode: "agentic"  # or "platform"
  
  queue:
    capacity: 100
    policy: "reject"  # or "drop-lru"
  
  scheduling:
    algorithm: "least-loaded"  # or "most-vram-free", "round-robin"
    eviction_policy: "lru"     # or "lfu", "manual"
  
  timeout:
    default_ms: 300000  # 5 minutes
    max_ms: 1800000     # 30 minutes
```

**Requirements**:
- Defaults MUST be applied when fields are omitted; values MUST be validated (e.g., `default_ms` ≤ `max_ms`).
- `mode` MUST be `agentic` or `platform`; unknown values MUST be rejected.
- `queue.capacity` MUST be a positive integer; `policy` MUST be one of `reject` or `drop-lru`.
- `scheduling.algorithm` SHOULD be one of the documented strategies; unknown algorithms MAY be rejected or fall back to a safe default with a warning.
- Timeouts MUST be enforced per job; values MAY be overridden per-request if allowed by policy.

---

### 11.2 Pool Manager Config

```yaml
pool-managerd:
  bind: "0.0.0.0:9200"
  pool_id: "pool-1"
  
  orchestrator:
    url: "http://orchestrator:8080"
    heartbeat_interval_ms: 15000
  
  models:
    cache_dir: "/var/cache/llama-orch/models"
```

**Requirements**:
- `bind` MUST be a valid socket address; in home mode it SHOULD bind loopback unless explicitly overridden.
- `pool_id` MUST be unique per orchestrator; collisions MUST be rejected at registration.
- `orchestrator.url` MUST be a reachable HTTP endpoint; `heartbeat_interval_ms` MUST be a positive integer and SHOULD default to 15000.
- `models.cache_dir` MUST be writable by pool-managerd; insufficient permissions MUST fail fast at startup.

---

### 11.3 Worker Config

```bash
worker-orcd \
  --worker-id worker-abc \
  --model /models/llama-7b.gguf \
  --gpu-device 0 \
  --port 8001 \
  --callback-url http://pool:9200/v2/internal/workers/ready
```

**Requirements**:
- `--worker-id` MUST be unique per pool; collisions MUST be rejected.
- `--model` MUST point to a readable file; startup MUST fail fast if missing or unreadable.
- `--gpu-device` MUST refer to a valid CUDA device index present on the node.
- `--port` SHOULD default to an ephemeral port if not provided; chosen port MUST be free at startup.
- `--callback-url` MUST be reachable by pool-managerd and MUST use HTTP(S); ready callback MUST include `model_ref` and `vram_bytes`.

---

## 12. Crate Dependency Graph

```
orchestratord
├── scheduling (admission, queue, job-tracker, scheduler, eviction)
├── platform-api (marketplace facade)
├── agentic-api (standard API)
├── pool-registry (track pools)
├── streaming (SSE relay)
├── task-cancellation (cancel propagation)
├── job-timeout (timeout enforcement)
└── backpressure (backpressure handling)

pool-managerd
├── gpu-inventory (NVML FFI)
├── capability-matcher (preflight validation)
├── model-cache (storage)
├── model-provisioner (download)
├── model-catalog (metadata)
├── worker-lifecycle (spawn/monitor workers)
├── control-api (HTTP API)
├── error-ops (operational cleanup)
└── pool-registration-client (register with orchestrator)

worker-orcd
├── http (HTTP handlers: execute, health)
├── startup (initialization and callbacks)
├── cuda (FFI to C++/CUDA)
├── health-monitor (self-monitoring)
└── error-handler (error handling)
```

---

## 13. Milestone Roadmap

### M0: Single GPU (v0.1.0)

**Goal**: Home lab, single user, single GPU

**Features**:
- Single orchestrator instance
- Single pool manager
- Single worker per GPU
- VRAM-only enforcement
- Determinism guarantee
- Basic API (task submission, streaming)

**Status**: In progress

---

### M1: Multi-GPU (v0.2.0)

**Goal**: Single node, multiple GPUs

**Features**:
- Tensor parallelism (split models across GPUs)
- Multiple workers per node
- Model hot-loading (swap models without restart)

---

### M2: Multi-Node (v0.3.0)

**Goal**: Enterprise, GPU cluster

**Features**:
- Multiple pool managers
- Cluster-wide orchestration
- Load balancing across nodes

---

### M3: Platform (v0.4.0)

**Goal**: GPU marketplace

**Features**:
- Provider registration
- Federated routing
- Billing and usage tracking
- Multi-tenancy

---

## 14. References

### Specifications

**Component specs**:
- `bin/orchestratord/.specs/00_orchestratord.md` (ORCH-1xxx)
- `bin/pool-managerd/.specs/00_pool-managerd.md` (POOL-2xxx)
- `bin/worker-orcd/.specs/00_worker-orcd.md` (WORK-3xxx)

**Crate specs**: See individual crate `.specs/` directories

### Documentation

- `README_LLM.md` — AI-optimized project overview
- `.docs/workflow.md` — Development workflow
- `.docs/.business/monetization.md` — Marketplace business model
- `.docs/test-case-discovery-method.md` — Testing approach
- `TODO.md` — Active roadmap
- `.plan/00_meta_plan.md` — Project plan

### Contracts

- `/contracts/openapi/*.yaml` — OpenAPI specs

---

## 15. Traceability

**Code**: Monorepo at `/home/vince/Projects/llama-orch/`  
**Tests**: Per-crate `tests/` directories, BDD features in `bdd/`  
**Artifacts**: `.proof_bundle/<type>/<run_id>/`  
**CI**: GitHub Actions (`.github/workflows/`)  
**Spec IDs**: SYS-0xxx (this document)

---

**Version**: 0.1.0  
**Last Updated**: 2025-10-03  
**Status**: Living document (updated as system evolves)

---

**End of System Specification**
