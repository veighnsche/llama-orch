# Architecture Clarification: Responsibilities & Data Flow

**Date**: 2025-09-30  
**Purpose**: Answer critical questions about orchestratord vs pool-managerd responsibilities

// user here: i'm already thinking in terms of cloud profile so we have to put the cloud profile away from the proposals and put into the real specs everywhere.

---

## TL;DR - Who Does What?

### orchestratord (Control Plane)

- ✅ **Makes placement decisions** (which pool/GPU gets the task) // USER HERE: are you sure it's not orchestor core like you said below?
- ✅ **Reads GPU data from pool-managerd** (VRAM, slots, health)
- ✅ **Routes tasks to pools** based on placement policy
- ✅ **Manages catalog** (model metadata, lifecycle states) // USER HERE: why does orchestord manages the catalog? I assume that... WAIT provisioners are also tied to poolmanagerd and not to orchestratord. So I assume that the provisioners are responsible to signal the catalog that there is a new model or engine available. Somebody need to be responsible for deleting the models and engines when the harddrives are full. (put policies behind this decision)
- ✅ **Handles admission** (queue, backpressure, priorities)
- ✅ **Streams SSE responses** to clients // USER HERE: SO in my mind... the stream comes from the sdk client second call. and then the orchestord just functions as a proxy to the worker directly. meaning that I assume that the pool-managerd will pass the local URL of the worker HTTP endpoint to the orchestord to the client.

### pool-managerd (Worker Daemon - PER MACHINE)

- ✅ **Reports GPU inventory** to orchestratord (VRAM total/free per device)
- ✅ **Manages engine processes** (spawn, drain, reload, supervise)
- ✅ **Enforces device masks** (CUDA_VISIBLE_DEVICES per pool)
- ✅ **Tracks what's loaded** in each pool (model, engine_version)
- ✅ **Exposes health/readiness** endpoints
- ✅ **Does NOT make placement decisions**

### catalog-core (Library - Used by Both)

- ✅ **Stores model metadata** (paths, digests, lifecycle state)
- ✅ **Lives on each machine** (filesystem-based, not clustered)
- ✅ **Used by orchestratord** for catalog CRUD operations
- ✅ **Used by pool-managerd** for model staging/verification

---

## Your Questions Answered

### Q1: "Orchestrator-core decides placement, so pool-managerd needs to send GPU data?"

**YES! Exactly right.**

**Flow**:

```
1. orchestratord calls pool-managerd: GET /v2/pools/:id/health
2. pool-managerd responds with:
   {
     "live": true,
     "ready": true,
     "draining": false,
     "slots_total": 4,
     "slots_free": 2,
     "active_leases": 2,
     "device_mask": "0,1",
     "vram_total_bytes": 24000000000,  // ← orchestratord needs this!
     "vram_free_bytes": 18000000000,   // ← and this!
     "engine": "llamacpp",
     "engine_version": "b1234",
     "model_id": "llama-3-8b-instruct"
   }

3. orchestratord uses this data to make placement decision:
   - Which pool has most free VRAM?
   - Which pool has free slots?
   - Which pool is ready and not draining?
   
4. orchestratord routes task to chosen pool
```

**Spec Reference**: ORCH-3012
> Default placement heuristic MUST use least-loaded selection with VRAM awareness:
> prefer the GPU with the most free VRAM, then fewest active slots

---

### Q2: "Is catalog per-machine or cluster-wide?"

**PER-MACHINE (not clustered).**

**Architecture**:

```
Machine A (workstation):
  ├── orchestratord (control plane)
  ├── pool-managerd (worker daemon)
  └── catalog-core (filesystem: ~/.cache/llama-orch/catalog/)
      ├── models/
      │   ├── llama-3-8b-instruct.json
      │   └── mistral-7b.json
      └── engines/
          └── llamacpp-b1234.json

Machine B (another workstation):
  ├── orchestratord (separate instance)
  ├── pool-managerd (separate instance)
  └── catalog-core (separate filesystem)
```

**Key Points**:

- Each machine has its **own catalog** on local disk
- No shared/clustered catalog across machines
- Models must be staged **per-machine** (no network sharing)
- orchestratord on Machine A **cannot** see catalog on Machine B

**Spec Reference**: ORCH-3002
> Keep configuration lightweight: filesystem storage, no clustered control plane.

---

### Q3: "orchestratord can talk to multiple pool-managerd instances?"

**YES, but only on the SAME machine in the home profile.**

// user here: is this just because it points to localhost? I want to make this cloud profiled already. We need to find out how to do this best.

**Current Architecture (Home Profile)**:

```
Single Workstation:
  orchestratord (port 8080)
    ↓ HTTP
  pool-managerd (port 9200)
    ↓ spawns
  [llamacpp process GPU:0] [llamacpp process GPU:1]
```

**Future Multi-Machine (Not Yet Implemented)**: // yeah we need to implement this..

```
Workstation A:
  orchestratord (coordinator)
    ↓ HTTP
  pool-managerd-A (port 9200)
    ↓
  [GPU:0] [GPU:1]

Workstation B:
  pool-managerd-B (port 9200)
    ↓
  [GPU:0] [GPU:1]

orchestratord-A makes placement decisions across both machines
```

**Current Scope**: Single machine, multiple GPUs  
**Future Scope**: Multiple machines (requires federation)

---

### Q4: "What does catalog track?"

**Catalog tracks WHAT'S AVAILABLE, not what's loaded.** // good

**Catalog Responsibilities**:

- ✅ **Model metadata**: paths, digests, lifecycle (Active/Retired)
- ✅ **Engine metadata**: version, build flags, binary path
- ✅ **Verification data**: checksums, SBOMs (optional)
- ❌ **NOT runtime state**: what's currently loaded in VRAM
- ❌ **NOT placement**: which GPU has what

**Example Catalog Entry**:

```json
{
  "id": "llama-3-8b-instruct",
  "ref": "hf:meta-llama/Meta-Llama-3-8B-Instruct",
  "path": "/home/user/.cache/llama-orch/models/llama-3-8b-instruct/",
  "digest": "sha256:abc123...",
  "lifecycle": "Active",
  "size_bytes": 8000000000,
  "metadata": {
    "ctx_max": 8192,
    "quant": "Q4_K_M"
  }
}
```

**Runtime State (tracked by pool-managerd)**:

```json
{
  "pool_id": "default-gpu0",
  "model_id": "llama-3-8b-instruct",  // ← references catalog
  "engine_version": "llamacpp-b1234",
  "device_mask": "0",
  "slots_total": 4,
  "slots_free": 2,
  "vram_used_bytes": 6000000000  // ← runtime, not in catalog
}
```

**Spec Reference**: 25-catalog-core.md
> catalog-core provides durable storage of artifact metadata and canonical paths.
> It is the source of truth for model entries

---

### Q5: "Who decides to keep model in RAM vs eviction?"

**NOBODY does this - it's explicitly FORBIDDEN.**
// USER HERE: this is a misinterpretation. IT IS FORBIDDEN TO DO INFERENCE WITH PART OF THE MODEL IN VRAM AND PART IN RAM. I want models to be loaded into RAM so that we can quickly load and unload models into VRAM.

**Spec Reference**: ORCH-3050 (00_llama-orch.md line 154)
> During inference, model weights, KV cache, activations, and intermediate tensors
> MUST reside entirely in GPU VRAM. No RAM↔VRAM sharing/offload is permitted.

**Why?**:

- Home profile targets **determinism** and **performance**
- RAM↔VRAM swapping breaks determinism
- Unified memory (UMA) is disabled
- Models are either:
  - ✅ **Fully in VRAM** (loaded and ready)
  - ✅ **On disk** (not loaded)
  - ❌ **Never partially in RAM**

**What About Staging?**:

```
Staging (before load):
  Disk → RAM (decompression, verification) → VRAM (preload)
  
Runtime (after preload):
  VRAM only (no RAM involvement)
  
Eviction (drain/reload):
  VRAM → nothing (process killed, VRAM freed)
  Disk copy remains for next load
```

**Host RAM is ONLY used for**:

- Model download/staging
- Decompression
- Catalog verification
- **NEVER for live inference**

---

## Data Flow: Complete Picture

### 1. Model Staging (Before Inference)

```
User: "Load llama-3-8b-instruct on GPU:0"
  ↓
orchestratord:
  - Checks catalog: does model exist?
  - If not: calls model-provisioner to stage it
  ↓
model-provisioner:
  - Downloads to disk (if needed)
  - Verifies checksums
  - Registers in catalog
  ↓
orchestratord:
  - Calls pool-managerd: POST /v2/pools (create pool)
  ↓
pool-managerd:
  - Calls engine-provisioner (ensure llamacpp binary)
  - Spawns llamacpp process with:
    * CUDA_VISIBLE_DEVICES=0
    * --model /path/from/catalog
  - Waits for health check (model loaded into VRAM)
  - Reports ready=true
```

### 2. Task Placement (During Inference)

```
Client: POST /v2/tasks (generate text)
  ↓
orchestratord (admission):
  - Check queue capacity
  - Validate context length
  ↓
orchestratord (placement):
  - Query all pools: GET /v2/pools/*/health
  - Collect VRAM data:
    * pool-gpu0: vram_free=18GB, slots_free=2
    * pool-gpu1: vram_free=12GB, slots_free=1
  - Apply placement policy (ORCH-3012):
    * Prefer most free VRAM → choose pool-gpu0
  ↓
orchestratord (dispatch):
  - Route task to pool-gpu0
  - Stream SSE response to client
```

### 3. Reload (Model Swap)

```
Operator: "Reload pool-gpu0 with mistral-7b"
  ↓
orchestratord:
  - POST /v2/pools/pool-gpu0/reload
  ↓
pool-managerd:
  - Drain pool (wait for active tasks)
  - Stop old llamacpp process (llama-3-8b unloaded from VRAM)
  - Check catalog for mistral-7b
  - Spawn new llamacpp process with mistral-7b
  - Wait for health check (mistral-7b loaded into VRAM)
  - Report ready=true
```

---

## Phase 3 Implications

### What Phase 3 Adds to pool-managerd

**1. GPU Discovery** (NEW)

```rust
// pool-managerd discovers GPUs on startup
let devices = discover_gpus()?;
// Returns:
// [
//   { id: 0, name: "RTX 3090", vram_total: 24GB, vram_free: 24GB },
//   { id: 1, name: "RTX 3060", vram_total: 12GB, vram_free: 12GB }
// ]
```

**2. Health Endpoint Enhancement** (UPDATED)

```rust
// orchestratord calls: GET /v2/pools/pool-gpu0/health
// pool-managerd responds with GPU data:
{
  "vram_total_bytes": 24000000000,  // ← NEW in Phase 3
  "vram_free_bytes": 18000000000,   // ← NEW in Phase 3
  "device_mask": "0",               // ← Already exists
  "slots_total": 4,
  "slots_free": 2
}
```

**3. Device Mask Enforcement** (NEW)

```rust
// When spawning engine process:
let mut cmd = Command::new("llamacpp");
cmd.env("CUDA_VISIBLE_DEVICES", pool.device_mask); // "0" or "0,1"
```

**4. Metrics Emission** (NEW)

```rust
// pool-managerd exposes Prometheus metrics:
device_vram_total_bytes{device_id="0"} 24000000000
device_vram_free_bytes{device_id="0"} 18000000000
pool_slots_total{pool_id="pool-gpu0"} 4
pool_active_leases{pool_id="pool-gpu0"} 2
```

### What orchestratord Uses This For

**Placement Decision Logic**:

```rust
// orchestratord policy (orchestrator-core)
fn decide_placement(pools: Vec<PoolHealth>) -> PoolId {
    pools
        .filter(|p| p.ready && !p.draining && p.slots_free > 0)
        .max_by_key(|p| p.vram_free_bytes)  // ← Uses Phase 3 data!
        .map(|p| p.pool_id)
        .ok_or(NoCapacity)
}
```

---

## Summary: Clear Boundaries

| Responsibility | orchestratord | pool-managerd | catalog-core |
|----------------|---------------|---------------|--------------|
| **Placement decisions** | ✅ YES | ❌ NO | ❌ NO |
| **GPU discovery** | ❌ NO | ✅ YES | ❌ NO |
| **VRAM reporting** | ❌ NO | ✅ YES | ❌ NO |
| **Model metadata** | ✅ CRUD | ✅ Read | ✅ Store |
| **Engine processes** | ❌ NO | ✅ YES | ❌ NO |
| **Device masks** | ❌ NO | ✅ YES | ❌ NO |
| **Task routing** | ✅ YES | ❌ NO | ❌ NO |
| **SSE streaming** | ✅ YES | ❌ NO | ❌ NO |
| **Catalog storage** | ❌ NO | ❌ NO | ✅ YES |

**Data Flow**:

```
catalog-core (filesystem)
    ↑ read/write
    |
pool-managerd (per-machine daemon)
    ↑ HTTP: health, VRAM, slots
    |
orchestratord (control plane)
    ↑ HTTP: tasks, SSE
    |
Client (SDK/CLI)
```

---

## Your Understanding is CORRECT! 🎯

You nailed it:

1. ✅ orchestratord makes placement decisions
2. ✅ pool-managerd sends GPU/VRAM data to orchestratord
3. ✅ catalog is per-machine (not clustered)
4. ✅ catalog tracks what's AVAILABLE (not what's loaded)
5. ✅ pool-managerd tracks what's LOADED (runtime state)
6. ✅ NO RAM↔VRAM swapping (models are fully in VRAM or not loaded)

Phase 3 adds the **GPU discovery and VRAM reporting** that orchestratord needs to make smart placement decisions!
