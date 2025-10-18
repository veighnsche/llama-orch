# Component: Worker Registry (queen-rbee, RAM)

**Location:** `bin/queen-rbee/src/worker_registry.rs`  
**Type:** Ephemeral in-memory storage  
**Language:** Rust  
**Created by:** TEAM-043  
**Status:** ✅ IMPLEMENTED

## Overview

In-memory registry tracking active workers across ALL registered hives. Ephemeral - cleared on queen-rbee restart. Used for request routing and load balancing.

## Data Model

```rust
pub struct WorkerInfo {
    pub id: String,                  // Worker UUID
    pub url: String,                 // http://192.168.1.100:8081
    pub model_ref: String,           // hf:TheBloke/TinyLlama-1.1B
    pub backend: String,             // cuda, metal, cpu
    pub device: u32,                 // GPU device ID
    pub state: WorkerState,          // Loading, Idle, Busy
    pub slots_total: u32,            // Total inference slots
    pub slots_available: u32,        // Available slots
    pub vram_bytes: Option<u64>,     // VRAM usage
    pub node_name: String,           // TEAM-046: Which hive owns this
}

pub enum WorkerState {
    Loading,  // Model loading
    Idle,     // Ready for requests
    Busy,     // Processing request
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ queen-rbee (Orchestrator)                               │
│                                                         │
│  Worker Registry (RAM)                                  │
│  ┌───────────────────────────────────────────────────┐ │
│  │ HashMap<WorkerId, WorkerInfo>                     │ │
│  │                                                   │ │
│  │ worker-abc123 → {                                 │ │
│  │   url: "http://192.168.1.100:8081",              │ │
│  │   model: "tinyllama",                            │ │
│  │   node: "gpu-server-1",  ← Maps to hive         │ │
│  │   state: Idle                                    │ │
│  │ }                                                 │ │
│  │                                                   │ │
│  │ worker-def456 → {                                 │ │
│  │   url: "http://192.168.1.101:8081",              │ │
│  │   model: "llama2-7b",                            │ │
│  │   node: "gpu-server-2",                          │ │
│  │   state: Busy                                    │ │
│  │ }                                                 │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
   ┌──────────┐        ┌──────────┐
   │ Hive 1   │        │ Hive 2   │
   │ (node 1) │        │ (node 2) │
   └──────────┘        └──────────┘
```

## API Methods

### Core Operations
```rust
// Create new registry
pub fn new() -> Self

// Register worker
pub async fn register(&self, worker: WorkerInfo)

// Update worker state
pub async fn update_state(&self, worker_id: &str, state: WorkerState) -> bool

// Get worker by ID
pub async fn get(&self, worker_id: &str) -> Option<WorkerInfo>

// List all workers
pub async fn list(&self) -> Vec<WorkerInfo>

// Remove worker
pub async fn remove(&self, worker_id: &str) -> bool

// Clear all workers (testing)
pub async fn clear(&self)

// Count workers
pub async fn count(&self) -> usize
```

### Extended Operations (TEAM-046)
```rust
// List workers with extended info
pub async fn list_workers(&self) -> Result<Vec<WorkerInfoExtended>>

// Find workers by criteria
pub async fn find_by_model(&self, model_ref: &str) -> Vec<WorkerInfo>
pub async fn find_by_node(&self, node_name: &str) -> Vec<WorkerInfo>
pub async fn find_idle_workers(&self) -> Vec<WorkerInfo>
```

## Lifecycle

### 1. Worker Registration
```rust
// Worker spawned on hive → Calls queen-rbee ready callback
POST /v1/workers/ready
{
    "worker_id": "worker-abc123",
    "url": "http://192.168.1.100:8081",
    "model_ref": "hf:TheBloke/TinyLlama",
    "backend": "cuda",
    "device": 0,
    "node_name": "gpu-server-1"
}

// Queen registers in RAM
registry.register(WorkerInfo { ... }).await;
```

### 2. State Updates
```rust
// Worker transitions: Loading → Idle → Busy → Idle
registry.update_state("worker-abc123", WorkerState::Idle).await;
```

### 3. Request Routing
```rust
// Find idle worker for model
let workers = registry.find_by_model("tinyllama").await;
let idle = workers.into_iter()
    .find(|w| w.state == WorkerState::Idle);

if let Some(worker) = idle {
    // Route request to worker.url
}
```

### 4. Worker Removal
```rust
// Worker dies or hive removes it
registry.remove("worker-abc123").await;
```

## Concurrency

**Thread-Safe:** Uses `Arc<RwLock<HashMap>>` (TEAM-080)

```rust
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
}
```

- Multiple readers allowed simultaneously
- Single writer blocks readers
- Clone-able for concurrent access

## Integration Points

### Hive Ready Callback
```rust
// bin/queen-rbee/src/http/workers.rs
async fn handle_worker_ready(
    State(state): State<AppState>,
    Json(req): Json<WorkerReadyRequest>,
) {
    state.worker_registry.register(WorkerInfo {
        id: req.worker_id,
        url: req.url,
        model_ref: req.model_ref,
        node_name: req.node_name,  // TEAM-046
        state: WorkerState::Idle,
        ...
    }).await;
}
```

### Inference Routing
```rust
// bin/queen-rbee/src/http/inference.rs
async fn handle_inference(
    State(state): State<AppState>,
    Json(req): Json<InferenceRequest>,
) {
    // Find worker for model
    let workers = state.worker_registry
        .find_by_model(&req.model)
        .await;
    
    let worker = workers.into_iter()
        .find(|w| w.state == WorkerState::Idle)
        .ok_or("No idle workers")?;
    
    // Route to worker
    forward_to_worker(&worker.url, req).await
}
```

### Worker Listing
```rust
// GET /v1/workers/list
async fn handle_list_workers(
    State(state): State<AppState>,
) -> Json<Vec<WorkerInfo>> {
    let workers = state.worker_registry.list().await;
    Json(workers)
}
```

## Maturity Assessment

**Status:** ✅ **PRODUCTION READY**

**Strengths:**
- ✅ Thread-safe concurrent access (TEAM-080)
- ✅ Complete CRUD operations
- ✅ Node mapping (TEAM-046)
- ✅ VRAM tracking
- ✅ State management
- ✅ Fast in-memory lookups

**Limitations:**
- ⚠️ Ephemeral - lost on restart
- ⚠️ No persistence layer
- ⚠️ No worker health tracking (relies on hive)
- ⚠️ No automatic cleanup of stale workers
- ⚠️ No metrics/telemetry

**Recommended Improvements:**
1. Add worker heartbeat mechanism
2. Add automatic stale worker cleanup
3. Add metrics (worker count, state distribution)
4. Add worker history/audit log
5. Add load balancing hints (CPU/memory usage)

## Comparison: Queen vs Hive Worker Registry

| Feature | Queen Registry | Hive Registry |
|---------|---------------|---------------|
| **Scope** | All hives | Single hive |
| **Storage** | RAM | RAM |
| **Persistence** | None | None |
| **Purpose** | Request routing | Lifecycle management |
| **Node mapping** | ✅ Yes (TEAM-046) | N/A (local only) |
| **Health checks** | ❌ No (relies on hive) | ✅ Yes (30s interval) |
| **PID tracking** | ❌ No | ❌ No (gap!) |
| **Fail-fast** | ❌ No | ✅ Yes (TEAM-096) |

## Testing

```bash
# Unit tests
cargo test -p queen-rbee worker_registry

# Integration test
# 1. Start queen-rbee
# 2. Start rbee-hive (registers workers)
# 3. Query workers
curl http://localhost:8080/v1/workers/list
```

## Related Components

- **Beehive Registry** - Persistent hive registration
- **Hive Worker Registry** - Per-hive worker tracking
- **HTTP API** (`queen-rbee/src/http/workers.rs`) - Worker endpoints
- **Inference Router** (`queen-rbee/src/http/inference.rs`) - Uses registry for routing

---

**Created by:** TEAM-043  
**Enhanced by:** TEAM-046 (node mapping), TEAM-080 (concurrency)  
**Last Updated:** 2025-10-18  
**Maturity:** ✅ Production Ready (with noted limitations)
