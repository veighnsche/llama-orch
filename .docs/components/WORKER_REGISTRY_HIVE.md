# Component: Worker Registry (rbee-hive, RAM)

**Location:** `bin/rbee-hive/src/registry.rs`  
**Type:** Ephemeral in-memory storage  
**Language:** Rust  
**Created by:** TEAM-026  
**Status:** ✅ IMPLEMENTED (with lifecycle gaps)

## Overview

In-memory registry tracking workers spawned by THIS hive only. Tightly coupled with worker lifecycle management. Ephemeral - cleared on hive restart.

## Data Model

```rust
pub struct WorkerInfo {
    pub id: String,                      // Worker UUID
    pub url: String,                     // http://127.0.0.1:8081
    pub model_ref: String,               // hf:TheBloke/TinyLlama
    pub backend: String,                 // cuda, metal, cpu
    pub device: u32,                     // GPU device ID
    pub state: WorkerState,              // Loading, Idle, Busy
    pub last_activity: SystemTime,       // For idle timeout
    pub slots_total: u32,                // Total slots
    pub slots_available: u32,            // Available slots
    pub failed_health_checks: u32,       // TEAM-096: Fail-fast counter
    // ❌ MISSING: pub pid: Option<u32>  // Process ID (critical gap!)
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
│ rbee-hive (Worker Pool Manager)                         │
│                                                         │
│  Worker Registry (RAM) ← Tightly coupled with lifecycle│
│  ┌───────────────────────────────────────────────────┐ │
│  │ HashMap<WorkerId, WorkerInfo>                     │ │
│  │                                                   │ │
│  │ worker-abc123 → {                                 │ │
│  │   url: "http://127.0.0.1:8081",                  │ │
│  │   state: Idle,                                   │ │
│  │   failed_health_checks: 0,  ← TEAM-096          │ │
│  │   // pid: MISSING!          ← Critical gap      │ │
│  │ }                                                 │ │
│  └───────────────────────────────────────────────────┘ │
│                        ▲                                │
│                        │                                │
│  ┌─────────────────────┴─────────────────────────────┐ │
│  │ Lifecycle Manager                                 │ │
│  │  - Spawn: Adds to registry                        │ │
│  │  - Health: Updates failed_health_checks           │ │
│  │  - Timeout: Removes from registry                 │ │
│  │  - Fail-fast: Removes after 3 failures            │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## API Methods

### Core Operations
```rust
// Create new registry
pub fn new() -> Self

// Register worker
pub async fn register(&self, worker: WorkerInfo)

// Update worker state
pub async fn update_state(&self, worker_id: &str, state: WorkerState)

// Get worker by ID
pub async fn get(&self, worker_id: &str) -> Option<WorkerInfo>

// List all workers
pub async fn list(&self) -> Vec<WorkerInfo>

// Remove worker
pub async fn remove(&self, worker_id: &str) -> Option<WorkerInfo>

// Clear all workers (shutdown)
pub async fn clear(&self)
```

### Lifecycle Operations (TEAM-096)
```rust
// Increment failed health checks (fail-fast protocol)
pub async fn increment_failed_health_checks(&self, worker_id: &str) -> Option<u32>

// Find idle worker for model
pub async fn find_idle_worker(&self, model_ref: &str) -> Option<WorkerInfo>

// Get idle workers (for timeout enforcement)
pub async fn get_idle_workers(&self) -> Vec<WorkerInfo>

// Find worker by node and model (ephemeral mode)
pub async fn find_by_node_and_model(&self, node: &str, model: &str) -> Option<WorkerInfo>
```

## Tight Coupling with Lifecycle

### 1. Worker Spawn
```rust
// http/workers.rs - handle_spawn_worker()
let worker = WorkerInfo {
    id: worker_id.clone(),
    state: WorkerState::Loading,
    failed_health_checks: 0,  // TEAM-096
    ...
};

// Register immediately after spawn
registry.register(worker).await;

// ❌ PROBLEM: child.id() not stored!
```

### 2. Health Monitoring (TEAM-096)
```rust
// monitor.rs - health_monitor_loop()
for worker in registry.list().await {
    match health_check(&worker.url).await {
        Ok(_) => {
            // Reset counter on success
            registry.update_state(&worker.id, worker.state).await;
        }
        Err(_) => {
            // Increment counter, remove after 3 failures
            let count = registry.increment_failed_health_checks(&worker.id).await;
            if count >= 3 {
                registry.remove(&worker.id).await;
            }
        }
    }
}
```

### 3. Idle Timeout (TEAM-027)
```rust
// timeout.rs - idle_timeout_loop()
for worker in registry.get_idle_workers().await {
    if idle_duration(&worker) > 5.minutes() {
        // Send shutdown request
        shutdown_worker(&worker.url).await;
        // Remove from registry
        registry.remove(&worker.id).await;
    }
}
```

### 4. Worker Ready Callback
```rust
// http/workers.rs - handle_worker_ready()
registry.update_state(&worker_id, WorkerState::Idle).await;
```

## Concurrency

**Thread-Safe:** Uses `Arc<RwLock<HashMap>>`

```rust
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
}
```

## Integration Points

### HTTP API
- `POST /v1/workers/spawn` → Registers worker
- `POST /v1/workers/ready` → Updates state to Idle
- `GET /v1/workers/list` → Returns registry contents

### Lifecycle Loops
- **Health Monitor** (30s) → Updates `failed_health_checks`
- **Idle Timeout** (60s) → Checks `last_activity`

### Queen-rbee Callback
- Worker calls queen's `/v1/workers/ready`
- Queen registers in its own registry

## Maturity Assessment

**Status:** 🟡 **FUNCTIONAL BUT INCOMPLETE**

**Strengths:**
- ✅ Thread-safe concurrent access
- ✅ Complete CRUD operations
- ✅ Fail-fast protocol (TEAM-096)
- ✅ Idle timeout support (TEAM-027)
- ✅ Tight lifecycle coupling
- ✅ State management

**Critical Gaps:**
- ❌ **No PID tracking** - Can't send signals to workers
- ❌ **No force kill** - Can't terminate hung workers
- ❌ **No process liveness check** - Relies only on HTTP
- ❌ **No restart policy** - Workers don't auto-restart on crash

**Recommended Improvements:**
1. **P0:** Add `pid: Option<u32>` field to WorkerInfo
2. **P0:** Store child.id() during spawn
3. **P0:** Add force_kill() method using PID
4. **P1:** Add process liveness checks (wait4/pidfd)
5. **P1:** Add restart policy (exponential backoff)
6. **P1:** Add heartbeat mechanism (faster than 30s health checks)
7. **P2:** Add metrics (spawn count, crash count, uptime)

## Comparison: Hive vs Queen Worker Registry

| Feature | Hive Registry | Queen Registry |
|---------|--------------|----------------|
| **Scope** | This hive only | All hives |
| **Lifecycle coupling** | ✅ Tight | ❌ Loose |
| **Health checks** | ✅ Yes (30s) | ❌ No |
| **Fail-fast** | ✅ Yes (TEAM-096) | ❌ No |
| **PID tracking** | ❌ No (gap!) | ❌ No |
| **Idle timeout** | ✅ Yes (5min) | ❌ No |
| **Node mapping** | N/A | ✅ Yes |

## Testing

```bash
# Unit tests
cargo test -p rbee-hive registry

# Integration test
cargo run -p rbee-hive -- daemon &
curl -X POST http://localhost:8081/v1/workers/spawn \
  -H "Content-Type: application/json" \
  -d '{"model_ref": "hf:model", "backend": "cpu", "device": 0}'
curl http://localhost:8081/v1/workers/list
```

## Related Components

- **Worker Lifecycle** - Spawn, health, timeout (tightly coupled)
- **Queen Worker Registry** - Aggregates workers from all hives
- **HTTP API** (`http/workers.rs`) - Worker management endpoints
- **Monitor** (`monitor.rs`) - Health checks, fail-fast
- **Timeout** (`timeout.rs`) - Idle timeout enforcement

---

**Created by:** TEAM-026  
**Enhanced by:** TEAM-027 (timeout), TEAM-030 (ephemeral), TEAM-096 (fail-fast)  
**Last Updated:** 2025-10-18  
**Maturity:** 🟡 Functional but incomplete (PID tracking critical gap)
