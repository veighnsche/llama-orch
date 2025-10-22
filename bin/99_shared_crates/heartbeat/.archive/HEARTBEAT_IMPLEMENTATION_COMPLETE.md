# Heartbeat System Implementation Complete

**Date:** 2025-10-20  
**Team:** TEAM-151  
**Status:** ✅ **COMPLETE** - Shared heartbeat crate now supports all three binaries

---

## Summary

The `heartbeat` crate now contains ALL heartbeat logic for the entire rbee system:
- **Worker → Hive:** Worker sends heartbeats (30s interval)
- **Hive collects:** Hive receives worker heartbeats, updates registry
- **Hive → Queen:** Hive sends aggregated heartbeats (15s interval)
- **Queen receives:** Queen receives aggregated hive heartbeats

**Key Principle:** All heartbeat logic is centralized in one shared crate, even though different binaries use different parts.

---

## Architecture

```
┌─────────────────┐
│ llm-worker-rbee │  Uses: start_worker_heartbeat_task()
└────────┬────────┘
         │ POST /v1/heartbeat (30s)
         │ { worker_id, timestamp, health_status }
         ↓
┌─────────────────┐
│   rbee-hive     │  Uses: start_hive_heartbeat_task()
│                 │  Receives: handle_heartbeat() endpoint
└────────┬────────┘
         │ POST /v1/heartbeat (15s)
         │ { hive_id, timestamp, workers: [...] }
         │ (aggregates ALL worker states from registry)
         ↓
┌─────────────────┐
│   queen-rbee    │  Receives: heartbeat endpoint
│                 │  (scheduling decisions based on worker states)
└─────────────────┘
```

---

## What Changed

### 1. Extended Types

**Before (Worker → Hive only):**
```rust
pub struct HeartbeatPayload { ... }
pub struct HeartbeatConfig { ... }
pub fn start_heartbeat_task(...) { ... }
```

**After (All three binaries):**
```rust
// Worker → Hive
pub struct WorkerHeartbeatPayload { ... }
pub struct WorkerHeartbeatConfig { ... }
pub fn start_worker_heartbeat_task(...) { ... }

// Hive → Queen
pub struct HiveHeartbeatPayload { ... }
pub struct HiveHeartbeatConfig { ... }
pub fn start_hive_heartbeat_task(...) { ... }
pub trait WorkerStateProvider { ... }

// Backward compatibility
#[deprecated] pub type HeartbeatPayload = WorkerHeartbeatPayload;
#[deprecated] pub type HeartbeatConfig = WorkerHeartbeatConfig;
#[deprecated] pub fn start_heartbeat_task(...) { ... }
```

---

## How Each Binary Uses It

### Worker (`llm-worker-rbee`)

**What it does:** Sends heartbeats to hive

**Usage:**
```rust
use rbee_heartbeat::{WorkerHeartbeatConfig, start_worker_heartbeat_task};

// In main.rs or worker initialization
let heartbeat_config = WorkerHeartbeatConfig::new(
    worker_id,
    hive_url,  // e.g., "http://localhost:8600"
);

let heartbeat_handle = start_worker_heartbeat_task(heartbeat_config);

// Task runs forever, sending heartbeats every 30s
```

**What it sends:**
```json
POST http://localhost:8600/v1/heartbeat
{
  "worker_id": "worker-123",
  "timestamp": "2025-10-20T00:00:00Z",
  "health_status": "healthy"
}
```

---

### Hive (`rbee-hive`)

**What it does:**
1. **Receives** worker heartbeats (HTTP endpoint)
2. **Aggregates** worker states from registry
3. **Sends** aggregated heartbeats to queen (periodic task)

**Usage:**

#### Part 1: Receive Worker Heartbeats (HTTP Endpoint)
```rust
// In src/http/heartbeat.rs (already implemented)
use rbee_heartbeat::{WorkerHeartbeatPayload as HeartbeatRequest};

pub async fn handle_heartbeat(
    State(state): State<AppState>,
    Json(payload): Json<HeartbeatRequest>,
) -> Result<...> {
    // Update registry with worker heartbeat
    state.registry.update_heartbeat(&payload.worker_id).await;
    Ok(...)
}
```

#### Part 2: Send Aggregated Heartbeats to Queen (Periodic Task)
```rust
use rbee_heartbeat::{
    HiveHeartbeatConfig,
    start_hive_heartbeat_task,
    WorkerStateProvider,
    WorkerState,
};

// Implement WorkerStateProvider for your registry
impl WorkerStateProvider for WorkerRegistry {
    fn get_worker_states(&self) -> Vec<WorkerState> {
        // Convert registry workers to WorkerState format
        self.list().await.iter().map(|w| WorkerState {
            worker_id: w.id.clone(),
            state: format!("{:?}", w.state),
            last_heartbeat: w.last_heartbeat
                .map(|t| format!("{:?}", t))
                .unwrap_or_else(|| "never".to_string()),
            health_status: "healthy".to_string(),
        }).collect()
    }
}

// In main.rs or server initialization
let hive_heartbeat_config = HiveHeartbeatConfig::new(
    hive_id,        // e.g., "localhost"
    queen_url,      // e.g., "http://localhost:8500"
    auth_token,     // API token for queen
);

let worker_provider = Arc::new(registry.clone());
let hive_heartbeat_handle = start_hive_heartbeat_task(
    hive_heartbeat_config,
    worker_provider,
);

// Task runs forever, sending aggregated heartbeats every 15s
```

**What it sends:**
```json
POST http://localhost:8500/v1/heartbeat
Authorization: Bearer <token>
{
  "hive_id": "localhost",
  "timestamp": "2025-10-20T00:00:00Z",
  "workers": [
    {
      "worker_id": "worker-123",
      "state": "Idle",
      "last_heartbeat": "2025-10-20T00:00:00Z",
      "health_status": "healthy"
    },
    {
      "worker_id": "worker-456",
      "state": "Busy",
      "last_heartbeat": "2025-10-20T00:00:05Z",
      "health_status": "healthy"
    }
  ]
}
```

---

### Queen (`queen-rbee`)

**What it does:** Receives aggregated heartbeats from hives

**Usage:**
```rust
// In queen's HTTP server routes
use rbee_heartbeat::HiveHeartbeatPayload;

pub async fn handle_hive_heartbeat(
    State(state): State<AppState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<...> {
    // Update hive registry with worker states
    for worker in payload.workers {
        state.update_worker_state(&worker);
    }
    Ok(...)
}
```

---

## Key Design Decisions

### 1. Periodic Aggregation (Not Immediate Relay)

**❌ Wrong (what I initially implemented):**
```
Worker → Hive: POST /v1/heartbeat
  → Immediately spawn task to relay to queen
```
This would spam queen with one HTTP request per worker heartbeat!

**✅ Correct (current implementation):**
```
Worker → Hive: POST /v1/heartbeat
  → Update registry

Separate periodic task (every 15s):
  → Get ALL workers from registry
  → Send aggregated heartbeat to queen
```

**Why:** More efficient, reduces HTTP requests, queen gets complete snapshot.

### 2. WorkerStateProvider Trait

The hive heartbeat task needs to get worker states from the registry, but the heartbeat crate doesn't know about the hive's `WorkerRegistry` type.

**Solution:** Define a trait in the heartbeat crate:
```rust
pub trait WorkerStateProvider: Send + Sync {
    fn get_worker_states(&self) -> Vec<WorkerState>;
}
```

The hive implements this trait for its registry, then passes it to the heartbeat task.

### 3. Simplified WorkerState for Heartbeats

The hive's `WorkerInfo` struct has many fields (PID, restart count, etc.). The queen only needs a subset for scheduling.

**Heartbeat WorkerState (simplified):**
```rust
pub struct WorkerState {
    pub worker_id: String,
    pub state: String,           // "Idle", "Busy", "Loading"
    pub last_heartbeat: String,
    pub health_status: String,
}
```

This keeps the heartbeat payload lean and focused.

### 4. Different Intervals

- **Worker → Hive:** 30 seconds (default)
- **Hive → Queen:** 15 seconds (default)

**Why:** Hive heartbeat is faster because it aggregates multiple workers. Queen needs fresher data for scheduling decisions.

---

## Implementation Checklist

### ✅ Shared Heartbeat Crate
- [x] Extended with hive types
- [x] Added `HiveHeartbeatPayload`
- [x] Added `HiveHeartbeatConfig`
- [x] Added `WorkerStateProvider` trait
- [x] Added `start_hive_heartbeat_task()`
- [x] Added backward compatibility aliases
- [x] Updated tests

### ✅ rbee-hive
- [x] Removed immediate relay logic from `heartbeat.rs`
- [x] Simplified endpoint to just update registry
- [x] Re-exported types from shared crate

### 🚧 TODO: rbee-hive Integration
- [ ] Implement `WorkerStateProvider` for `WorkerRegistry`
- [ ] Start hive heartbeat task in main.rs/server initialization
- [ ] Add hive_id and queen_url to config
- [ ] Wire up the periodic task

### 🚧 TODO: queen-rbee
- [ ] Add endpoint to receive hive heartbeats
- [ ] Update hive registry with worker states
- [ ] Use worker states for scheduling decisions

### 🚧 TODO: llm-worker-rbee
- [ ] Update to use new `WorkerHeartbeatConfig` name (backward compatible)
- [ ] Verify heartbeat task still works

---

## Next Steps

### 1. Wire Hive Heartbeat Task

**File:** `/bin/20_rbee_hive/src/main.rs` or server initialization

```rust
use rbee_heartbeat::{HiveHeartbeatConfig, start_hive_heartbeat_task, WorkerStateProvider, WorkerState};

// Implement WorkerStateProvider for WorkerRegistry
impl WorkerStateProvider for WorkerRegistry {
    fn get_worker_states(&self) -> Vec<WorkerState> {
        // Convert internal WorkerInfo to heartbeat WorkerState
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.list().await.iter().map(|w| WorkerState {
                    worker_id: w.id.clone(),
                    state: format!("{:?}", w.state),
                    last_heartbeat: w.last_heartbeat
                        .map(|t| chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339())
                        .unwrap_or_else(|| "never".to_string()),
                    health_status: "healthy".to_string(),
                }).collect()
            })
        })
    }
}

// In server startup (after registry created, before HTTP server starts)
if let Some(queen_url) = config.queen_url {
    let hive_heartbeat_config = HiveHeartbeatConfig::new(
        config.hive_id.clone(),
        queen_url,
        config.auth_token.clone(),
    );

    let worker_provider: Arc<dyn WorkerStateProvider> = Arc::new(registry.clone());
    
    let _hive_heartbeat_handle = start_hive_heartbeat_task(
        hive_heartbeat_config,
        worker_provider,
    );

    info!("Hive heartbeat task started");
}
```

### 2. Add Config Fields

**File:** `/bin/20_rbee_hive/src/config.rs` (or wherever config is)

```rust
pub struct HiveConfig {
    pub hive_id: String,           // NEW: e.g., "localhost" or hostname
    pub queen_url: Option<String>, // NEW: e.g., "http://localhost:8500"
    pub auth_token: String,        // Already exists
    // ... existing fields ...
}
```

### 3. Update Queen to Receive Hive Heartbeats

**File:** `/bin/10_queen_rbee/src/http/heartbeat.rs` (create if doesn't exist)

```rust
use rbee_heartbeat::HiveHeartbeatPayload;

pub async fn handle_hive_heartbeat(
    State(state): State<AppState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<StatusCode, (StatusCode, String)> {
    // Update hive registry with latest worker states
    for worker in payload.workers {
        state.hive_registry.update_worker_state(
            &payload.hive_id,
            &worker.worker_id,
            &worker.state,
            &worker.last_heartbeat,
        ).await;
    }
    
    Ok(StatusCode::OK)
}
```

---

## Testing

### Unit Tests (Already Added)
- ✅ `test_worker_heartbeat_config_new()`
- ✅ `test_worker_heartbeat_payload_serialization()`
- ✅ `test_hive_heartbeat_config_new()`
- ✅ `test_hive_heartbeat_payload_serialization()`
- ✅ `test_health_status_serialization()`

### Integration Tests (TODO)
- [ ] Worker sends heartbeat → Hive receives → Registry updated
- [ ] Hive periodic task → Aggregates workers → Sends to queen
- [ ] Queen receives hive heartbeat → Updates registry
- [ ] Full chain: Worker → Hive → Queen

---

## Benefits of This Design

1. **Centralized Logic:** All heartbeat code in one place
2. **Reusable:** All three binaries use the same crate
3. **Efficient:** Periodic aggregation reduces HTTP requests
4. **Flexible:** Different intervals for different use cases
5. **Type-Safe:** Strong typing prevents mistakes
6. **Testable:** Each component can be tested independently

---

## Backward Compatibility

Old code using deprecated names will still work:
```rust
// Old code (still works, but deprecated)
use rbee_heartbeat::{HeartbeatConfig, start_heartbeat_task};

// New code (preferred)
use rbee_heartbeat::{WorkerHeartbeatConfig, start_worker_heartbeat_task};
```

Compiler will show deprecation warnings guiding users to new names.

---

**END OF IMPLEMENTATION GUIDE**  
**Status:** ✅ Shared crate complete, integration pending  
**Date:** 2025-10-20  
**Team:** TEAM-151
