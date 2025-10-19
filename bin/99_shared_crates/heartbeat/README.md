# rbee-heartbeat

**Status:** 🚧 STUB (Created by TEAM-135)  
**Purpose:** Generic heartbeat protocol for health monitoring  
**Location:** `bin/shared-crates/heartbeat/` (SHARED - used by multiple binaries)

---

## Overview

The `rbee-heartbeat` crate provides a **generic heartbeat mechanism** for health monitoring in the llama-orch system. It is used by both **workers** and **hives** to send periodic heartbeat messages to their parent components.

### System Context

In the llama-orch architecture, there is a **two-level heartbeat chain**:

```
┌─────────────────┐
│ llm-worker-rbee │  ← Worker process (USES THIS CRATE)
│  (worker-orcd)  │
└────────┬────────┘
         │
         │ Heartbeat #1: Worker → Hive (30s interval)
         │ POST /v1/heartbeat
         │ Payload: { worker_id, timestamp, health_status }
         ↓
┌─────────────────┐
│   rbee-hive     │  ← Pool manager (USES THIS CRATE)
│ (pool-managerd) │  ← Receives worker heartbeats + sends own heartbeat
└────────┬────────┘
         │
         │ Heartbeat #2: Hive → Queen (15s interval)
         │ POST /v2/pools/{id}/heartbeat
         │ Payload: { pool_id, gpus[], workers[], timestamp }
         │ (aggregates ALL worker heartbeats + GPU state)
         ↓
┌─────────────────┐
│   queen-rbee    │  ← Orchestrator (receives aggregated heartbeats)
│ (orchestratord) │  ← Uses for scheduling decisions
└─────────────────┘
         
         NO heartbeat to rbee-keeper (not needed)
```

### Key Insight: Two Different Use Cases

**Use Case 1: Worker → Hive (Heartbeat #1)**
- **Who:** `llm-worker-rbee` uses this crate
- **To:** `rbee-hive` (pool manager)
- **Interval:** 30 seconds (default)
- **Payload:** Simple worker health (`worker_id`, `timestamp`, `health_status`)
- **Purpose:** Indicate worker is alive and ready for inference

**Use Case 2: Hive → Queen (Heartbeat #2)**
- **Who:** `rbee-hive` uses this crate
- **To:** `queen-rbee` (orchestrator)
- **Interval:** 15 seconds (default)
- **Payload:** Aggregated state (`pool_id`, `gpus[]`, `workers[]`, `timestamp`)
- **Purpose:** Report complete pool state (GPU VRAM + all worker states)
- **Special:** Hive collects worker heartbeats asynchronously, then sends aggregated snapshot to queen

### Why This is a Shared Crate

This crate is in `shared-crates/` because:
1. **Workers** send heartbeats to hives
2. **Hives** send heartbeats to queen
3. Both use the same underlying mechanism (periodic HTTP POST with retry logic)
4. Code reuse: ~200 LOC shared instead of duplicated

**Key Responsibilities:**
- Provide generic heartbeat task spawning
- Handle periodic interval timing
- HTTP POST with timeout and error handling
- Non-fatal failure handling (log but don't crash)
- Configurable intervals and payloads

---

## Architecture Principles

### 1. Generic Heartbeat Mechanism

This crate provides a **generic heartbeat sender** that:
- Spawns background task for periodic heartbeats
- Handles HTTP POST with configurable payload
- Logs errors but never crashes (non-fatal failures)
- Works for ANY component that needs to send heartbeats

### 2. Non-Fatal Failures

Heartbeat failures are **logged but non-fatal**:
- Sender continues running even if heartbeat fails
- Receiver detects stale senders via missed heartbeats
- Temporary network issues don't crash the process

### 3. Asynchronous Collection + Synchronous Aggregation

**Workers send heartbeats asynchronously:**
- Each worker sends heartbeat independently (30s interval)
- Workers don't coordinate with each other
- rbee-hive receives worker heartbeats at different times

**Hive aggregates synchronously:**
- rbee-hive collects all worker heartbeats in memory
- Every 15 seconds, hive takes a **snapshot** of current state
- Snapshot includes: GPU VRAM state + all worker states (from their last heartbeat)
- Hive sends aggregated snapshot to queen-rbee

**Why this works:**
- Workers heartbeat more frequently than needed (30s)
- Hive aggregates less frequently (15s)
- Queen always gets fresh data (worker heartbeats are <45s old)
- No synchronization needed between workers

---

## Protocol Specification

This crate supports **two different heartbeat protocols** (same mechanism, different payloads).

### Heartbeat #1: Worker → Hive

Workers send heartbeats via `POST /v1/heartbeat` to rbee-hive:

```json
{
  "worker_id": "worker-abc123",
  "timestamp": "2025-10-19T12:34:56Z",
  "health_status": "healthy"
}
```

**Fields:**
- `worker_id` (string): Unique worker identifier
- `timestamp` (string): ISO 8601 timestamp
- `health_status` (enum): `"healthy"` or `"degraded"`

**Timing:**
- **Interval:** 30 seconds (default, configurable)
- **Timeout:** 5 seconds per request
- **Failure behavior:** Log warning, continue sending

---

### Heartbeat #2: Hive → Queen

Hives send heartbeats via `POST /v2/pools/{id}/heartbeat` to queen-rbee:

```json
{
  "pool_id": "pool-1",
  "timestamp": "2025-10-19T12:34:56Z",
  "gpus": [
    {
      "id": 0,
      "device_name": "NVIDIA RTX 4090",
      "total_vram_bytes": 24000000000,
      "free_vram_bytes": 12000000000
    }
  ],
  "workers": [
    {
      "id": "worker-abc123",
      "status": "idle",
      "model_ref": "llama-7b",
      "vram_bytes": 8000000000,
      "gpu_id": 0,
      "uri": "http://localhost:8001",
      "capabilities": ["text-gen"],
      "protocol": "sse"
    }
  ]
}
```

**Fields:**
- `pool_id` (string): Unique pool identifier
- `timestamp` (string): ISO 8601 timestamp
- `gpus` (array): GPU state (VRAM totals/free)
- `workers` (array): All worker states (aggregated from worker heartbeats)

**Timing:**
- **Interval:** 15 seconds (default, configurable)
- **Timeout:** 5 seconds per request
- **Failure behavior:** Log warning, continue sending

**Aggregation Logic:**
- Hive maintains in-memory registry of all workers
- Each worker heartbeat updates the registry
- Every 15s, hive takes snapshot of registry + GPU state
- Snapshot is sent to queen-rbee

---

### Health Status (Worker Heartbeats Only)

- **`healthy`**: Worker is operating normally, ready for inference
- **`degraded`**: Worker is experiencing issues (e.g., high memory usage) but still functional

---

## API Design

### Core Components

This crate provides a **generic heartbeat sender** with these components:

1. **`HeartbeatConfig<T>`**: Generic configuration for heartbeat task
   - `sender_id`: Identifier of the sender (worker_id or pool_id)
   - `callback_url`: URL to send heartbeats to
   - `interval_secs`: Heartbeat interval
   - `payload_builder`: Function to build payload of type `T`

2. **`start_heartbeat_task<T>()`**: Spawns background heartbeat task
   - Generic over payload type `T: Serialize`
   - Returns `JoinHandle` for task management
   - Runs forever until shutdown
   - Logs errors but never crashes

### Example Usage: Worker → Hive

```rust
use rbee_heartbeat::{HeartbeatConfig, start_heartbeat_task};
use serde::Serialize;

#[derive(Serialize)]
struct WorkerHeartbeat {
    worker_id: String,
    timestamp: String,
    health_status: String,
}

// Configure worker heartbeat
let config = HeartbeatConfig::new(
    "worker-abc123".to_string(),
    "http://localhost:9200/v1/heartbeat".to_string(),
    30, // 30 seconds
    |sender_id| WorkerHeartbeat {
        worker_id: sender_id,
        timestamp: chrono::Utc::now().to_rfc3339(),
        health_status: "healthy".to_string(),
    },
);

// Start background task
let heartbeat_handle = start_heartbeat_task(config);

// Worker continues running...
// On shutdown:
heartbeat_handle.abort();
```

### Example Usage: Hive → Queen

```rust
use rbee_heartbeat::{HeartbeatConfig, start_heartbeat_task};
use serde::Serialize;

#[derive(Serialize)]
struct PoolHeartbeat {
    pool_id: String,
    timestamp: String,
    gpus: Vec<GpuState>,
    workers: Vec<WorkerState>,
}

// Configure hive heartbeat
let registry = Arc::clone(&worker_registry);
let gpu_monitor = Arc::clone(&gpu_monitor);

let config = HeartbeatConfig::new(
    "pool-1".to_string(),
    "http://queen-rbee:8080/v2/pools/pool-1/heartbeat".to_string(),
    15, // 15 seconds
    move |pool_id| {
        // Take snapshot of current state
        let gpus = gpu_monitor.get_all_gpu_states();
        let workers = registry.get_all_worker_states();
        
        PoolHeartbeat {
            pool_id,
            timestamp: chrono::Utc::now().to_rfc3339(),
            gpus,
            workers,
        }
    },
);

// Start background task
let heartbeat_handle = start_heartbeat_task(config);

// Hive continues running...
// On shutdown:
heartbeat_handle.abort();
```

---

## Dependencies

### Required

- **`tokio`**: Async runtime for background task and HTTP client
- **`reqwest`**: HTTP client for sending heartbeats
- **`serde`**: Serialization for JSON payloads
- **`chrono`**: Timestamp generation (ISO 8601)
- **`tracing`**: Structured logging

### Optional

- **`serde_json`**: JSON serialization (via `serde`)

---

## Implementation Status

### Phase 1: Core Functionality
- [ ] `HeartbeatConfig` struct
- [ ] `HeartbeatPayload` struct
- [ ] `HealthStatus` enum
- [ ] `start_heartbeat_task()` function
- [ ] Background task with interval timer
- [ ] HTTP POST to rbee-hive
- [ ] Error handling (log but don't crash)

### Phase 2: Testing
- [ ] Unit tests for config builder
- [ ] Unit tests for payload serialization
- [ ] Integration tests with mock rbee-hive
- [ ] Failure scenario tests (network errors, timeouts)

### Phase 3: Documentation
- [ ] API documentation (rustdoc)
- [ ] Usage examples
- [ ] Integration guide

### Phase 4: Observability
- [ ] Structured logging (tracing)
- [ ] Metrics (heartbeat success/failure counters)
- [ ] Health status reporting

---

## Design Decisions

### 1. Why Background Task?

Heartbeats run as a **background task** (not inline) because:
- Worker must continue serving inference requests during heartbeat
- Heartbeat failures should not block worker operations
- Decouples heartbeat timing from inference timing

### 2. Why Non-Fatal Failures?

Heartbeat failures are **logged but non-fatal** because:
- Temporary network issues should not crash the worker
- rbee-hive detects stale workers via missed heartbeats
- Worker may recover without restart

### 3. Why 30-Second Default?

The 30-second interval balances:
- **Responsiveness**: Detect failures within ~90s (3 missed heartbeats)
- **Overhead**: Minimal HTTP traffic (2 requests/minute per worker)
- **Network tolerance**: Allows for temporary network hiccups

---

## Integration Points

### With `llm-worker-rbee` Binary

The heartbeat crate is used by the main worker binary:

```rust
// In llm-worker-rbee/src/main.rs
use worker_rbee_heartbeat::{HeartbeatConfig, start_heartbeat_task};

#[tokio::main]
async fn main() {
    // ... worker initialization ...
    
    // Start heartbeat
    let heartbeat_config = HeartbeatConfig::new(
        worker_id.clone(),
        hive_callback_url.clone(),
    );
    let _heartbeat_handle = start_heartbeat_task(heartbeat_config);
    
    // Start HTTP server for inference
    // ...
}
```

### With `rbee-hive` HTTP Server

rbee-hive receives heartbeats at `POST /v1/heartbeat`:

```rust
// In rbee-hive/src/http/heartbeat.rs
pub async fn handle_heartbeat(
    State(state): State<AppState>,
    Json(payload): Json<HeartbeatRequest>,
) -> Result<...> {
    // Update worker's last_heartbeat timestamp
    state.registry.update_heartbeat(&payload.worker_id).await;
    // ...
}
```

### With `queen-rbee` Orchestrator

queen-rbee uses heartbeat data (via rbee-hive) for:
- **Scheduling**: Exclude stale workers from placement decisions
- **Monitoring**: Track worker health across the cluster
- **Eviction**: Detect failed workers for cleanup

---

## Specification References

- **SYS-6.2.4**: Heartbeat Protocol (pool-managerd perspective)
- **SYS-6.3.1**: Worker Self-Containment
- **SYS-6.3.4**: Ready Callback Contract (worker registration)
- **SYS-8.3.x**: Reliability requirements

See: `/home/vince/Projects/llama-orch/bin/.specs/00_llama-orch.md`

---

## Migration Notes

### From `llm-worker-rbee.bak/src/heartbeat.rs`

The original implementation (TEAM-115, 182 LOC) provides:
- ✅ Complete heartbeat protocol
- ✅ Background task with interval timer
- ✅ Error handling (non-fatal failures)
- ✅ Unit tests for config and serialization

**Migration steps:**
1. Copy `heartbeat.rs` to `src/lib.rs`
2. Update module structure for library crate
3. Add public API documentation
4. Update dependencies in `Cargo.toml`
5. Run tests to verify functionality

---

## Testing Strategy

### Unit Tests

- Config builder (default values, custom interval)
- Payload serialization (JSON format, field names)
- Health status enum (serialization to `"healthy"`/`"degraded"`)

### Integration Tests

- Mock rbee-hive server (verify HTTP POST format)
- Heartbeat interval timing (verify periodic sending)
- Failure scenarios (network errors, timeouts, 404 responses)

### Property Tests

- Timestamp format (always valid ISO 8601)
- Payload size (reasonable bounds)

---

## Performance Considerations

### Overhead

- **Network:** 2 HTTP requests/minute per worker (~200 bytes each)
- **CPU:** Negligible (background task sleeps between heartbeats)
- **Memory:** ~1KB per worker (config + HTTP client)

### Scalability

- **100 workers:** 200 requests/minute to rbee-hive (~3 req/sec)
- **1000 workers:** 2000 requests/minute (~33 req/sec)

rbee-hive aggregates heartbeats and sends summary to queen-rbee (default: 15s interval).

---

## Future Enhancements

### Phase 1 (M1)
- [ ] Configurable timeout per heartbeat
- [ ] Exponential backoff on repeated failures
- [ ] Metrics emission (success/failure counters)

### Phase 2 (M2)
- [ ] Health status auto-detection (memory usage, CPU load)
- [ ] Graceful shutdown (send final heartbeat)
- [ ] Correlation ID propagation

### Phase 3 (M3+)
- [ ] Multi-hive support (send heartbeats to multiple hives)
- [ ] Heartbeat batching (reduce HTTP overhead)
- [ ] TLS support for secure communication

---

## Related Crates

### Used By
- **`llm-worker-rbee`**: Uses this crate to send worker heartbeats to rbee-hive
- **`rbee-hive`**: Uses this crate to send pool heartbeats to queen-rbee

### Integrates With
- **`rbee-hive-crates/worker-registry`**: Stores worker state, updated by worker heartbeats
- **`rbee-hive-crates/http-server`**: Receives worker heartbeats, sends pool heartbeats
- **`queen-rbee-crates/pool-registry`**: Stores pool state, updated by pool heartbeats
- **`rbee-hive-crates/device-detection`**: Detects GPU capabilities, reported in pool heartbeats

---

## Crate Location

**Location:** `bin/shared-crates/heartbeat/` ✅

**Why shared:** Both workers AND hives use this crate to send heartbeats, so it belongs in `shared-crates/` (not `worker-rbee-crates/` or `rbee-hive-crates/`).

---

## Team History

- **TEAM-115**: Original implementation in `llm-worker-rbee.bak/src/heartbeat.rs`
- **TEAM-135**: Scaffolding for new crate-based architecture

---

**Next Steps:**
1. ✅ **Move crate** from `worker-rbee-crates/` to `shared-crates/` (DONE)
2. Migrate implementation from `llm-worker-rbee.bak/src/heartbeat.rs` to this crate
3. Make API generic (support both worker and pool payloads via `HeartbeatConfig<T>`)
4. Implement hive → queen heartbeat (currently missing in `.bak`)
5. Update `llm-worker-rbee` to use this crate (worker heartbeats)
6. Update `rbee-hive` to use this crate (pool heartbeats)
7. Add comprehensive tests for both use cases

---

## Key Design Insight: Why Aggregation Works

**The Problem:**
- Workers send heartbeats asynchronously (each on their own 30s timer)
- Hive needs to send aggregated state to queen (every 15s)
- How does hive get consistent snapshots if workers are unsynchronized?

**The Solution:**
1. **Workers heartbeat frequently** (30s) → Hive always has recent data
2. **Hive aggregates infrequently** (15s) → Takes snapshot of current state
3. **Queen tolerates staleness** (up to 45s) → 3 missed heartbeats = stale

**Example Timeline:**
```
T=0s:   Worker A heartbeat → Hive (updates registry)
T=5s:   Worker B heartbeat → Hive (updates registry)
T=15s:  Hive heartbeat → Queen (snapshot: A=0s old, B=10s old) ✅
T=30s:  Worker A heartbeat → Hive (updates registry)
T=30s:  Hive heartbeat → Queen (snapshot: A=0s old, B=25s old) ✅
T=35s:  Worker B heartbeat → Hive (updates registry)
T=45s:  Hive heartbeat → Queen (snapshot: A=15s old, B=10s old) ✅
T=60s:  Worker A heartbeat → Hive (updates registry)
T=60s:  Hive heartbeat → Queen (snapshot: A=0s old, B=25s old) ✅
```

**Why this works:**
- Worker heartbeats (30s) < Hive aggregation (15s) × 3 = 45s timeout
- Queen always sees worker data that's <45s old
- No coordination needed between workers
- Hive just takes snapshots of whatever it has

This is the **beauty of asynchronous collection + synchronous aggregation**!
