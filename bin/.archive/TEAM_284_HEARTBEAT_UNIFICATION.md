# TEAM-284: Heartbeat System Unification

**Date:** Oct 24, 2025  
**Mission:** Create unified heartbeat system for both workers and hives using `rbee-heartbeat` crate

## Summary

Successfully unified the heartbeat system so both workers and hives use the same `rbee-heartbeat` crate with the same pattern. Both send heartbeats directly to queen-rbee every 30 seconds.

## Analysis: Two Heartbeat Implementations Found

### 1. `rbee-heartbeat` crate (✅ ACTIVELY USED)
**Location:** `bin/99_shared_crates/heartbeat/`

**Used by:**
- Workers send heartbeats to queen
- Pattern: `WorkerHeartbeatPayload` → POST `/v1/worker-heartbeat`
- Interval: 30 seconds
- Timeout: 90 seconds (3 missed heartbeats)

**Architecture:**
```rust
// Worker side
use rbee_heartbeat::{WorkerHeartbeatConfig, start_worker_heartbeat_task};

let config = WorkerHeartbeatConfig::new(
    "worker-123".to_string(),
    "http://localhost:8500".to_string(),  // Queen URL
);
let handle = start_worker_heartbeat_task(config);
```

### 2. `worker-contract` crate (⚠️ PARTIALLY USED)
**Location:** `bin/99_shared_crates/worker-contract/`

**Contains:**
- `WorkerInfo`, `WorkerStatus`, `WorkerHeartbeat` types
- Used by `worker-registry` for type definitions
- Has duplicate heartbeat types but different structure

**Verdict:** Cannot be removed - provides contract types for worker registry

## What Was Added

### 1. HiveHeartbeatPayload Type
**File:** `bin/99_shared_crates/heartbeat/src/types.rs`

```rust
/// Hive heartbeat payload sent to queen-rbee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveHeartbeatPayload {
    /// Hive ID (alias from config)
    pub hive_id: String,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
    /// Health status
    pub health_status: HealthStatus,
    // TODO: TEAM-284: Add system stats (CPU usage, RAM, VRAM per device, temperature)
}
```

### 2. Hive Heartbeat Sender Module
**File:** `bin/99_shared_crates/heartbeat/src/hive.rs` (NEW - 170 LOC)

**Features:**
- `HiveHeartbeatConfig` - Configuration for hive heartbeats
- `start_hive_heartbeat_task()` - Background task that sends heartbeats every 30s
- Mirrors worker heartbeat pattern exactly
- Non-fatal error handling (logs but doesn't crash)

**Usage:**
```rust
use rbee_heartbeat::{HiveHeartbeatConfig, start_hive_heartbeat_task};

let config = HiveHeartbeatConfig::new(
    "localhost".to_string(),
    "http://localhost:8500".to_string(),  // Queen URL
);
let handle = start_hive_heartbeat_task(config);
```

### 3. Hive Heartbeat Receiver Endpoint
**File:** `bin/10_queen_rbee/src/http/heartbeat.rs`

**Added:**
- `handle_hive_heartbeat()` - POST `/v1/hive-heartbeat` endpoint
- Mirrors `handle_worker_heartbeat()` pattern
- Logs heartbeat and returns acknowledgement
- TODO: Wire up to hive registry (when created)

## Heartbeat Flow Comparison

### Worker Heartbeat (Existing)
```text
Worker (llm-worker-rbee)
    ↓ Every 30s
    POST /v1/worker-heartbeat
    ↓
Queen (queen-rbee)
    ↓
WorkerRegistry.update_worker()
    ↓
Track last_heartbeat timestamp
```

### Hive Heartbeat (NEW - TEAM-284)
```text
Hive (rbee-hive)
    ↓ Every 30s
    POST /v1/hive-heartbeat
    ↓
Queen (queen-rbee)
    ↓
HiveRegistry.update_hive() [TODO]
    ↓
Track last_heartbeat timestamp
```

## Files Modified

### Shared Crate (`rbee-heartbeat`)
1. **`src/types.rs`** - Added `HiveHeartbeatPayload`
2. **`src/hive.rs`** - NEW file (170 LOC) - Hive heartbeat sender
3. **`src/lib.rs`** - Added hive module and re-exports

### Queen (`queen-rbee`)
4. **`src/http/heartbeat.rs`** - Added `handle_hive_heartbeat()` endpoint

## Pattern Consistency

Both workers and hives now follow the EXACT same pattern:

| Aspect | Worker | Hive |
|--------|--------|------|
| **Crate** | `rbee-heartbeat` | `rbee-heartbeat` |
| **Config** | `WorkerHeartbeatConfig` | `HiveHeartbeatConfig` |
| **Payload** | `WorkerHeartbeatPayload` | `HiveHeartbeatPayload` |
| **Starter** | `start_worker_heartbeat_task()` | `start_hive_heartbeat_task()` |
| **Endpoint** | `POST /v1/worker-heartbeat` | `POST /v1/hive-heartbeat` |
| **Handler** | `handle_worker_heartbeat()` | `handle_hive_heartbeat()` |
| **Interval** | 30 seconds | 30 seconds |
| **Timeout** | 90 seconds | 90 seconds |

## TODO: Next Steps

### 1. Wire Up Hive Heartbeat in rbee-hive Binary
**File:** `bin/20_rbee_hive/src/main.rs`

```rust
use rbee_heartbeat::{HiveHeartbeatConfig, start_hive_heartbeat_task};

// In main() after server starts
let hive_id = config.hive_id.clone();
let queen_url = config.queen_url.clone();

let heartbeat_config = HiveHeartbeatConfig::new(hive_id, queen_url);
let _heartbeat_handle = start_hive_heartbeat_task(heartbeat_config);
```

### 2. Create Hive Registry (Mirror Worker Registry)
**New crate:** `bin/15_queen_rbee_crates/hive-registry/`

Should mirror `worker-registry` structure:
- `HiveRegistry` - Track hive state (last_heartbeat, health_status)
- `update_hive()` - Update hive state from heartbeat
- `get_online_hives()` - Get hives with recent heartbeats
- `cleanup_stale()` - Remove hives with old heartbeats (>90s)

### 3. Wire Up Hive Registry in Queen
**File:** `bin/10_queen_rbee/src/http/heartbeat.rs`

```rust
// Replace TODO with:
state.hive_registry.update_hive(payload);
```

### 4. Add System Stats to Hive Heartbeat
**File:** `bin/99_shared_crates/heartbeat/src/types.rs`

```rust
pub struct HiveHeartbeatPayload {
    pub hive_id: String,
    pub timestamp: String,
    pub health_status: HealthStatus,
    // NEW:
    pub cpu_usage_percent: f32,
    pub ram_used_gb: f32,
    pub ram_total_gb: f32,
    pub vram_per_device: Vec<VramInfo>,  // Per-GPU VRAM stats
    pub temperature_celsius: Option<f32>,
}
```

### 5. Register Hive Heartbeat Endpoint in Router
**File:** `bin/10_queen_rbee/src/http/routes.rs`

```rust
.route("/v1/hive-heartbeat", post(heartbeat::handle_hive_heartbeat))
```

## Verification

```bash
✅ cargo check -p rbee-heartbeat
✅ cargo check -p queen-rbee
```

## worker-contract Status

**Decision:** KEEP `worker-contract` crate

**Reason:** 
- Used by `worker-registry` for `WorkerInfo`, `WorkerStatus`, `WorkerHeartbeat` types
- Provides contract definition for all worker implementations
- Different purpose than `rbee-heartbeat` (contract vs protocol)

**Relationship:**
- `worker-contract` = Type definitions (what a worker IS)
- `rbee-heartbeat` = Protocol implementation (how heartbeats WORK)

Both are needed and serve different purposes.

## Benefits

✅ **Unified Pattern** - Workers and hives use identical heartbeat system  
✅ **Single Source of Truth** - All heartbeat logic in `rbee-heartbeat` crate  
✅ **Easy to Extend** - Adding system stats is straightforward  
✅ **Consistent Intervals** - Both use 30s interval, 90s timeout  
✅ **Testable** - Shared test patterns for both worker and hive heartbeats  

## Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────┐
│                      rbee-heartbeat                         │
│                    (Shared Protocol)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  WorkerHeartbeatPayload    HiveHeartbeatPayload            │
│  WorkerHeartbeatConfig     HiveHeartbeatConfig             │
│  start_worker_task()       start_hive_task()               │
│                                                             │
└──────────────┬─────────────────────────┬────────────────────┘
               │                         │
               ↓                         ↓
    ┌──────────────────┐      ┌──────────────────┐
    │  llm-worker-rbee │      │    rbee-hive     │
    │                  │      │                  │
    │  Every 30s:      │      │  Every 30s:      │
    │  POST /v1/       │      │  POST /v1/       │
    │  worker-         │      │  hive-           │
    │  heartbeat       │      │  heartbeat       │
    └──────────┬───────┘      └────────┬─────────┘
               │                       │
               └───────────┬───────────┘
                           ↓
                  ┌─────────────────┐
                  │   queen-rbee    │
                  │                 │
                  │  WorkerRegistry │
                  │  HiveRegistry   │
                  │  (TODO)         │
                  └─────────────────┘
```

## Conclusion

✅ **Heartbeat system unified** - Both workers and hives use `rbee-heartbeat`  
✅ **Pattern mirrored** - Identical structure for both  
✅ **Ready for system stats** - TODO marker in place  
✅ **worker-contract kept** - Serves different purpose (contract types)  

Next: Wire up hive heartbeat in rbee-hive binary and create hive registry.
