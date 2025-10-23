# queen-rbee-worker-registry

**Status:** ✅ IMPLEMENTED  
**Purpose:** In-memory registry for tracking real-time runtime state of all workers

**History:**
- TEAM-262: Renamed from `hive-registry` to `worker-registry`
- TEAM-261: Simplified - workers send heartbeats directly to queen

## Overview

In-memory (RAM) registry for tracking real-time runtime state of all workers across all hives.

This is DIFFERENT from `hive-catalog` (SQLite - persistent storage):
- **Catalog** = Persistent config (host, port, SSH, device capabilities)
- **Registry** = Runtime state (workers, VRAM usage, last heartbeat)

## Features

- ✅ Thread-safe (RwLock for concurrent access)
- ✅ Fast lookups for scheduling decisions
- ✅ Heartbeat processing and liveness tracking
- ✅ Resource usage calculation (VRAM/RAM)
- ✅ Active hive filtering
- ✅ Comprehensive test coverage

## Usage

```rust
use queen_rbee_worker_registry::WorkerRegistry;
use rbee_heartbeat::WorkerHeartbeatPayload;

let registry = WorkerRegistry::new();

// Update from worker heartbeat (TEAM-261: workers send directly to queen)
let payload = WorkerHeartbeatPayload {
    worker_id: "worker-123".to_string(),
    timestamp: "2025-10-21T10:00:00Z".to_string(),
    health_status: HealthStatus::Healthy,
};
// Note: Internal API still uses hive_id for grouping
registry.update_hive_state("localhost", payload);

// Check if hive is online (heartbeat in last 30 seconds)
let is_online = registry.is_hive_online("localhost", 30_000);

// Get active hives for scheduling
let active_hives = registry.list_active_hives(30_000);

// Get resource usage
if let Some(resources) = registry.get_available_resources("localhost") {
    println!("Workers: {}, VRAM: {}GB", 
        resources.worker_count, 
        resources.vram_used_gb);
}
```

## Public API

- `update_hive_state()` - Process heartbeat and update state
- `get_hive_state()` - Get current runtime state
- `list_active_hives()` - Get hives with recent heartbeat
- `get_available_resources()` - Get resource usage for scheduling
- `remove_hive()` - Remove hive from registry
- `list_all_hives()` - Get all hive IDs
- `get_worker_count()` - Quick worker count lookup
- `is_hive_online()` - Check if hive is online
- `hive_count()` - Total number of hives

## Implementation Status

- ✅ Core functionality
- ✅ Tests (18 unit tests + 1 concurrency test)
- ✅ Documentation
- ✅ Thread safety

## Architecture Changes (TEAM-261/262)

**Before TEAM-261:**
```
Worker → Hive: POST /v1/heartbeat
Hive → Queen: POST /v1/heartbeat (aggregated)
```

**After TEAM-261:**
```
Worker → Queen: POST /v1/worker-heartbeat (direct)
```

**TEAM-262 Rename:**
- Struct: `HiveRegistry` → `WorkerRegistry`
- Crate: `queen-rbee-hive-registry` → `queen-rbee-worker-registry`
- Rationale: Name now reflects actual purpose (tracking workers, not hives)

## Dependencies

- `serde` - Serialization
- `chrono` - Timestamp handling
- `rbee-heartbeat` - Heartbeat types (TEAM-262: simplified)

## See Also

- `SPECS.md` - Complete specifications
- `hive-catalog` - Persistent storage (SQLite)
