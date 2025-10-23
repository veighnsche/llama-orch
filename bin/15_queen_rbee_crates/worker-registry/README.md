# queen-rbee-hive-registry

**Status:** ✅ IMPLEMENTED  
**Purpose:** In-memory registry for tracking real-time runtime state of all hives

## Overview

In-memory (RAM) registry for tracking real-time runtime state of all hives.

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
use queen_rbee_hive_registry::HiveRegistry;
use rbee_heartbeat::HiveHeartbeatPayload;

let registry = HiveRegistry::new();

// Update from heartbeat
let payload = HiveHeartbeatPayload {
    hive_id: "localhost".to_string(),
    timestamp: "2025-10-21T10:00:00Z".to_string(),
    workers: vec![],
};
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

## Dependencies

- `serde` - Serialization
- `chrono` - Timestamp handling
- `rbee-heartbeat` - Heartbeat types

## See Also

- `SPECS.md` - Complete specifications
- `hive-catalog` - Persistent storage (SQLite)
