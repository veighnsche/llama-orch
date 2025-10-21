# Heartbeat Architecture Issue - Missing Registry Update

## Problem Identified

The heartbeat endpoint currently ONLY updates `hive-catalog` (persistent storage) but does NOT update `hive-registry` (in-memory runtime state).

## Current Architecture (INCOMPLETE)

```
POST /v1/heartbeat
    ↓
handle_heartbeat()
    ↓
hive_catalog.update_heartbeat()  ✅ Updates SQLite
    ↓
❌ MISSING: hive_registry.update_heartbeat()  ← Should update RAM!
```

## Two Separate Concerns

### 1. Hive Catalog (Persistent - SQLite)
**Location**: `bin/15_queen_rbee_crates/hive-catalog`

**Purpose**: Persistent storage of hive metadata

**Stores**:
- ✅ Hive ID, host, port
- ✅ SSH credentials
- ✅ Device capabilities (CPU cores, total RAM, GPUs, total VRAM)
- ✅ Last heartbeat timestamp
- ✅ Hive status (Online/Offline)

**Updated**:
- On hive registration (CRUD operations)
- On device detection (first heartbeat)
- On heartbeat (timestamp only)

### 2. Hive Registry (In-Memory - RAM)
**Location**: `bin/15_queen_rbee_crates/hive-registry`

**Status**: 🚧 **STUB - NOT IMPLEMENTED YET!**

**Purpose**: Runtime state tracking

**Should Store**:
- ❌ Active workers (count, IDs, models)
- ❌ Current VRAM usage (available vs total)
- ❌ Current RAM usage (available vs total)
- ❌ Worker heartbeats
- ❌ Real-time metrics
- ❌ Last heartbeat timestamp (for quick checks)

**Should Be Updated**:
- On every heartbeat (worker count, resource usage)
- On worker spawn/delete
- On model load/unload

## Current Heartbeat Payload

```rust
pub struct HiveHeartbeatPayload {
    pub hive_id: String,
    pub timestamp: String,
    pub workers: Vec<WorkerInfo>,  // ← This data is IGNORED!
}
```

**Problem**: The `workers` field is sent but never processed!

## What Should Happen

### On Heartbeat Receipt

```rust
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<...> {
    // 1. Update CATALOG (persistent) - timestamp only
    state.hive_catalog
        .update_heartbeat(&payload.hive_id, timestamp_ms)
        .await?;
    
    // 2. Update REGISTRY (in-memory) - full runtime state
    state.hive_registry
        .update_hive_state(&payload.hive_id, HiveRuntimeState {
            workers: payload.workers,
            last_heartbeat: timestamp_ms,
            // Calculate from workers:
            vram_used: calculate_vram_used(&payload.workers),
            ram_used: calculate_ram_used(&payload.workers),
        })
        .await?;
    
    Ok(...)
}
```

## Missing HeartbeatState Field

### Current (INCOMPLETE)
```rust
pub struct HeartbeatState {
    pub hive_catalog: Arc<HiveCatalog>,  // ✅ Has catalog
    // ❌ MISSING: hive_registry
}
```

### Should Be
```rust
pub struct HeartbeatState {
    pub hive_catalog: Arc<HiveCatalog>,      // Persistent storage
    pub hive_registry: Arc<HiveRegistry>,    // In-memory state
}
```

## Hive Registry Implementation Needed

### Minimal Implementation

```rust
// bin/15_queen_rbee_crates/hive-registry/src/lib.rs

use std::collections::HashMap;
use std::sync::RwLock;

/// Runtime state for a single hive
#[derive(Debug, Clone)]
pub struct HiveRuntimeState {
    pub workers: Vec<WorkerInfo>,
    pub last_heartbeat_ms: i64,
    pub vram_used_gb: f32,
    pub ram_used_gb: f32,
}

/// In-memory registry for tracking hive runtime state
pub struct HiveRegistry {
    hives: RwLock<HashMap<String, HiveRuntimeState>>,
}

impl HiveRegistry {
    pub fn new() -> Self {
        Self {
            hives: RwLock::new(HashMap::new()),
        }
    }
    
    /// Update hive runtime state from heartbeat
    pub fn update_hive_state(&self, hive_id: &str, state: HiveRuntimeState) {
        let mut hives = self.hives.write().unwrap();
        hives.insert(hive_id.to_string(), state);
    }
    
    /// Get current runtime state for a hive
    pub fn get_hive_state(&self, hive_id: &str) -> Option<HiveRuntimeState> {
        let hives = self.hives.read().unwrap();
        hives.get(hive_id).cloned()
    }
    
    /// List all active hives (received heartbeat recently)
    pub fn list_active_hives(&self, max_age_ms: i64) -> Vec<String> {
        let hives = self.hives.read().unwrap();
        let now = chrono::Utc::now().timestamp_millis();
        
        hives
            .iter()
            .filter(|(_, state)| now - state.last_heartbeat_ms < max_age_ms)
            .map(|(id, _)| id.clone())
            .collect()
    }
    
    /// Get available resources for a hive
    pub fn get_available_resources(&self, hive_id: &str) -> Option<(f32, f32)> {
        // Returns (available_vram_gb, available_ram_gb)
        let hives = self.hives.read().unwrap();
        let state = hives.get(hive_id)?;
        
        // Would need total capacity from catalog
        // For now, just return used amounts
        Some((state.vram_used_gb, state.ram_used_gb))
    }
}
```

## Action Items

### 1. Implement Hive Registry
- [ ] Create basic in-memory HashMap-based registry
- [ ] Add methods: `update_hive_state()`, `get_hive_state()`, `list_active_hives()`
- [ ] Add resource calculation helpers

### 2. Update HeartbeatState
- [ ] Add `hive_registry: Arc<HiveRegistry>` field
- [ ] Initialize in `main.rs`

### 3. Update Heartbeat Handler
- [ ] Process `payload.workers` field
- [ ] Update both catalog AND registry
- [ ] Calculate resource usage from workers

### 4. Use Registry for Scheduling
- [ ] When spawning workers, check `hive_registry.get_available_resources()`
- [ ] When listing hives, use `hive_registry.list_active_hives()`
- [ ] For real-time metrics, query registry (not catalog)

## Benefits of Separation

### Catalog (SQLite)
- ✅ Survives restarts
- ✅ Persistent configuration
- ✅ Audit trail
- ❌ Slow for frequent updates
- ❌ Not suitable for real-time metrics

### Registry (RAM)
- ✅ Fast lookups
- ✅ Real-time state
- ✅ Perfect for scheduling decisions
- ❌ Lost on restart (but rebuilt from heartbeats)
- ❌ No persistence

## Example: Worker Scheduling

### Current (WRONG)
```rust
// Checking catalog for available resources
let hive = catalog.get_hive(hive_id).await?;
// ❌ Catalog doesn't know current VRAM usage!
```

### Correct (FUTURE)
```rust
// Check registry for real-time availability
let (available_vram, available_ram) = registry
    .get_available_resources(hive_id)
    .ok_or("Hive not active")?;

if available_vram >= model.vram_required {
    // Spawn worker
}
```

## Summary

**Current State**:
- ✅ Catalog implemented and working
- ❌ Registry is a stub (not implemented)
- ❌ Heartbeat only updates catalog
- ❌ Worker data in heartbeat is ignored
- ❌ No real-time resource tracking

**Required**:
1. Implement `hive-registry` crate
2. Add registry to `HeartbeatState`
3. Update heartbeat handler to update BOTH catalog and registry
4. Use registry for scheduling decisions

This is a critical architectural gap that needs to be filled!
