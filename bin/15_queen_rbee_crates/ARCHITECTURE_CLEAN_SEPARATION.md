# Architecture: Clean Separation of Concerns

## The Correct, Simple Architecture ✅

### Catalog (SQLite) - CONFIGURATION ONLY
**Purpose**: Persistent storage that survives restarts

**Stores ONLY**:
- ✅ Hive ID (name)
- ✅ Host, port
- ✅ SSH credentials (host, port, user)
- ✅ Device capabilities (CPU cores, GPUs, VRAM)
- ✅ Created/updated timestamps

**NEVER stores**:
- ❌ Last heartbeat timestamp
- ❌ Status (Online/Offline)
- ❌ Worker information
- ❌ VRAM/RAM usage
- ❌ Anything that changes frequently

### Registry (RAM) - EVERYTHING LIVE
**Purpose**: Real-time state for scheduling decisions

**Stores ALL heartbeat data**:
- ✅ Workers (full list with details)
- ✅ Worker states (Idle/Busy/Loading)
- ✅ Worker URLs (for direct inference)
- ✅ VRAM/RAM usage per worker
- ✅ CPU/GPU usage per worker
- ✅ Models loaded on workers
- ✅ Last heartbeat timestamp
- ✅ Online/Offline status (derived from heartbeat age)

## Why This Separation?

### The Problem with Mixing
If we store heartbeat data in SQLite:
- ❌ Database writes every 5 seconds (slow!)
- ❌ Disk I/O for every heartbeat
- ❌ Scheduling queries hit database (slow!)
- ❌ Complexity for no benefit

### The Solution: Separate Concerns
```
┌─────────────────────────────────────────────────────────┐
│                    QUEEN RESTARTS                        │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Catalog      │    │ Registry     │    │ What happens │
│ (SQLite)     │    │ (RAM)        │    │              │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ ✅ Survives  │    │ ❌ LOST!     │    │ Queen knows  │
│              │    │              │    │ which hives  │
│ localhost    │    │ (empty)      │    │ SHOULD exist │
│ 127.0.0.1    │    │              │    │              │
│ port: 8600   │    │              │    │ Waits for    │
│ 2x RTX 4090  │    │              │    │ heartbeats   │
│              │    │              │    │ (max 5s)     │
└──────────────┘    └──────────────┘    └──────────────┘
                            │
                            │ Heartbeats arrive every 5s
                            ▼
                    ┌──────────────┐
                    │ Registry     │
                    │ (RAM)        │
                    ├──────────────┤
                    │ ✅ REBUILT   │
                    │              │
                    │ Workers:     │
                    │ - worker-1   │
                    │   Idle       │
                    │   8GB VRAM   │
                    │   URL: ...   │
                    └──────────────┘
```

**Amnesia Duration**: Max 5 seconds (heartbeat interval)

**Is 5 seconds worth the complexity?** NO!

## The Flow

### On Heartbeat Received (Every 5 seconds):
```rust
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<...> {
    // ONLY update registry (RAM) - Fast!
    state.hive_registry
        .update_hive_state(&payload.hive_id, payload);
    
    // ❌ NO catalog update!
    // ❌ NO database write!
    // ❌ NO disk I/O!
    
    Ok(())
}
```

### For Scheduling (1000x per second):
```rust
// Use REGISTRY (RAM) - Instant!
let best_worker = registry.find_best_worker_for_model("llama-3-8b")?;
let worker_url = registry.get_worker_url(&worker_id)?;

// ❌ NOT catalog - That's for configuration!
```

### For Configuration (Rare):
```rust
// Use CATALOG (SQLite) - When needed
let hive = catalog.get_hive("localhost").await?;
println!("SSH: {}@{}", hive.ssh_user?, hive.ssh_host?);

// Trigger device detection on new hives
let need_detection = catalog.find_hives_without_devices().await?;
```

## Hard Block: Heartbeat → SQLite ❌

**RULE**: Nothing from heartbeat touches SQLite!

```rust
// ❌ WRONG - Don't do this!
catalog.update_heartbeat(&hive_id, timestamp).await?;
catalog.update_hive_status(&hive_id, HiveStatus::Online).await?;

// ✅ RIGHT - Only registry!
registry.update_hive_state(&hive_id, payload);
```

## API Comparison

### Catalog API (6 functions - Configuration)
```rust
// CREATE
add_hive(hive) -> Result<()>

// READ
get_hive(id) -> Result<Option<HiveRecord>>
list_hives() -> Result<Vec<HiveRecord>>

// UPDATE
update_hive(hive) -> Result<()>
update_devices(id, devices) -> Result<()>

// DELETE
remove_hive(id) -> Result<()>

// QUERY (Configuration)
find_hives_with_devices() -> Result<Vec<HiveRecord>>
find_hives_without_devices() -> Result<Vec<HiveRecord>>
hive_exists(id) -> Result<bool>
```

### Registry API (18 functions - Runtime)
```rust
// HIVE OPERATIONS (9 functions)
update_hive_state(hive_id, payload)
get_hive_state(hive_id) -> Option<HiveRuntimeState>
list_active_hives(max_age_ms) -> Vec<String>
get_available_resources(hive_id) -> Option<ResourceInfo>
remove_hive(hive_id) -> bool
list_all_hives() -> Vec<String>
get_worker_count(hive_id) -> Option<usize>
is_hive_online(hive_id, max_age_ms) -> bool
hive_count() -> usize

// WORKER OPERATIONS (9 functions)
get_worker(worker_id) -> Option<(String, WorkerInfo)>
get_worker_url(worker_id) -> Option<String>  // ← For direct inference!
list_all_workers() -> Vec<(String, WorkerInfo)>
find_idle_workers() -> Vec<(String, WorkerInfo)>
find_workers_by_model(model_id) -> Vec<(String, WorkerInfo)>
find_workers_by_backend(backend) -> Vec<(String, WorkerInfo)>
find_best_worker_for_model(model_id) -> Option<(String, WorkerInfo)>
total_worker_count() -> usize
get_workers_on_hive(hive_id) -> Vec<WorkerInfo>
```

## Data Models

### HiveRecord (Catalog - Configuration)
```rust
pub struct HiveRecord {
    pub id: String,
    pub host: String,
    pub port: u16,
    pub ssh_host: Option<String>,
    pub ssh_port: Option<u16>,
    pub ssh_user: Option<String>,
    pub devices: Option<DeviceCapabilities>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    // ❌ NO status
    // ❌ NO last_heartbeat_ms
}
```

### HiveRuntimeState (Registry - Runtime)
```rust
pub struct HiveRuntimeState {
    pub hive_id: String,
    pub workers: Vec<WorkerInfo>,
    pub last_heartbeat_ms: i64,
    pub vram_used_gb: f32,
    pub ram_used_gb: f32,
    pub worker_count: usize,
}

pub struct WorkerInfo {
    pub worker_id: String,
    pub state: String,  // Idle, Busy, Loading
    pub url: String,  // For direct inference!
    pub model_id: Option<String>,
    pub backend: Option<String>,
    pub device_id: Option<u32>,
    pub vram_bytes: Option<u64>,
    pub ram_bytes: Option<u64>,
    pub cpu_percent: Option<f32>,
    pub gpu_percent: Option<f32>,
    pub last_heartbeat: String,
    pub health_status: String,
}
```

## Benefits of Clean Separation

### ✅ Performance
- **Heartbeats**: No disk I/O (RAM only)
- **Scheduling**: Instant lookups (RAM)
- **Configuration**: Rare, can be slow

### ✅ Simplicity
- Clear boundaries
- No confusion about where data lives
- Easy to reason about

### ✅ Scalability
- Heartbeats can be 1 second interval (no problem!)
- 1000s of scheduling queries per second (no problem!)
- Database only for configuration changes

### ✅ Reliability
- Configuration survives restarts
- Runtime state rebuilds from heartbeats (5 seconds max)
- No data corruption from frequent writes

## What We Removed

### From Catalog (SQLite)
- ❌ `status` field
- ❌ `last_heartbeat_ms` field
- ❌ `update_hive_status()` function
- ❌ `update_heartbeat()` function
- ❌ `find_hives_by_status()` function
- ❌ `find_online_hives()` function
- ❌ `find_offline_hives()` function
- ❌ `find_stale_hives()` function
- ❌ `count_hives()` function
- ❌ `count_by_status()` function

**All of these belong in the registry (RAM)!**

## Migration Path

### Old Code (Wrong)
```rust
// ❌ Mixing concerns
catalog.update_heartbeat(&hive_id, timestamp).await?;
catalog.update_hive_status(&hive_id, HiveStatus::Online).await?;
```

### New Code (Right)
```rust
// ✅ Clean separation
registry.update_hive_state(&hive_id, payload);
```

## Summary

**Catalog**: Configuration that survives restarts (SSH, devices)  
**Registry**: Everything else (workers, heartbeat, status)

**Heartbeat interval**: 5 seconds  
**Amnesia duration**: Max 5 seconds  
**Complexity saved**: Massive!

**The trade-off is worth it!** 5 seconds of amnesia is acceptable for the simplicity gained.
