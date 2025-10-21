# Hive Registry - Complete Specifications

## Purpose

**In-memory (RAM) registry for tracking real-time runtime state of all hives AND workers.**

This crate serves as a **unified registry** - it replaces the need for a separate worker-registry crate.

### Key Insight

The hive-registry contains EVERYTHING needed for worker tracking because:
- Hives send heartbeats with complete worker information
- Each worker has: URL, state, model, resources, backend, device, etc.
- Worker URLs enable direct inference (bypassing hive as middleman)
- All data is already in memory for fast lookups

**Result**: The `worker-registry` crate was redundant and has been removed.

---

## Catalog vs Registry

This is DIFFERENT from `hive-catalog` (SQLite - persistent storage):
- **Catalog** = Persistent config (host, port, SSH, device capabilities)
- **Registry** = Runtime state (workers, VRAM usage, last heartbeat)

## Related Crates

### hive-catalog (SQLite - Persistent)
**Purpose**: Persistent storage for hive configuration

**Stores**:
- Host, port, SSH credentials
- Device capabilities
- Last heartbeat timestamp
- Status (Unknown/Online/Offline)

**Use case**: Configuration that survives restarts

### hive-lifecycle (Orchestration)
**Purpose**: Start, stop, and manage hive processes

**Responsibilities**:
- Spawn hives (localhost or remote via SSH)
- Add hives to catalog
- Fire-and-forget pattern
- Heartbeat callback mechanism

**Flow**:
```
hive-lifecycle          hive-catalog          hive-registry
      │                       │                      │
      │ 1. Start hive         │                      │
      ├──────────────────────>│                      │
      │    Add to catalog     │                      │
      │                       │                      │
      │ 2. Spawn process      │                      │
      │                       │                      │
      │ 3. Hive sends heartbeat                      │
      ├──────────────────────────────────────────────>│
      │              Update runtime state             │
      │                       │                      │
      │              Update catalog timestamp         │
      │<──────────────────────┤                      │
```

### Integration Example
```rust
// 1. Start hive (hive-lifecycle)
let response = execute_hive_start(catalog.clone(), request).await?;
println!("Hive spawning at {}", response.hive_url);

// 2. Hive sends heartbeat
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<...> {
    // Update catalog (persistent) - hive-catalog
    state.hive_catalog
        .update_heartbeat(&payload.hive_id, timestamp_ms)
        .await?;
    
    // Update registry (in-memory) - hive-registry
    state.hive_registry
        .update_hive_state(&payload.hive_id, payload);
    
    Ok(...)
}

// 3. Query workers for scheduling
let best_worker = registry.find_best_worker_for_model("llama-3-8b")?;
```

## Core Responsibilities

### 1. Track Real-Time Hive State
- Active workers on each hive
- Current VRAM/RAM usage
- Last heartbeat timestamp
- Hive online/offline status

### 2. Fast Lookups for Scheduling

The registry provides detailed, real-time information about both hives AND workers for intelligent scheduling decisions:

**Hive-Level Queries**:
- Which hives are online (received heartbeat recently)?
- Which hive has available VRAM for model X?
- Total VRAM/RAM usage per hive

**Worker-Level Queries** (The registry tracks ALL worker details):
- **Which workers** are on each hive (not just count)
- **Worker state**: Are they Idle, Busy, or Loading?
- **Resource usage per worker**:
  - VRAM used (bytes)
  - RAM used (bytes)  
  - CPU usage (percentage)
  - GPU usage (percentage)
- **Worker URL**: For direct inference (bypassing hive as middleman)
- **Model loaded**: Which model is running on this worker?
- **Backend**: cuda, cpu, or metal
- **Device ID**: GPU index
- **Last seen**: When worker last sent heartbeat to hive

### 3. Unified Hive + Worker Registry

**Important**: The hive-registry serves as BOTH the hive registry AND the worker registry. The separate `worker-registry` crate is **redundant and removed**.

**Why?** The heartbeat already contains complete worker information:
- Hive sends heartbeat with full worker list
- Each worker in the list has: state, URL, resources, model, backend, etc.
- Registry stores this in memory for fast lookups
- Worker URLs enable direct inference routing (no hive middleman needed)

**Architecture Decision**:
```
Before (redundant):
├── hive-registry    → Track hives
└── worker-registry  → Track workers (REDUNDANT!)

After (unified):
└── hive-registry    → Track hives + workers (all-in-one!)
``` 

### 4. Heartbeat Processing
- Update hive state from `HiveHeartbeatPayload`
- Calculate resource usage from worker list
- Track hive liveness

## Data Structures

### HiveRuntimeState
```rust
/// Runtime state for a single hive (in-memory only)
#[derive(Debug, Clone)]
pub struct HiveRuntimeState {
    /// Hive ID
    pub hive_id: String,
    
    /// All workers currently running on this hive
    pub workers: Vec<WorkerInfo>,
    
    /// Last heartbeat timestamp (milliseconds since epoch)
    pub last_heartbeat_ms: i64,
    
    /// Total VRAM used by all workers (GB)
    pub vram_used_gb: f32,
    
    /// Total RAM used by all workers (GB)
    pub ram_used_gb: f32,
    
    /// Number of active workers
    pub worker_count: usize,
}
```

### WorkerInfo
```rust
/// Worker information from heartbeat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Worker ID
    pub worker_id: String,
    
    /// Worker state ("Idle", "Busy", "Loading")
    pub state: String,
    
    /// Last heartbeat from worker
    pub last_heartbeat: String,
    
    /// Health status ("healthy", "degraded")
    pub health_status: String,
}
```

## Public API Summary

**Total: 18 Public Functions**
- 9 Hive Management Functions
- 9 Worker Registry Functions

The hive-registry provides comprehensive functionality for both hive-level and worker-level operations, serving as a unified registry for the entire system.

---

## Public API

### Hive Management Functions (9 functions)

#### 1. Update Hive State (from heartbeat)
```rust
pub fn update_hive_state(&self, hive_id: &str, payload: HiveHeartbeatPayload)
```
**Purpose**: Process incoming heartbeat and update runtime state
**Called by**: Heartbeat handler
**Updates**:
- Worker list
- Last heartbeat timestamp
- VRAM/RAM usage (calculated from workers)
- Worker count

#### 2. Get Hive State
```rust
pub fn get_hive_state(&self, hive_id: &str) -> Option<HiveRuntimeState>
```
**Purpose**: Get current runtime state for a hive
**Used for**: Scheduling decisions, status queries

#### 3. List Active Hives
```rust
pub fn list_active_hives(&self, max_age_ms: i64) -> Vec<String>
```
**Purpose**: Get all hives that sent heartbeat recently
**Parameters**:
- `max_age_ms`: Maximum age of last heartbeat (e.g., 30000 = 30 seconds)
**Returns**: List of hive IDs that are considered "online"

#### 4. Get Available Resources
```rust
pub fn get_available_resources(&self, hive_id: &str) -> Option<ResourceInfo>

pub struct ResourceInfo {
    pub vram_used_gb: f32,
    pub ram_used_gb: f32,
    pub worker_count: usize,
}
```
**Purpose**: Get current resource usage for scheduling
**Used for**: Deciding where to spawn new workers

#### 5. Remove Hive
```rust
pub fn remove_hive(&self, hive_id: &str) -> bool
```
**Purpose**: Remove hive from registry (e.g., when hive stops)
**Returns**: true if hive was removed, false if not found

#### 6. List All Hives
```rust
pub fn list_all_hives(&self) -> Vec<String>
```
**Purpose**: Get all hive IDs in registry (regardless of heartbeat age)

#### 7. Get Worker Count
```rust
pub fn get_worker_count(&self, hive_id: &str) -> Option<usize>
```
**Purpose**: Quick lookup of worker count for a hive

#### 8. Is Hive Online
```rust
pub fn is_hive_online(&self, hive_id: &str, max_age_ms: i64) -> bool
```
**Purpose**: Check if hive is considered online (recent heartbeat)

---

### Worker Registry Functions

The registry provides **9 additional functions** for worker-level operations, enabling direct worker access and intelligent routing:

#### 1. Get Worker
```rust
pub fn get_worker(&self, worker_id: &str) -> Option<(String, WorkerInfo)>
```
**Purpose**: Find worker by ID across all hives  
**Returns**: `(hive_id, worker_info)` if found

**Why important**: Locate a specific worker when you have the worker ID

#### 2. Get Worker URL
```rust
pub fn get_worker_url(&self, worker_id: &str) -> Option<String>
```
**Purpose**: Get worker URL for **direct inference routing**  
**Critical**: This enables bypassing the hive as a middleman

**Example**: Route inference directly to worker
```rust
if let Some(url) = registry.get_worker_url("worker-123") {
    // Send inference request directly to worker
    let response = http_client.post(&url).json(&request).send().await?;
}
```

#### 3. List All Workers
```rust
pub fn list_all_workers(&self) -> Vec<(String, WorkerInfo)>
```
**Purpose**: Get all workers across all hives  
**Returns**: List of `(hive_id, worker_info)` tuples

#### 4. Find Idle Workers
```rust
pub fn find_idle_workers(&self) -> Vec<(String, WorkerInfo)>
```
**Purpose**: Find all workers in "Idle" state (available for work)  
**Use case**: Find available capacity for new inference requests

#### 5. Find Workers by Model
```rust
pub fn find_workers_by_model(&self, model_id: &str) -> Vec<(String, WorkerInfo)>
```
**Purpose**: Find workers with specific model already loaded  
**Use case**: Avoid model loading delay by routing to workers with model ready

#### 6. Find Workers by Backend
```rust
pub fn find_workers_by_backend(&self, backend: &str) -> Vec<(String, WorkerInfo)>
```
**Purpose**: Find workers using specific backend (cuda, cpu, metal)  
**Use case**: Hardware-specific routing (e.g., prefer CUDA workers)

#### 7. Find Best Worker for Model
```rust
pub fn find_best_worker_for_model(&self, model_id: &str) -> Option<(String, WorkerInfo)>
```
**Purpose**: Intelligent worker selection with optimization  
**Algorithm**:
1. Prefer workers with model already loaded
2. Among those, pick worker with lowest GPU usage
3. Fallback to any idle worker with lowest GPU usage

**This is the key scheduling function!**

#### 8. Total Worker Count
```rust
pub fn total_worker_count(&self) -> usize
```
**Purpose**: Get total workers across all hives  
**Use case**: Capacity monitoring, statistics

#### 9. Get Workers on Hive
```rust
pub fn get_workers_on_hive(&self, hive_id: &str) -> Vec<WorkerInfo>
```
**Purpose**: Get all workers on a specific hive  
**Use case**: Hive-specific monitoring, load distribution

---

## Implementation Details

### Thread Safety
- Use `RwLock<HashMap<String, HiveRuntimeState>>` for concurrent access
- Read-heavy workload (scheduling queries)
- Write on heartbeat (less frequent)

### Resource Calculation
```rust
fn calculate_resources(workers: &[WorkerInfo]) -> (f32, f32) {
    // For now, return placeholder values
    // Future: Calculate from worker metadata
    let vram_used = workers.len() as f32 * 4.0; // Assume 4GB per worker
    let ram_used = workers.len() as f32 * 2.0;  // Assume 2GB per worker
    (vram_used, ram_used)
}
```

### Heartbeat Age Check
```rust
fn is_recent(last_heartbeat_ms: i64, max_age_ms: i64) -> bool {
    let now = chrono::Utc::now().timestamp_millis();
    now - last_heartbeat_ms < max_age_ms
}
```

## Usage Examples

### Example 1: Update from Heartbeat
```rust
use rbee_heartbeat::HiveHeartbeatPayload;

let registry = HiveRegistry::new();

// Heartbeat arrives
let payload = HiveHeartbeatPayload {
    hive_id: "localhost".to_string(),
    timestamp: "2025-10-21T10:00:00Z".to_string(),
    workers: vec![
        WorkerState {
            worker_id: "worker-1".to_string(),
            state: "Idle".to_string(),
            last_heartbeat: "2025-10-21T10:00:00Z".to_string(),
            health_status: "healthy".to_string(),
        },
    ],
};

registry.update_hive_state("localhost", payload);
```

### Example 2: Check if Hive is Online
```rust
// Check if hive sent heartbeat in last 30 seconds
let is_online = registry.is_hive_online("localhost", 30_000);

if is_online {
    println!("Hive is online!");
}
```

### Example 3: Get Active Hives for Scheduling
```rust
// Get all hives that are online (heartbeat in last 30s)
let active_hives = registry.list_active_hives(30_000);

for hive_id in active_hives {
    if let Some(resources) = registry.get_available_resources(&hive_id) {
        println!("Hive {} has {} workers, using {}GB VRAM",
            hive_id, resources.worker_count, resources.vram_used_gb);
    }
}
```

### Example 4: Find Hive with Capacity
```rust
// Find hive with least VRAM usage
let active_hives = registry.list_active_hives(30_000);
let best_hive = active_hives
    .iter()
    .filter_map(|hive_id| {
        registry.get_available_resources(hive_id)
            .map(|res| (hive_id, res.vram_used_gb))
    })
    .min_by(|(_, vram_a), (_, vram_b)| {
        vram_a.partial_cmp(vram_b).unwrap()
    });

if let Some((hive_id, vram)) = best_hive {
    println!("Best hive: {} with {}GB VRAM used", hive_id, vram);
}
```

## Integration with Heartbeat Handler

```rust
// In heartbeat handler
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<...> {
    // 1. Update catalog (persistent) - timestamp only
    state.hive_catalog
        .update_heartbeat(&payload.hive_id, timestamp_ms)
        .await?;
    
    // 2. Update registry (in-memory) - full runtime state
    state.hive_registry
        .update_hive_state(&payload.hive_id, payload);
    
    Ok(...)
}
```

## Testing Requirements

### Unit Tests
1. ✅ Create registry
2. ✅ Update hive state
3. ✅ Get hive state
4. ✅ List active hives (with age filter)
5. ✅ Remove hive
6. ✅ Get available resources
7. ✅ Worker count tracking
8. ✅ Online status check
9. ✅ Thread safety (concurrent reads/writes)
10. ✅ Heartbeat age calculation

### Integration Tests
1. ✅ Multiple hives updating concurrently
2. ✅ Hive goes offline (old heartbeat)
3. ✅ Hive comes back online
4. ✅ Worker count changes over time

## Performance Requirements

- **Read latency**: < 1ms (in-memory HashMap lookup)
- **Write latency**: < 5ms (RwLock write + calculation)
- **Concurrent reads**: Unlimited (RwLock allows multiple readers)
- **Memory usage**: ~1KB per hive (100 hives = ~100KB)

## Future Enhancements

### Phase 2: Resource Tracking
- Track actual VRAM/RAM from worker metadata
- Track model affinity (which models are loaded)
- Track GPU utilization

### Phase 3: Metrics
- Heartbeat frequency tracking
- Worker churn rate
- Resource usage trends

### Phase 4: Eviction
- Auto-remove hives with old heartbeats
- Configurable TTL

## Dependencies

```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
chrono = "0.4"
rbee-heartbeat = { path = "../../../99_shared_crates/heartbeat" }
```

## File Structure

```
hive-registry/
├── Cargo.toml
├── README.md
├── SPECS.md (this file)
├── src/
│   ├── lib.rs          # Main registry implementation
│   └── types.rs        # HiveRuntimeState, ResourceInfo
└── tests/
    └── integration_tests.rs
```

## Success Criteria

✅ All public API functions implemented
✅ Thread-safe (RwLock)
✅ All unit tests passing
✅ Integration tests passing
✅ Documentation complete
✅ Used by heartbeat handler
✅ Used by scheduler for hive selection

## See Also

- **hive-catalog** (`/bin/15_queen_rbee_crates/hive-catalog/SPECS.md`) - Persistent storage (SQLite)
- **hive-lifecycle** (`/bin/15_queen_rbee_crates/hive-lifecycle/SPECS.md`) - Start/stop hive processes
- **heartbeat** (`/bin/99_shared_crates/heartbeat/`) - Heartbeat types and handlers
- **Device detection** - Capabilities tracking
- **Scheduler** - Uses registry for intelligent worker selection
