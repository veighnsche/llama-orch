# Hive Registry - Complete Specifications

## Purpose

**In-memory (RAM) registry for tracking real-time runtime state of all hives.**

This is DIFFERENT from `hive-catalog` (SQLite - persistent storage):
- **Catalog** = Persistent config (host, port, SSH, device capabilities)
- **Registry** = Runtime state (workers, VRAM usage, last heartbeat)

## Core Responsibilities

### 1. Track Real-Time Hive State
- Active workers on each hive
- Current VRAM/RAM usage
- Last heartbeat timestamp
- Hive online/offline status

### 2. Fast Lookups for Scheduling
- Which hives are online (received heartbeat recently)?
- Which hive has available VRAM for model X?
- How many workers are running on hive Y? // so not how many that is not enough info. it should be which workers. are they currently working? how much (V)RAM is this worker taken and how much CPU and GPU resources is it taking. and in the heartbeat should also contain when the last time the worker is seen. so the queen needs a lot of info from the hive and their workers.

// Alright so the worker registry is now actually redundant now I think of it. The heartbeat contains all the info about the worker registry. the only thing that the worker registry should save was the URL of the worker so that the queen can connect to it for inference without needing the hive as a middleman.

// So please update the this crate so that it also does the worker registry work. 

### 3. Heartbeat Processing
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

## Public API

### Core Operations

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
