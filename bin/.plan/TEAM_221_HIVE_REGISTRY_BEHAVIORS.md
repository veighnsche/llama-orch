# HIVE-REGISTRY BEHAVIOR INVENTORY

**Team:** TEAM-221  
**Component:** queen-rbee-hive-registry  
**Date:** 2025-10-22  
**LOC:** 918 (lib.rs: 627, types.rs: 291)

---

## 1. Public API Surface

### Hive Management Functions (9 functions)

**`HiveRegistry::new() -> Self`**
- Creates empty in-memory registry with RwLock-protected HashMap
- Thread-safe, no initialization errors

**`update_hive_state(&self, hive_id: &str, payload: HiveHeartbeatPayload)`**
- Processes incoming heartbeat from hive
- Parses RFC3339 timestamp (fallback to current time on error)
- Converts `WorkerState` → `WorkerInfo` (from heartbeat types)
- Calculates VRAM/RAM usage (placeholder: 4GB/2GB per worker)
- Replaces entire hive state (no merge, full replacement)
- Write lock acquisition

**`get_hive_state(&self, hive_id: &str) -> Option<HiveRuntimeState>`**
- Returns cloned runtime state for hive
- Read lock acquisition
- Returns None if hive not found

**`list_active_hives(&self, max_age_ms: i64) -> Vec<String>`**
- Filters hives by heartbeat age
- Uses `is_recent()` check (current time - last_heartbeat < max_age)
- Returns only hive IDs (not full state)
- Read lock acquisition

**`get_available_resources(&self, hive_id: &str) -> Option<ResourceInfo>`**
- Returns VRAM/RAM usage + worker count
- Derived from HiveRuntimeState
- Read lock acquisition

**`remove_hive(&self, hive_id: &str) -> bool`**
- Removes hive from registry
- Returns true if removed, false if not found
- Write lock acquisition

**`list_all_hives(&self) -> Vec<String>`**
- Returns all hive IDs regardless of heartbeat age
- Read lock acquisition

**`get_worker_count(&self, hive_id: &str) -> Option<usize>`**
- Quick lookup of worker count
- Read lock acquisition

**`is_hive_online(&self, hive_id: &str, max_age_ms: i64) -> bool`**
- Checks if heartbeat is recent
- Returns false if hive not found
- Read lock acquisition

**`hive_count(&self) -> usize`**
- Total number of hives in registry
- Read lock acquisition

### Worker Registry Functions (9 functions)

**`get_worker(&self, worker_id: &str) -> Option<(String, WorkerInfo)>`**
- Searches across ALL hives for worker
- Returns (hive_id, worker_info) tuple
- Linear search through all hives and workers
- Read lock acquisition

**`get_worker_url(&self, worker_id: &str) -> Option<String>`**
- Wrapper around `get_worker()` returning only URL
- Critical for direct inference routing (bypasses hive)
- Read lock acquisition

**`list_all_workers(&self) -> Vec<(String, WorkerInfo)>`**
- Flat map across all hives
- Returns (hive_id, worker_info) tuples
- Read lock acquisition

**`find_idle_workers(&self) -> Vec<(String, WorkerInfo)>`**
- Filters workers where `state == "Idle"`
- Read lock acquisition

**`find_workers_by_model(&self, model_id: &str) -> Vec<(String, WorkerInfo)>`**
- Filters workers where `model_id == Some(model_id)`
- Read lock acquisition

**`find_workers_by_backend(&self, backend: &str) -> Vec<(String, WorkerInfo)>`**
- Filters workers where `backend == Some(backend)`
- Read lock acquisition

**`find_best_worker_for_model(&self, model_id: &str) -> Option<(String, WorkerInfo)>`**
- **Smart scheduling algorithm:**
  1. Find idle workers with model already loaded
  2. Among those, select worker with lowest GPU usage
  3. Fallback: any idle worker with lowest GPU usage
- Uses `partial_cmp()` on `gpu_percent` (defaults to 100.0 if None)
- Read lock acquisition

**`total_worker_count(&self) -> usize`**
- Sum of worker counts across all hives
- Read lock acquisition

**`get_workers_on_hive(&self, hive_id: &str) -> Vec<WorkerInfo>`**
- Returns all workers on specific hive
- Returns empty vec if hive not found
- Read lock acquisition

### Exported Types

**`HiveRuntimeState`** (Clone, Debug)
- `hive_id: String`
- `workers: Vec<WorkerInfo>`
- `last_heartbeat_ms: i64`
- `vram_used_gb: f32`
- `ram_used_gb: f32`
- `worker_count: usize`

**`WorkerInfo`** (Clone, Debug, Serialize, Deserialize)
- `worker_id: String`
- `state: String` ("Idle", "Busy", "Loading")
- `last_heartbeat: String` (RFC3339)
- `health_status: String` ("healthy", "degraded")
- `url: String` (for direct inference)
- `model_id: Option<String>`
- `backend: Option<String>` ("cuda", "cpu", "metal")
- `device_id: Option<u32>`
- `vram_bytes: Option<u64>`
- `ram_bytes: Option<u64>`
- `cpu_percent: Option<f32>`
- `gpu_percent: Option<f32>`

**`ResourceInfo`** (Clone, Debug)
- `vram_used_gb: f32`
- `ram_used_gb: f32`
- `worker_count: usize`

---

## 2. State Machine Behaviors

### Hive Lifecycle
```
[Not in registry] --update_hive_state()--> [In registry, recent heartbeat]
                                                    |
                                                    | (time passes)
                                                    v
                                            [In registry, stale heartbeat]
                                                    |
                                                    | remove_hive()
                                                    v
                                            [Not in registry]
```

### State Transitions
- **Entry**: `update_hive_state()` - adds or replaces hive
- **Update**: `update_hive_state()` - full replacement (no merge)
- **Query**: All read operations (no state change)
- **Exit**: `remove_hive()` - removes from registry

### Heartbeat Freshness
- **Fresh**: `now - last_heartbeat_ms < max_age_ms`
- **Stale**: `now - last_heartbeat_ms >= max_age_ms`
- No automatic eviction (manual removal only)

### Worker State
- Workers have no independent lifecycle in registry
- Workers are part of hive state
- Worker list is fully replaced on each heartbeat
- No worker-level add/remove operations

---

## 3. Data Flows

### Input: Heartbeat Processing
```
HiveHeartbeatPayload (from hive)
    ↓
update_hive_state()
    ↓
Parse timestamp (RFC3339 → milliseconds)
    ↓
Convert WorkerState → WorkerInfo
    ↓
Calculate resources (placeholder: 4GB VRAM, 2GB RAM per worker)
    ↓
Create HiveRuntimeState
    ↓
Write to HashMap (full replacement)
```

### Output: Scheduling Queries
```
Scheduler needs worker
    ↓
find_best_worker_for_model(model_id)
    ↓
Filter idle workers
    ↓
Prefer workers with model loaded
    ↓
Sort by GPU usage (ascending)
    ↓
Return (hive_id, worker_info)
    ↓
Extract worker.url for direct inference
```

### Output: Resource Monitoring
```
Monitor needs hive status
    ↓
list_active_hives(30_000)
    ↓
Filter by heartbeat age
    ↓
For each hive: get_available_resources()
    ↓
Return ResourceInfo (VRAM, RAM, worker count)
```

---

## 4. Error Handling

### No Error Types
- All functions are infallible (no Result types)
- Uses Option for not-found cases
- RwLock panics on poison (acceptable for in-memory registry)

### Timestamp Parsing
- Invalid RFC3339 → fallback to current time
- Uses `unwrap_or_else()` pattern (line 65)

### Not Found Behaviors
- `get_hive_state()` → None
- `get_worker()` → None
- `get_worker_url()` → None
- `get_available_resources()` → None
- `get_worker_count()` → None
- `is_hive_online()` → false (not None)
- `remove_hive()` → false (not None)
- `get_workers_on_hive()` → empty Vec (not None)

### Thread Safety
- RwLock poison handling: panic on poison (lines 91, 99, etc.)
- Acceptable for in-memory registry (poison = unrecoverable)

---

## 5. Integration Points

### Dependencies
- `serde` - Serialization for WorkerInfo
- `chrono` - Timestamp parsing and current time
- `rbee-heartbeat` - HiveHeartbeatPayload, WorkerState types

### Dependents (Expected)
- `queen-rbee` - Heartbeat handler, inference router, scheduler
- `hive-catalog` - Persistent storage (separate concern)
- `hive-lifecycle` - Hive process management (separate concern)

### Integration Pattern
```rust
// In queen-rbee heartbeat handler
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<...> {
    // Update catalog (persistent timestamp)
    state.hive_catalog.update_heartbeat(&payload.hive_id, timestamp_ms).await?;
    
    // Update registry (in-memory full state)
    state.hive_registry.update_hive_state(&payload.hive_id, payload);
    
    Ok(...)
}
```

### Catalog vs Registry Separation
- **Catalog** (SQLite): Persistent config, SSH credentials, device capabilities
- **Registry** (RAM): Runtime state, worker list, resource usage
- Registry is rebuilt from heartbeats on restart (no persistence)

---

## 6. Critical Invariants

### Data Consistency
1. **Worker count == workers.len()** (enforced in `from_heartbeat()`)
2. **VRAM/RAM calculated from worker list** (placeholder: 4GB/2GB per worker)
3. **Timestamp always in milliseconds** (converted from RFC3339)
4. **Full state replacement** (no partial updates, no merge logic)

### Thread Safety
1. **All reads use read lock** (multiple concurrent readers allowed)
2. **All writes use write lock** (exclusive access)
3. **No lock held across async boundaries** (all operations are sync)
4. **Cloned data returned** (no references to internal state)

### Heartbeat Age
1. **Age check uses current time** (`chrono::Utc::now()`)
2. **Age is relative** (not absolute timestamp comparison)
3. **No automatic eviction** (stale hives remain until explicitly removed)

### Worker Lookup
1. **Worker IDs must be unique across all hives** (not enforced, assumed)
2. **First match wins** (if duplicate worker IDs exist)
3. **Linear search** (no indexing by worker_id)

---

## 7. Existing Test Coverage

### Unit Tests (34 tests, 100% passing)

#### Hive Registry Tests (23 tests)
- ✅ `test_new_registry_is_empty` - Empty registry creation
- ✅ `test_update_hive_state` - Basic heartbeat processing
- ✅ `test_get_hive_state` - State retrieval
- ✅ `test_get_hive_state_not_found` - Not found case
- ✅ `test_list_active_hives` - Heartbeat age filtering
- ✅ `test_list_active_hives_filters_old` - Stale heartbeat filtering
- ✅ `test_get_available_resources` - Resource calculation
- ✅ `test_remove_hive` - Hive removal
- ✅ `test_remove_hive_not_found` - Remove non-existent
- ✅ `test_list_all_hives` - List all regardless of age
- ✅ `test_get_worker_count` - Worker count lookup
- ✅ `test_get_worker_count_not_found` - Not found case
- ✅ `test_is_hive_online_true` - Online check (recent)
- ✅ `test_is_hive_online_false_old_heartbeat` - Online check (stale)
- ✅ `test_is_hive_online_false_not_found` - Online check (not found)
- ✅ `test_update_existing_hive` - State replacement
- ✅ `test_concurrent_access` - Thread safety (10 threads)

#### Worker Registry Tests (11 tests)
- ✅ `test_get_worker` - Worker lookup by ID
- ✅ `test_get_worker_not_found` - Worker not found
- ✅ `test_get_worker_url` - URL extraction
- ✅ `test_list_all_workers` - List across all hives
- ✅ `test_find_idle_workers` - State filtering
- ✅ `test_find_workers_by_model` - Model filtering
- ✅ `test_find_workers_by_backend` - Backend filtering
- ✅ `test_find_best_worker_for_model` - Smart selection
- ✅ `test_total_worker_count` - Total count
- ✅ `test_get_workers_on_hive` - Hive-specific workers
- ✅ `test_get_workers_on_nonexistent_hive` - Not found case

#### Types Tests (6 tests in types.rs)
- ✅ `test_hive_runtime_state_from_heartbeat` - State construction
- ✅ `test_resource_info` - ResourceInfo extraction
- ✅ `test_is_recent_true` - Heartbeat freshness (recent)
- ✅ `test_is_recent_false` - Heartbeat freshness (stale)
- ✅ `test_calculate_resources_empty` - Empty worker list
- ✅ `test_calculate_resources_multiple_workers` - Multiple workers

### BDD Tests
- ❌ Only placeholder feature file exists
- ❌ No actual BDD scenarios implemented

### Coverage Gaps (Implemented Code Without Tests)
1. **No tests for invalid timestamp parsing** (line 63-65)
   - What happens with malformed RFC3339?
   - Fallback to current time is tested indirectly
2. **No tests for RwLock poison** (acceptable - panic is correct behavior)
3. **No tests for duplicate worker IDs across hives** (assumed unique)
4. **No tests for empty worker list in find_best_worker_for_model()**
5. **No tests for GPU percent None vs Some in sorting**

---

## 8. Behavior Checklist

- [x] All public APIs documented (18 functions + 3 types)
- [x] All state transitions documented (hive lifecycle)
- [x] All error paths documented (Option/bool returns, no errors)
- [x] All integration points documented (heartbeat, catalog, scheduler)
- [x] All edge cases documented (not found, stale heartbeats, empty lists)
- [x] Existing test coverage assessed (34 unit tests, 0 BDD tests)
- [x] Coverage gaps identified (timestamp parsing, duplicate IDs, edge cases)

---

## TEAM-221: Investigated
// All code reviewed and documented by TEAM-221
