# Registry Consolidation Analysis

**Date:** Oct 31, 2025  
**Status:** 🔍 ANALYSIS  
**Problem:** 3 registry crates with confusing names and overlapping responsibilities

---

## Current State: 3 Registry Crates

### 1. `heartbeat-registry` (Generic, Shared)
**Location:** `bin/99_shared_crates/heartbeat-registry`  
**Purpose:** Generic heartbeat tracking with `HeartbeatItem` trait  
**LOC:** ~388 lines  
**Status:** ✅ CORRECT - This is the foundation

**API:**
```rust
pub trait HeartbeatItem {
    type Info;
    fn id(&self) -> &str;
    fn info(&self) -> Self::Info;
    fn is_recent(&self) -> bool;
    fn is_available(&self) -> bool;
}

pub struct HeartbeatRegistry<T: HeartbeatItem> {
    items: RwLock<HashMap<String, T>>,
}
```

**Methods:**
- `update()`, `get()`, `remove()`
- `list_all()`, `list_online()`, `list_available()`
- `count_online()`, `count_available()`, `count_total()`
- `cleanup_stale()`

### 2. `hive-registry` (Queen-specific)
**Location:** `bin/15_queen_rbee_crates/hive-registry`  
**Purpose:** Track hive state + worker telemetry  
**LOC:** ~302 lines  
**Status:** ⚠️ CONFUSING - Stores both hives AND workers

**Structure:**
```rust
pub struct HiveRegistry {
    inner: HeartbeatRegistry<HiveHeartbeat>,  // Hive heartbeats
    workers: RwLock<HashMap<String, Vec<ProcessStats>>>,  // Worker telemetry!
}
```

**API:**
- Hive methods: `update_hive()`, `get_hive()`, `list_online_hives()`, etc.
- **Worker methods:** `update_workers()`, `get_workers()`, `find_idle_workers()`, etc.

**Problem:** Name says "hive" but stores BOTH hives and workers!

### 3. `worker-registry` (Queen-specific)
**Location:** `bin/15_queen_rbee_crates/worker-registry`  
**Purpose:** Track worker heartbeats (DEPRECATED - workers don't send heartbeats anymore!)  
**LOC:** ~289 lines  
**Status:** ❌ DEPRECATED - Workers monitored via hive telemetry (TEAM-362)

**Structure:**
```rust
pub struct WorkerRegistry {
    inner: HeartbeatRegistry<WorkerHeartbeat>,  // Worker heartbeats
}
```

**Problem:** Workers DON'T send heartbeats to Queen anymore! Hive sends worker telemetry.

---

## Architecture Confusion

### Current (Confusing) Flow

```
Hive → POST /v1/hive-heartbeat → Queen
  ↓
  Contains: HiveInfo + Vec<ProcessStats> (workers)
  ↓
Queen stores:
  - HiveInfo → HiveRegistry.inner (HeartbeatRegistry<HiveHeartbeat>)
  - Workers → HiveRegistry.workers (HashMap<String, Vec<ProcessStats>>)
```

**Problem:** `HiveRegistry` is actually a "Hive + Worker Registry"!

### Worker Registry (Unused)

```
Worker → (NO HEARTBEAT) → Queen
```

**Reality:** Workers are monitored via hive telemetry, not direct heartbeats.

**`WorkerRegistry` is NEVER populated!**

---

## Usage Analysis

### Where `HiveRegistry` is Used

1. **`main.rs`** - Creates instance
   ```rust
   let hive_registry = Arc::new(queen_rbee_hive_registry::HiveRegistry::new());
   ```

2. **`http/heartbeat.rs`** - Receives hive heartbeats
   ```rust
   state.hive_registry.update_hive(heartbeat);
   state.hive_registry.update_workers(&heartbeat.hive.id, heartbeat.workers);
   ```

3. **`hive_subscriber.rs`** - Receives SSE telemetry (Phase 2)
   ```rust
   hive_registry.update_workers(&hive_id, parsed_workers);
   ```

4. **`http/heartbeat_stream.rs`** - Queries for SSE stream
   ```rust
   let workers = state.hive_registry.get_all_workers();
   let hives = state.hive_registry.list_online_hives();
   ```

### Where `WorkerRegistry` is Used

1. **`main.rs`** - Creates instance
   ```rust
   let worker_registry = Arc::new(queen_rbee_worker_registry::WorkerRegistry::new());
   ```

2. **`http/heartbeat_stream.rs`** - Queries for SSE stream
   ```rust
   let workers = state.worker_registry.list_online_workers();  // ALWAYS EMPTY!
   ```

3. **`job_router.rs`** - Passed to JobState
   ```rust
   pub struct JobState {
       pub hive_registry: Arc<queen_rbee_worker_registry::WorkerRegistry>,  // WRONG NAME!
   }
   ```

**Problem:** Field named `hive_registry` but type is `WorkerRegistry`!

---

## The Real Architecture (Post-TEAM-362)

### What Actually Happens

```
Hive monitors workers (cgroup + GPU)
    ↓
Hive sends telemetry every 1s
    ↓
POST /v1/hive-heartbeat (legacy) OR SSE stream (new)
    ↓
Queen receives:
  - HiveInfo (hive metadata)
  - Vec<ProcessStats> (worker telemetry)
    ↓
Queen stores in HiveRegistry:
  - Hive heartbeats
  - Worker telemetry (per hive)
```

### Scheduler Needs

```rust
// Scheduler queries HiveRegistry for:
hive_registry.find_idle_workers()  // gpu_util_pct == 0.0
hive_registry.find_workers_with_model("llama-3-8b")
hive_registry.find_workers_with_capacity(8192)  // VRAM in MB
```

**All worker queries go through `HiveRegistry`!**

---

## Proposed Solution: Consolidate to ONE Registry

### Option A: Rename `HiveRegistry` → `TelemetryRegistry`

```rust
// bin/15_queen_rbee_crates/telemetry-registry

pub struct TelemetryRegistry {
    hives: HeartbeatRegistry<HiveHeartbeat>,
    workers: RwLock<HashMap<String, Vec<ProcessStats>>>,
}

impl TelemetryRegistry {
    // Hive methods
    pub fn update_hive(&self, heartbeat: HiveHeartbeat) { ... }
    pub fn get_hive(&self, hive_id: &str) -> Option<HiveInfo> { ... }
    pub fn list_online_hives(&self) -> Vec<HiveInfo> { ... }
    
    // Worker methods
    pub fn update_workers(&self, hive_id: &str, workers: Vec<ProcessStats>) { ... }
    pub fn get_workers(&self, hive_id: &str) -> Option<Vec<ProcessStats>> { ... }
    pub fn get_all_workers(&self) -> Vec<ProcessStats> { ... }
    pub fn find_idle_workers(&self) -> Vec<ProcessStats> { ... }
    pub fn find_workers_with_model(&self, model: &str) -> Vec<ProcessStats> { ... }
    pub fn find_workers_with_capacity(&self, vram_mb: u64) -> Vec<ProcessStats> { ... }
}
```

**Benefits:**
- ✅ Name reflects reality (telemetry from hives)
- ✅ Single source of truth
- ✅ Scheduler has one registry to query
- ✅ Clear separation: hives vs workers

### Option B: Keep `HiveRegistry`, Delete `WorkerRegistry`

```rust
// bin/15_queen_rbee_crates/hive-registry

pub struct HiveRegistry {
    hives: HeartbeatRegistry<HiveHeartbeat>,
    workers: RwLock<HashMap<String, Vec<ProcessStats>>>,  // Documented clearly
}
```

**Benefits:**
- ✅ Minimal changes
- ✅ Name still makes sense (hives send the data)
- ❌ Less clear that it stores workers too

---

## Recommended Action: Option A (Rename)

### Step 1: Rename Crate

```bash
mv bin/15_queen_rbee_crates/hive-registry \
   bin/15_queen_rbee_crates/telemetry-registry
```

### Step 2: Update `Cargo.toml`

```toml
[package]
name = "queen-rbee-telemetry-registry"
```

### Step 3: Update Imports

```rust
// Before
use queen_rbee_hive_registry::HiveRegistry;

// After
use queen_rbee_telemetry_registry::TelemetryRegistry;
```

### Step 4: Delete `worker-registry`

```bash
rm -rf bin/15_queen_rbee_crates/worker-registry
```

### Step 5: Update All Usage Sites

**Files to update:**
- `bin/10_queen_rbee/src/main.rs`
- `bin/10_queen_rbee/src/http/heartbeat.rs`
- `bin/10_queen_rbee/src/http/heartbeat_stream.rs`
- `bin/10_queen_rbee/src/hive_subscriber.rs`
- `bin/10_queen_rbee/src/job_router.rs`
- `bin/10_queen_rbee/Cargo.toml`

### Step 6: Fix Naming Confusion

```rust
// BEFORE (confusing):
pub struct JobState {
    pub hive_registry: Arc<queen_rbee_worker_registry::WorkerRegistry>,  // WRONG!
}

// AFTER (clear):
pub struct JobState {
    pub telemetry: Arc<queen_rbee_telemetry_registry::TelemetryRegistry>,
}
```

---

## Impact Analysis

### Files Affected: ~10 files

1. `bin/15_queen_rbee_crates/hive-registry/` → rename to `telemetry-registry/`
2. `bin/15_queen_rbee_crates/worker-registry/` → DELETE
3. `bin/10_queen_rbee/Cargo.toml` - update dependencies
4. `bin/10_queen_rbee/src/main.rs` - update imports + variable names
5. `bin/10_queen_rbee/src/http/heartbeat.rs` - update imports
6. `bin/10_queen_rbee/src/http/heartbeat_stream.rs` - update imports
7. `bin/10_queen_rbee/src/hive_subscriber.rs` - update imports
8. `bin/10_queen_rbee/src/job_router.rs` - update imports + field names
9. `bin/15_queen_rbee_crates/scheduler/` - update imports (if used)

### LOC Saved: ~289 lines (delete worker-registry)

### Compilation Risk: LOW
- All changes are renames
- Type system catches all errors
- No logic changes

---

## Alternative: Keep Current Names, Document Better

If renaming is too disruptive, we can:

1. **Keep `HiveRegistry`** - Document that it stores workers too
2. **Delete `WorkerRegistry`** - It's unused
3. **Fix field names** - Rename `hive_registry: WorkerRegistry` → `hive_registry: HiveRegistry`

**Documentation:**
```rust
/// Hive registry
///
/// Stores:
/// 1. Hive heartbeats (HiveInfo + timestamp)
/// 2. Worker telemetry (ProcessStats per hive)
///
/// Workers are monitored BY hives, not directly by Queen.
/// Hives send worker telemetry via heartbeats or SSE streams.
pub struct HiveRegistry {
    hives: HeartbeatRegistry<HiveHeartbeat>,
    workers: RwLock<HashMap<String, Vec<ProcessStats>>>,
}
```

---

## Decision Matrix

| Option | Clarity | Effort | Risk | Recommended |
|--------|---------|--------|------|-------------|
| A: Rename to TelemetryRegistry | ⭐⭐⭐⭐⭐ | Medium | Low | ✅ YES |
| B: Keep HiveRegistry | ⭐⭐⭐ | Low | Low | ⚠️ OK |
| C: Do nothing | ⭐ | None | None | ❌ NO |

---

## Summary

**Problem:** 3 registries with confusing names and overlapping responsibilities.

**Reality:**
- `heartbeat-registry` - Generic foundation ✅
- `hive-registry` - Stores hives AND workers ⚠️
- `worker-registry` - Unused, workers don't send heartbeats ❌

**Solution:** Rename `hive-registry` → `telemetry-registry`, delete `worker-registry`.

**Benefit:** Single source of truth, clear naming, scheduler has one registry to query.

**Next Steps:**
1. Decide: Option A (rename) or Option B (keep name)
2. Delete `worker-registry` (unused)
3. Fix field naming confusion (`hive_registry: WorkerRegistry` → proper type)
4. Update all imports
5. Verify compilation

---

**Estimated Effort:** 1-2 hours  
**Risk:** Low (type system catches everything)  
**Value:** High (eliminates confusion)
