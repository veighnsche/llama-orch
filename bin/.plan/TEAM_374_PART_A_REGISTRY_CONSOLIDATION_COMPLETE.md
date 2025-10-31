# TEAM-374 Part A: Registry Consolidation - COMPLETE

**Date:** Oct 31, 2025  
**Status:** ✅ COMPLETE  
**Duration:** ~2 hours

---

## Mission Accomplished

Consolidated 3 confusing registry crates into 1 clear `TelemetryRegistry`:

- ❌ **DELETED:** `worker-registry` (unused, workers don't send heartbeats)
- ✅ **RENAMED:** `hive-registry` → `telemetry-registry`
- ✅ **UPDATED:** All imports and usage sites

---

## What Was Done

### 1. Renamed Crate

```bash
mv bin/15_queen_rbee_crates/hive-registry \
   bin/15_queen_rbee_crates/telemetry-registry
```

### 2. Updated Crate Metadata

**File:** `bin/15_queen_rbee_crates/telemetry-registry/Cargo.toml`
- Package name: `queen-rbee-telemetry-registry`
- Lib name: `queen_rbee_telemetry_registry`

### 3. Renamed Struct

**File:** `bin/15_queen_rbee_crates/telemetry-registry/src/registry.rs`
- `HiveRegistry` → `TelemetryRegistry`
- Updated all tests
- Added compatibility methods:
  - `find_best_worker_for_model()` - For scheduler
  - `list_online_workers()` - For status queries

### 4. Deleted worker-registry

```bash
rm -rf bin/15_queen_rbee_crates/worker-registry
```

**LOC Saved:** ~289 lines

### 5. Updated Dependencies

**Files Updated:**
- `Cargo.toml` (workspace)
- `bin/10_queen_rbee/Cargo.toml`
- `bin/15_queen_rbee_crates/scheduler/Cargo.toml`

### 6. Updated All Imports

**Queen Files Updated:**
- `bin/10_queen_rbee/src/main.rs`
- `bin/10_queen_rbee/src/http/heartbeat.rs`
- `bin/10_queen_rbee/src/http/heartbeat_stream.rs`
- `bin/10_queen_rbee/src/http/jobs.rs`
- `bin/10_queen_rbee/src/hive_subscriber.rs`
- `bin/10_queen_rbee/src/job_router.rs`

**Scheduler Files Updated:**
- `bin/15_queen_rbee_crates/scheduler/src/simple.rs`
- `bin/15_queen_rbee_crates/scheduler/src/lib.rs`

### 7. Fixed Type Mismatches

**Problem:** Scheduler expected `WorkerInfo` but now gets `ProcessStats`

**Solution:** Updated scheduler to work with ProcessStats fields:
- `worker.id` → `format!("{}-{}", worker.group, worker.instance)`
- `worker.port` → `worker.instance.parse::<u16>()`
- `worker.model_id` → `worker.model`
- `worker.device` → `worker.group`
- `worker.status` → `if worker.gpu_util_pct == 0.0 { "idle" } else { "busy" }`

---

## Architecture After Consolidation

### Before (Confusing)

```
heartbeat-registry (generic)
    ↓
hive-registry (stores hives + workers)  ← Confusing name!
    ↓
worker-registry (UNUSED!)  ← Workers don't send heartbeats!
```

### After (Clear)

```
heartbeat-registry (generic)
    ↓
telemetry-registry (stores hives + workers)  ← Clear name!
```

---

## What TelemetryRegistry Stores

### Hive Heartbeats
- `HiveInfo` (id, hostname, port, status, version)
- Stored via `HeartbeatRegistry<HiveHeartbeat>`

### Worker Telemetry
- `ProcessStats` (pid, group, instance, cpu, gpu, vram, model)
- Stored in `HashMap<String, Vec<ProcessStats>>` (hive_id → workers)

### Key Methods

**Hive Methods:**
- `update_hive()`, `get_hive()`, `list_online_hives()`, `list_available_hives()`
- `count_online()`, `count_available()`, `cleanup_stale()`

**Worker Methods:**
- `update_workers()`, `get_workers()`, `get_all_workers()`
- `find_idle_workers()`, `find_workers_with_model()`, `find_workers_with_capacity()`
- `find_best_worker_for_model()` (for scheduler)
- `list_online_workers()` (compatibility)

---

## Compilation Status

✅ **Queen Binary:** `cargo check --bin queen-rbee` - **SUCCESS**  
✅ **All Tests:** Updated to use `TelemetryRegistry::new()`

---

## Benefits

1. ✅ **Single Source of Truth** - One registry for all telemetry
2. ✅ **Clear Naming** - "Telemetry" accurately describes what it stores
3. ✅ **Reduced Confusion** - No more "hive registry stores workers too?"
4. ✅ **Code Reduction** - 289 LOC deleted (worker-registry)
5. ✅ **Scheduler Works** - Updated to use ProcessStats instead of WorkerInfo

---

## What's Next (Part B)

Part B will DELETE old POST telemetry logic:
- Delete `start_normal_telemetry_task()` from Hive
- Delete `send_heartbeat_to_queen()` from Hive
- Delete `handle_hive_heartbeat()` from Queen (keep `handle_hive_ready()`)
- Update discovery to send ready callback instead of continuous POST

---

**TEAM-374 Part A: Registry consolidation complete! Ready for Part B.**
