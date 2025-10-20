# TEAM-164: Heartbeat Split - Binary-Specific Logic

**Date:** 2025-10-20  
**Status:** ✅ COMPLETE

---

## What Was Done

### 1. Removed Deprecated Function
- ❌ Deleted `ensure_hive_running()` from hive-lifecycle
- Reason: Misleading name - queen doesn't wait, heartbeat is the callback

### 2. Created Binary-Specific Heartbeat Files

Each binary now has its own `src/heartbeat.rs` with binary-specific logic:

#### `bin/10_queen_rbee/src/heartbeat.rs`
**Purpose:** Queen receives heartbeats from hives

**Functions:**
- `handle_hive_heartbeat()` - Process hive heartbeat
  - Check if first heartbeat
  - Trigger device detection on first heartbeat
  - Update catalog with heartbeat timestamp
  - Return acknowledgement

**Flow:**
```
Hive sends heartbeat (callback when ready)
    ↓
Queen receives heartbeat
    ↓
If first heartbeat → detect devices
    ↓
Update catalog
    ↓
Return acknowledgement
```

#### `bin/20_rbee_hive/src/heartbeat.rs`
**Purpose:** Hive receives heartbeats from workers, sends to queen

**Functions:**
- `handle_worker_heartbeat()` - Process worker heartbeat
- `send_heartbeat_to_queen()` - Send aggregated heartbeat to queen

**Flow:**
```
Worker → Hive: Heartbeat (I'm alive)
    ↓
Hive updates worker registry
    ↓
Hive → Queen: Aggregated heartbeat (all workers)
```

#### `bin/30_llm_worker_rbee/src/heartbeat.rs`
**Purpose:** Worker sends heartbeats to hive

**Functions:**
- `send_heartbeat_to_hive()` - Send heartbeat to hive
- `start_heartbeat_task()` - Periodic heartbeat task (every 30s)

**Flow:**
```
Worker starts
    ↓
Spawn heartbeat task
    ↓
Every 30s → Send heartbeat to hive
```

---

## What Stays in Shared Crate

**`99_shared_crates/heartbeat/`** - Shared types and traits:

- **Types:**
  - `HiveHeartbeatPayload`
  - `WorkerHeartbeatPayload`
  - `HealthStatus`
  - `WorkerState`

- **Traits:**
  - `DeviceDetector`
  - `DeviceResponse`
  - `CpuInfo`
  - `GpuInfo`

- **Common logic:**
  - Serialization/deserialization
  - Type definitions
  - Trait abstractions

---

## Architecture

### Before (Shared Crate Did Everything)
```
rbee-heartbeat (shared)
├── handle_hive_heartbeat() ← Queen logic
├── handle_worker_heartbeat() ← Hive logic
├── send_heartbeat() ← Worker logic
└── types + traits
```

**Problem:** Binary-specific logic mixed with shared code

### After (Binary-Specific Files)
```
queen-rbee/src/heartbeat.rs
├── handle_hive_heartbeat() ← Queen-specific

rbee-hive/src/heartbeat.rs
├── handle_worker_heartbeat() ← Hive-specific
└── send_heartbeat_to_queen() ← Hive-specific

llm-worker-rbee/src/heartbeat.rs
├── send_heartbeat_to_hive() ← Worker-specific
└── start_heartbeat_task() ← Worker-specific

rbee-heartbeat (shared)
└── types + traits ← Shared only
```

**Benefit:** Clear separation, binary-specific logic in binaries

---

## Files Created

1. `/bin/10_queen_rbee/src/heartbeat.rs` - Queen heartbeat logic
2. `/bin/20_rbee_hive/src/heartbeat.rs` - Hive heartbeat logic
3. `/bin/30_llm_worker_rbee/src/heartbeat.rs` - Worker heartbeat logic

## Files Modified

1. `/bin/10_queen_rbee/src/main.rs` - Added `mod heartbeat`
2. `/bin/10_queen_rbee/src/http.rs` - Use local heartbeat module
3. `/bin/15_queen_rbee_crates/hive-lifecycle/src/lib.rs` - Removed deprecated function

---

## Verification

✅ **Build:** `cargo build --bin queen-rbee` - Success  
✅ **Test:** `cargo xtask e2e:hive` - PASSED  
✅ **Architecture:** Binary-specific logic in binaries, shared types in shared crate

---

## Benefits

1. **Clear Separation**
   - Binary logic in binary files
   - Shared types in shared crate

2. **Easy to Find**
   - Queen heartbeat? → `queen-rbee/src/heartbeat.rs`
   - Hive heartbeat? → `rbee-hive/src/heartbeat.rs`
   - Worker heartbeat? → `llm-worker-rbee/src/heartbeat.rs`

3. **No Pollution**
   - Shared crate doesn't have binary-specific logic
   - Binaries don't duplicate shared types

4. **Maintainable**
   - Each binary owns its heartbeat behavior
   - Shared types ensure compatibility

---

**TEAM-164 OUT** 🎯
