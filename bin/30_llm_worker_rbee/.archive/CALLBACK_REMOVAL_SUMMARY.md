# Callback Feature Removal Summary

**Date:** 2025-10-19  
**Reason:** Simplified architecture - workers now only use heartbeat mechanism

---

## Changes Made

### 1. Removed Callback Module

**Deleted:**
- `src/common/startup.rs` (318 lines) - Entire callback implementation including tests

**Reason:** Callback feature replaced with heartbeat-only approach

---

### 2. Updated All Binaries

Changed CLI argument from `--callback-url` to `--hive-url`:

#### Main Binary (`src/main.rs`)
- ❌ Removed `callback_ready` import
- ❌ Removed `ACTION_CALLBACK_READY` constant usage
- ✅ Changed `--callback-url` → `--hive-url`
- ✅ Removed callback logic (lines 180-257)
- ✅ Replaced with heartbeat task startup (lines 185-195)
- ✅ Updated flow comments (Step 2 → Step 3)

#### CPU Binary (`src/bin/cpu.rs`)
- ❌ Removed `callback_ready` import
- ✅ Changed `--callback-url` → `--hive-url`
- ✅ Removed callback logic and test mode check
- ✅ Added heartbeat task startup

#### CUDA Binary (`src/bin/cuda.rs`)
- ❌ Removed `callback_ready` import
- ✅ Changed `--callback-url` → `--hive-url`
- ✅ Removed callback logic and test mode check
- ✅ Added heartbeat task startup

#### Metal Binary (`src/bin/metal.rs`)
- ❌ Removed `callback_ready` import
- ✅ Changed `--callback-url` → `--hive-url`
- ✅ Removed callback logic and test mode check
- ✅ Added heartbeat task startup

---

### 3. Updated Module Exports

#### `src/lib.rs`
- ❌ Removed `callback_ready` from re-exports
- ✅ Kept other common types (InferenceResult, SamplingConfig, etc.)

#### `src/common/mod.rs`
- ❌ Removed `pub mod startup;`
- ❌ Removed `pub use startup::callback_ready;`

---

### 4. Cleaned Up Narration Constants

#### `src/narration.rs`
- ❌ Removed `ACTION_CALLBACK_READY` constant (line 79)
- ❌ Removed `ACTION_CALLBACK_ERROR` constant (line 106)
- ❌ Removed `ACTION_TEST_MODE` constant (line 109)

**Reason:** These constants were only used for callback feature

---

### 5. Updated Documentation

#### `docs/metal.md`
- ✅ Changed all `--callback-url` examples → `--hive-url`
- ✅ Updated CLI arguments table
- ✅ Removed "test mode" callback skip explanation
- ✅ Changed "Pool manager callback URL" → "Hive URL for heartbeats"
- ✅ Updated example commands (3 locations)

---

## New Architecture

### Before (Callback + Heartbeat)
```
1. Worker starts
2. Load model
3. ❌ Callback to hive (POST /ready) - REMOVED
4. Start heartbeat task
5. Start HTTP server
```

### After (Heartbeat Only)
```
1. Worker starts
2. Load model
3. Start heartbeat task ✅
4. Start HTTP server
```

### Heartbeat Implementation
```rust
let heartbeat_config = llm_worker_rbee::heartbeat::HeartbeatConfig::new(
    args.worker_id.clone(),
    args.hive_url.clone(),
);
let _heartbeat_handle = llm_worker_rbee::heartbeat::start_heartbeat_task(heartbeat_config);
tracing::info!("Heartbeat task started (30s interval)");
```

**Location:** Uses shared `rbee_heartbeat` crate from `bin/99_shared_crates/heartbeat/`

---

## CLI Changes

### Old Arguments
```bash
llm-worker-rbee \
  --worker-id <uuid> \
  --model <path> \
  --port <port> \
  --callback-url http://localhost:8600 \  # ❌ REMOVED
  --backend <backend> \
  --device <device>
```

### New Arguments
```bash
llm-worker-rbee \
  --worker-id <uuid> \
  --model <path> \
  --port <port> \
  --hive-url http://localhost:8600 \  # ✅ NEW
  --backend <backend> \
  --device <device>
```

**Change:** `--callback-url` → `--hive-url`  
**Purpose:** URL where worker sends periodic heartbeats (not one-time callback)

---

## Files Modified

### Source Files (7)
1. `src/main.rs` - Main binary
2. `src/lib.rs` - Module exports
3. `src/common/mod.rs` - Common module exports
4. `src/narration.rs` - Narration constants
5. `src/bin/cpu.rs` - CPU binary
6. `src/bin/cuda.rs` - CUDA binary
7. `src/bin/metal.rs` - Metal binary

### Documentation (1)
8. `docs/metal.md` - Metal backend docs

### Deleted (1)
9. `src/common/startup.rs` - Callback implementation (318 lines deleted)

---

## Verification

### Build Status
```bash
cargo check -p llm-worker-rbee
```
**Result:** ✅ Success (exit code 0)

### Test Status
All callback-related tests removed with `startup.rs` module.

---

## Migration Notes

### For rbee-hive

**Before:** Expected callback from worker at startup:
```rust
POST /v1/workers/ready
{
  "worker_id": "...",
  "url": "http://localhost:8601",
  "model_ref": "...",
  "backend": "cuda",
  "device": 0
}
```

**After:** Only receives periodic heartbeats:
```rust
POST /heartbeat/workers
{
  "worker_id": "...",
  "state": "Ready",
  "model_loaded": true
}
```

**Action Required:** Remove `/v1/workers/ready` endpoint from rbee-hive HTTP server.

---

## Benefits

1. **Simpler Architecture:** Single communication mechanism (heartbeat only)
2. **Less Code:** Removed 318 lines of callback logic + tests
3. **Consistent Pattern:** All workers use same heartbeat mechanism
4. **Better Reliability:** Periodic heartbeats detect failures faster than one-time callback
5. **Cleaner API:** No special "test mode" logic (`localhost:9999` check removed)

---

## Breaking Changes

⚠️ **CLI Argument Change:** `--callback-url` → `--hive-url`

**Impact:** Any scripts or tools spawning workers must update argument name.

**Example Migration:**
```bash
# Old
./llm-worker-rbee --callback-url http://localhost:8600

# New
./llm-worker-rbee --hive-url http://localhost:8600
```

---

**Status:** ✅ Complete  
**Build:** ✅ Passing  
**Next Steps:** Update rbee-hive to remove `/v1/workers/ready` endpoint
