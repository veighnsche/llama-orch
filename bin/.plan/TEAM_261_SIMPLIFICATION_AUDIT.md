# TEAM-261: Hive Simplification Audit

**Date:** Oct 23, 2025  
**Decision:** Keep daemon, remove hive heartbeat, simplify state  
**Status:** 🔍 AUDIT

---

## Current State of rbee-hive

### Files Present
```
bin/20_rbee_hive/src/
├── main.rs (219 LOC)
├── heartbeat.rs (80 LOC) ← TO REMOVE
├── job_router.rs (267 LOC) ← KEEP
├── http/
│   ├── mod.rs (5 LOC) ← KEEP
│   └── jobs.rs (135 LOC) ← KEEP
├── narration.rs (40 LOC) ← KEEP
└── lib.rs (15 LOC) ← KEEP
```

### Current Features

#### ✅ KEEP: Core Functionality
1. **HTTP Server** (main.rs lines 103-124)
   - `/health` endpoint
   - `/capabilities` endpoint
   - `/v1/jobs` endpoint (POST)
   - `/v1/jobs/{job_id}/stream` endpoint (GET)

2. **Job Server Pattern** (job_router.rs)
   - Operation routing
   - Worker lifecycle operations (TODO)
   - Model management operations (TODO)
   - SSE streaming

3. **Capabilities Detection** (main.rs lines 153-218)
   - GPU detection via nvidia-smi
   - CPU detection
   - Device enumeration

4. **Narration** (narration.rs)
   - Observability events
   - SSE routing

#### ❌ REMOVE: Hive Heartbeat
1. **Heartbeat Task** (main.rs lines 76-92)
   - `start_hive_heartbeat_task()`
   - 5 second interval
   - Aggregates worker states
   - Sends to queen

2. **Heartbeat Module** (heartbeat.rs)
   - `handle_worker_heartbeat()`
   - `send_heartbeat_to_queen()`
   - Worker state aggregation

3. **CLI Args** (main.rs lines 40-46)
   - `--hive-id` argument
   - `--queen-url` argument

4. **Dependencies** (Cargo.toml line 30)
   - `rbee-heartbeat` crate

---

## What to Remove

### 1. Hive Heartbeat Task (main.rs)

**Lines to Remove:**
- Lines 40-46: CLI args for hive_id and queen_url
- Lines 49-59: HiveWorkerProvider struct
- Lines 76-92: Heartbeat initialization and startup

**Before:**
```rust
/// TEAM-190: Hive ID (defaults to "localhost")
#[arg(long, default_value = "localhost")]
hive_id: String,

/// TEAM-190: Queen URL for heartbeat reporting
#[arg(long, default_value = "http://localhost:8500")]
queen_url: String,

// ...

// TEAM-190: Start heartbeat task (5 second interval)
let heartbeat_config = HiveHeartbeatConfig::new(
    args.hive_id.clone(),
    args.queen_url.clone(),
    "".to_string(),
)
.with_interval(5);

let worker_provider = Arc::new(HiveWorkerProvider);
let _heartbeat_handle = start_hive_heartbeat_task(heartbeat_config, worker_provider);

// TEAM-202: Narrate heartbeat startup
NARRATE
    .action(ACTION_HEARTBEAT)
    .context("5s")
    .human("💓 Heartbeat task started ({} interval)")
    .emit();
```

**After:**
```rust
// No hive_id or queen_url args
// No heartbeat task
// Just port argument
```

### 2. Heartbeat Module (heartbeat.rs)

**Action:** DELETE entire file

**Reason:** Workers send heartbeats directly to queen, not through hive

### 3. Heartbeat Imports (main.rs)

**Lines to Remove:**
- Line 24-26: `use rbee_heartbeat::{...}`
- Line 18: `ACTION_HEARTBEAT` from narration

### 4. Heartbeat Dependency (Cargo.toml)

**Line to Remove:**
- Line 30: `rbee-heartbeat = { path = "../99_shared_crates/heartbeat" }`

**Note:** Keep the dependency if it's used elsewhere, but remove if only for hive heartbeat

---

## What to Keep

### 1. HTTP Server ✅
- Fast operations (1-5ms)
- Real-time SSE streaming
- Consistent with queen/worker

### 2. Job Server Pattern ✅
- Operation routing
- SSE streaming
- Job isolation

### 3. Capabilities Endpoint ✅
- GPU detection
- CPU detection
- Called by queen during hive start

### 4. Worker Lifecycle Operations ✅
- WorkerSpawn (TODO)
- WorkerStop (TODO)
- WorkerList (TODO)
- Model operations (TODO)

---

## Architectural Decision

### Why Remove Hive Heartbeat?

**Old Architecture:**
```
Workers → Hive (aggregates) → Queen
         (heartbeat)         (heartbeat)
```

**New Architecture:**
```
Workers → Queen (direct)
         (heartbeat)
```

**Benefits:**
1. ✅ **Simpler:** No aggregation logic
2. ✅ **Faster:** Direct communication
3. ✅ **Single source of truth:** Queen knows all workers
4. ✅ **No state sync:** No distributed state
5. ✅ **Less overhead:** No hive heartbeat task

### Why Keep Daemon?

**Reasons:**
1. ✅ **Performance:** 1-5ms vs 80-350ms (SSH)
2. ✅ **UX:** Real-time SSE streaming
3. ✅ **Security:** No command injection risk
4. ✅ **Consistency:** Matches queen/worker patterns

---

## Impact Analysis

### Files to Modify
1. `bin/20_rbee_hive/src/main.rs` (~30 lines removed)
2. `bin/20_rbee_hive/src/heartbeat.rs` (DELETE)
3. `bin/20_rbee_hive/Cargo.toml` (~1 line removed)
4. `bin/20_rbee_hive/src/narration.rs` (remove ACTION_HEARTBEAT)

### Files to Keep Unchanged
1. `bin/20_rbee_hive/src/job_router.rs` ✅
2. `bin/20_rbee_hive/src/http/jobs.rs` ✅
3. `bin/20_rbee_hive/src/http/mod.rs` ✅
4. `bin/20_rbee_hive/src/lib.rs` ✅

### Dependencies to Remove
1. `rbee-heartbeat` (if only used for hive heartbeat)

### Dependencies to Keep
1. `axum` ✅
2. `job-server` ✅
3. `rbee-operations` ✅
4. `observability-narration-core` ✅
5. `rbee-hive-device-detection` ✅

---

## Worker Changes Needed

### Workers Must Send Heartbeats to Queen

**Current (via Hive):**
```rust
// Worker sends to hive
POST http://hive:9000/v1/heartbeat
```

**New (Direct to Queen):**
```rust
// Worker sends to queen
POST http://queen:8500/v1/worker-heartbeat
```

**Files to Update:**
- `bin/30_llm_worker_rbee/src/heartbeat.rs`
- Worker must know queen URL (not hive URL)

---

## Queen Changes Needed

### Queen Must Accept Worker Heartbeats

**New Endpoint:**
```rust
// bin/10_queen_rbee/src/http/heartbeat.rs
POST /v1/worker-heartbeat
{
    "worker_id": "worker-123",
    "hive_id": "localhost",
    "status": "running",
    "model": "llama-2-7b",
    "device": "GPU-0",
    "timestamp_ms": 1234567890
}
```

**New Registry:**
```rust
// bin/10_queen_rbee/src/worker_registry.rs
struct WorkerRegistry {
    workers: HashMap<String, WorkerInfo>,
}
```

---

## Testing Impact

### Tests to Update
1. Hive startup tests (no heartbeat)
2. Worker spawn tests (no heartbeat aggregation)
3. Integration tests (workers → queen direct)

### Tests to Remove
1. Hive heartbeat tests
2. Worker aggregation tests

---

## Documentation Updates Needed

### Existing Docs to Update
1. `bin/.plan/TEAM_261_INVESTIGATION_REPORT.md` ← Add decision
2. `bin/.plan/TEAM_261_PHASE_1_COMPLETE.md` ← Add decision
3. `bin/.plan/TEAM_261_ARCHITECTURE_CLARITY.md` ← Add decision
4. `bin/.plan/TEAM_261_COMPLETE_SUMMARY.md` ← Add decision
5. `README.md` (if exists) ← Update architecture

### What to Document
- ✅ Decision to keep daemon
- ✅ Decision to remove hive heartbeat
- ✅ Workers send heartbeats to queen directly
- ✅ Hive only manages worker lifecycle
- ✅ Queen is single source of truth

---

## Implementation Steps

### Phase 1: Remove Hive Heartbeat (1 hour)
1. Remove heartbeat task from main.rs
2. Remove heartbeat.rs file
3. Remove heartbeat imports
4. Remove CLI args (hive_id, queen_url)
5. Update Cargo.toml
6. Test compilation

### Phase 2: Update Workers (1 hour)
1. Change worker heartbeat target to queen
2. Update worker CLI args (queen_url instead of hive_url)
3. Test worker heartbeat to queen

### Phase 3: Update Queen (2 hours)
1. Add /v1/worker-heartbeat endpoint
2. Create worker registry
3. Track workers directly
4. Test heartbeat reception

### Phase 4: Update Documentation (1 hour)
1. Update existing docs with decision
2. Add architectural notes
3. Update README

**Total Time:** ~5 hours

---

## Success Criteria

### After Simplification
- ✅ Hive daemon still runs
- ✅ Hive responds to HTTP requests (1-5ms)
- ✅ Hive provides real-time SSE streaming
- ✅ No hive heartbeat task
- ✅ Workers send heartbeats to queen
- ✅ Queen tracks all workers
- ✅ Compilation successful
- ✅ Tests pass

---

**TEAM-261 Simplification Audit**  
**Date:** Oct 23, 2025  
**Status:** 🔍 READY FOR IMPLEMENTATION  
**Next:** Remove hive heartbeat, update docs
