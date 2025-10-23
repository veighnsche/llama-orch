# TEAM-261: Next Steps - Hive Simplification

**Date:** Oct 23, 2025  
**Status:** 📋 READY TO IMPLEMENT  
**Team:** TEAM-261

---

## Summary

We've decided to **keep hive as daemon** but **remove hive heartbeat**.

**Documentation Updated:**
- ✅ `TEAM_261_ARCHITECTURE_CLARITY.md` - Added decision
- ✅ `TEAM_261_COMPLETE_SUMMARY.md` - Added decision
- ✅ `TEAM_261_PHASE_1_COMPLETE.md` - Added decision
- ✅ `TEAM_261_INVESTIGATION_REPORT.md` - Added decision
- ✅ `bin/20_rbee_hive/README.md` - Added decision

**New Documents Created:**
- ✅ `TEAM_261_PIVOT_PROS.md` (520 LOC) - Arguments FOR CLI
- ✅ `TEAM_261_PIVOT_CONS.md` (520 LOC) - Arguments AGAINST CLI
- ✅ `TEAM_261_PIVOT_DECISION_MATRIX.md` (300 LOC) - Decision framework
- ✅ `TEAM_261_SIMPLIFICATION_AUDIT.md` (400 LOC) - Implementation guide

---

## What to Remove from rbee-hive

### 1. Hive Heartbeat Task (main.rs)

**Lines to Remove:**
```rust
// Lines 40-46: CLI args
#[arg(long, default_value = "localhost")]
hive_id: String,

#[arg(long, default_value = "http://localhost:8500")]
queen_url: String,

// Lines 49-59: HiveWorkerProvider
struct HiveWorkerProvider;
impl WorkerStateProvider for HiveWorkerProvider { ... }

// Lines 76-92: Heartbeat initialization
let heartbeat_config = HiveHeartbeatConfig::new(...);
let _heartbeat_handle = start_hive_heartbeat_task(...);
NARRATE.action(ACTION_HEARTBEAT)...
```

### 2. Heartbeat Module (heartbeat.rs)

**Action:** DELETE entire file (80 LOC)

### 3. Heartbeat Imports (main.rs)

**Lines to Remove:**
```rust
// Line 24-26
use rbee_heartbeat::{
    start_hive_heartbeat_task, HiveHeartbeatConfig, WorkerState, WorkerStateProvider,
};

// Line 18
ACTION_HEARTBEAT  // from narration imports
```

### 4. Heartbeat Dependency (Cargo.toml)

**Line to Consider:**
```toml
# Line 30 - Check if used elsewhere first
rbee-heartbeat = { path = "../99_shared_crates/heartbeat" }
```

**Note:** Only remove if not used by other parts of hive

### 5. Narration Constants (narration.rs)

**Check and potentially remove:**
```rust
pub const ACTION_HEARTBEAT: &str = "heartbeat";
```

---

## What to Keep

### ✅ HTTP Server
- `/health` endpoint
- `/capabilities` endpoint
- `/v1/jobs` endpoint (POST)
- `/v1/jobs/{job_id}/stream` endpoint (GET)

### ✅ Job Server Pattern
- `job_router.rs` (267 LOC)
- `http/jobs.rs` (135 LOC)
- `http/mod.rs` (5 LOC)

### ✅ Capabilities Detection
- GPU detection via nvidia-smi
- CPU detection
- Device enumeration

### ✅ Narration
- Observability events
- SSE routing

---

## Current File Structure

```
bin/20_rbee_hive/src/
├── main.rs (219 LOC)          ← MODIFY (remove ~30 lines)
├── heartbeat.rs (80 LOC)      ← DELETE
├── job_router.rs (267 LOC)    ← KEEP
├── http/
│   ├── mod.rs (5 LOC)         ← KEEP
│   └── jobs.rs (135 LOC)      ← KEEP
├── narration.rs (40 LOC)      ← MODIFY (remove ACTION_HEARTBEAT)
└── lib.rs (15 LOC)            ← KEEP
```

**After Simplification:**
```
bin/20_rbee_hive/src/
├── main.rs (~190 LOC)         ← Simplified
├── job_router.rs (267 LOC)    ← Unchanged
├── http/
│   ├── mod.rs (5 LOC)         ← Unchanged
│   └── jobs.js (135 LOC)      ← Unchanged
├── narration.rs (~35 LOC)     ← Simplified
└── lib.rs (15 LOC)            ← Unchanged
```

**Lines Removed:** ~80 LOC (heartbeat.rs) + ~30 LOC (main.rs) = ~110 LOC

---

## Implementation Checklist

### Phase 1: Remove Hive Heartbeat (30 min)

- [ ] Remove CLI args from main.rs (hive_id, queen_url)
- [ ] Remove HiveWorkerProvider struct
- [ ] Remove heartbeat initialization code
- [ ] Remove heartbeat imports
- [ ] Delete heartbeat.rs file
- [ ] Remove ACTION_HEARTBEAT from narration.rs
- [ ] Check if rbee-heartbeat dependency can be removed
- [ ] Test compilation: `cargo check -p rbee-hive`

### Phase 2: Update Worker (1 hour)

- [ ] Change worker heartbeat target to queen
- [ ] Update worker CLI args (queen_url instead of hive_url)
- [ ] Update worker heartbeat payload
- [ ] Test worker heartbeat to queen

### Phase 3: Update Queen (2 hours)

- [ ] Add /v1/worker-heartbeat endpoint
- [ ] Create worker registry in queen
- [ ] Track workers directly
- [ ] Remove hive heartbeat endpoint
- [ ] Test heartbeat reception

### Phase 4: Test End-to-End (1 hour)

- [ ] Start queen
- [ ] Start hive (no heartbeat)
- [ ] Start worker
- [ ] Verify worker heartbeat goes to queen
- [ ] Verify hive operations work
- [ ] Verify SSE streaming works

---

## Expected Results

### Before
```
Hive startup:
🐝 Starting on port 9000, hive_id: localhost, queen: http://localhost:8500
💓 Heartbeat task started (5s interval)
✅ Listening on http://127.0.0.1:9000
✅ Hive ready

Hive sends heartbeat to queen every 5s
```

### After
```
Hive startup:
🐝 Starting on port 9000
✅ Listening on http://127.0.0.1:9000
✅ Hive ready

No heartbeat task
Workers send heartbeat to queen directly
```

---

## Benefits

### Performance
- ✅ Same fast HTTP (1-5ms)
- ✅ Same real-time SSE streaming

### Simplicity
- ✅ ~110 LOC removed from hive
- ✅ No heartbeat aggregation
- ✅ No state synchronization
- ✅ Single source of truth (queen)

### Architecture
- ✅ Clearer responsibilities
- ✅ Workers → queen direct
- ✅ Hive only manages lifecycle

---

## Testing Strategy

### Unit Tests
- [ ] Test hive startup (no heartbeat)
- [ ] Test HTTP endpoints
- [ ] Test capabilities detection
- [ ] Test job routing

### Integration Tests
- [ ] Test worker spawn
- [ ] Test worker list
- [ ] Test model operations
- [ ] Test SSE streaming

### End-to-End Tests
- [ ] Test keeper → queen → hive flow
- [ ] Test worker heartbeat to queen
- [ ] Test hive operations without heartbeat

---

## Risk Assessment

### Low Risk ✅
- Removing hive heartbeat is safe
- Workers already have heartbeat code
- Just changing target URL

### Medium Risk ⚠️
- Queen needs new endpoint
- Queen needs worker registry
- Must coordinate changes

### Mitigation
- Implement in phases
- Test each phase independently
- Keep old code in git history

---

## Timeline

**Total Estimated Time:** ~5 hours

1. **Phase 1:** Remove hive heartbeat (30 min)
2. **Phase 2:** Update worker (1 hour)
3. **Phase 3:** Update queen (2 hours)
4. **Phase 4:** Test end-to-end (1 hour)
5. **Buffer:** 30 min

---

## Success Criteria

- ✅ Hive compiles without heartbeat code
- ✅ Hive starts and responds to HTTP requests
- ✅ Workers send heartbeats to queen
- ✅ Queen tracks workers
- ✅ Hive operations work (spawn, list, etc.)
- ✅ SSE streaming works
- ✅ All tests pass

---

**TEAM-261 Next Steps**  
**Date:** Oct 23, 2025  
**Status:** 📋 READY TO IMPLEMENT  
**Next:** Remove hive heartbeat from main.rs
