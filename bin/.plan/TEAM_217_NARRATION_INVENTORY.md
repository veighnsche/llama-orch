# TEAM-217: queen-rbee NARRATION INVENTORY

**Component:** `bin/10_queen_rbee`  
**Date:** Oct 22, 2025  
**Status:** ✅ COMPLETE

---

## Summary

queen-rbee uses narration in 2 locations:
1. **main.rs** - Server lifecycle (startup, listen, ready, errors)
2. **job_router.rs** - Job routing and operation dispatch

**CRITICAL:** ALL narration in job_router.rs includes `.job_id(&job_id)` for SSE routing.

---

## 1. main.rs Narration

**Factory:** `const NARRATE: NarrationFactory = NarrationFactory::new("queen");`

### Startup Narrations (NO job_id)

```rust
// Line 62-66: Server starting
NARRATE
    .action("start")
    .context(args.port.to_string())
    .human("Queen-rbee starting on port {}")
    .emit();

// Line 78-82: Config loaded
NARRATE
    .action("start")
    .context(config_dir.display().to_string())
    .human("Loaded config from {}")
    .emit();

// Line 99: Listening
NARRATE.action("listen").context(addr.to_string()).human("Listening on http://{}").emit();

// Line 101: Ready
NARRATE.action("ready").human("Ready to accept connections").emit();
```

### Error Narrations (NO job_id)

```rust
// Line 105-110: Server error
NARRATE
    .action("error")
    .context(e.to_string())
    .human("Server error: {}")
    .error_kind("server_failed")
    .emit();
```

**Behavior:** These narrations go to **stderr** (no job_id = no SSE routing).

---

## 2. job_router.rs Narration

**Factory:** `const NARRATE: NarrationFactory = NarrationFactory::new("qn-router");`

### Job Creation (WITH job_id)

```rust
// Line 73-78: Job created
NARRATE
    .action("job_create")
    .context(&job_id)
    .job_id(&job_id) // ← CRITICAL for SSE routing
    .human("Job {} created, waiting for client connection")
    .emit();
```

### Operation Routing (WITH job_id)

```rust
// Line 121-126: Route operation
NARRATE
    .action("route_job")
    .context(operation_name)
    .job_id(&job_id) // ← CRITICAL for SSE routing
    .human("Executing operation: {}")
    .emit();
```

### Status Operation (WITH job_id)

```rust
// Line 137-141: Status request
NARRATE
    .action("status")
    .job_id(&job_id) // ← CRITICAL for SSE routing
    .human("📊 Fetching live status from registry")
    .emit();

// Line 147-150: No active hives
NARRATE
    .action("status_empty")
    .job_id(&job_id) // ← CRITICAL for SSE routing
    .human("⚠️  No active hives found...")
    .emit();

// Line 193-196: Status result
NARRATE
    .action("status_result")
    .job_id(&job_id) // ← CRITICAL for SSE routing
    .context(active_hive_ids.len().to_string())
    .human("📊 Found {} active hive(s)")
    .emit();
```

### SSH Test Operation (WITH job_id)

```rust
// Line 225-228: SSH test success
NARRATE
    .action("ssh_test_ok")
    .job_id(&job_id) // ← CRITICAL for SSE routing
    .context(response.test_output.unwrap_or_default())
    .human("✅ SSH test passed: {}")
    .emit();
```

**Behavior:** ALL job_router narrations include `.job_id(&job_id)` → Routes to SSE channel → Client receives via `/v1/jobs/{job_id}/stream`

---

## 3. Delegation Pattern

queen-rbee **delegates** most narration to:
- `hive-lifecycle` crate (TEAM-220) - All hive operations
- `worker-lifecycle` crate (TEAM-221) - All worker operations (not yet implemented)

**What queen does:**
- Job creation narration
- Operation routing narration
- Status operation narration (aggregates from registry)
- SSH test result narration

**What queen does NOT do:**
- Hive start/stop/install/uninstall narration → Delegated to hive-lifecycle
- Worker spawn/delete narration → Delegated to worker-lifecycle
- Model download narration → Delegated to model-lifecycle (not yet implemented)

---

## 4. job_id Propagation

**Flow:**
```
POST /v1/jobs → create_job() → Generate job_id → Create SSE channel
                                      ↓
GET /v1/jobs/{job_id}/stream → execute_job() → route_operation()
                                      ↓
                        Pass job_id to hive-lifecycle functions
                                      ↓
                        hive-lifecycle includes .job_id() in ALL narrations
                                      ↓
                        Narrations route to job-specific SSE channel
                                      ↓
                        Client receives via SSE stream
```

**CRITICAL:** job_id MUST be passed to ALL delegated functions (hive-lifecycle, worker-lifecycle, etc.)

---

## 5. Narration Destinations

### Stderr (NO job_id)
- Server startup messages (main.rs)
- Server error messages (main.rs)
- Health check responses (http/health.rs - not investigated)

### SSE (WITH job_id)
- Job creation (job_router.rs)
- Operation routing (job_router.rs)
- Status operation (job_router.rs)
- SSH test results (job_router.rs)
- ALL hive-lifecycle narrations (delegated)
- ALL worker-lifecycle narrations (delegated)

---

## 6. Findings

### ✅ Correct Behaviors
1. **Consistent job_id usage** - ALL job_router narrations include `.job_id(&job_id)`
2. **Clean delegation** - Passes job_id to hive-lifecycle functions
3. **Proper separation** - Server lifecycle (stderr) vs job execution (SSE)
4. **SSE channel creation** - Creates job-specific channel before execution

### ⚠️ Potential Issues
1. **No correlation_id** - Not yet implemented (see NARRATION_AND_JOB_ID_ARCHITECTURE.md)
2. **No narration in http/ modules** - Health/heartbeat endpoints don't narrate
3. **Status operation complexity** - Aggregates from registry, could be moved to separate module

### 📋 Recommendations
1. Add correlation_id support when implementing end-to-end tracing
2. Consider adding narration to health/heartbeat endpoints (stderr only, no job_id)
3. Monitor SSE channel cleanup (channels created but cleanup not visible in this code)

---

## 7. Code Signatures

All investigated code marked with:
```rust
//! TEAM-217: Investigated Oct 22, 2025 - Behavior inventory complete
```

**Files investigated:**
- `bin/10_queen_rbee/src/main.rs` (lines 1-160)
- `bin/10_queen_rbee/src/job_router.rs` (lines 1-375)
- `bin/10_queen_rbee/src/narration.rs` (not read, only factory definition)

---

**TEAM-217 COMPLETE** ✅
