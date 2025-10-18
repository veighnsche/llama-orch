# TEAM-096: Port Allocation & Fail-Fast Investigation

**Status:** 🟡 TESTING - Awaiting end-to-end verification

## Problem

When `rbee-hive` spawns multiple workers, subsequent workers fail with "Address already in use (os error 98)" and get stuck in infinite 404 error loops.

```
Caused by:
    Address already in use (os error 98)
2025-10-18T09:34:48.034132Z ERROR rbee_hive::monitor: Worker unhealthy worker_id=worker-70256df4-f1ea-4ca4-97bc-38c9da37613f url=http://127.0.0.1:8081 status=404 Not Found
```

**Root causes:**
1. Port allocation reused ports from failed workers still in registry
2. No fail-fast protocol - workers never removed from registry
3. Insufficient narration during spawn/health checks

## Investigation Results

### Root Cause 1: Broken Port Allocation

**File:** `bin/rbee-hive/src/http/workers.rs` line 144

**Old code:**
```rust
let workers = state.registry.list().await;
let port = 8081 + workers.len() as u16;
```

**Problem:** 
- Port = 8081 + count of ALL workers (including dead ones)
- Worker 1 spawns on 8081 ✅
- Worker 1 dies but stays in registry ❌
- Worker 2 tries 8081 + 1 = 8082 ✅
- Worker 3 tries 8081 + 2 = 8083 ✅
- Worker 1 removed, worker 4 tries 8081 + 2 = 8083 (collision!) ❌

### Root Cause 2: No Fail-Fast Protocol

**File:** `bin/rbee-hive/src/monitor.rs` line 40-56

**Old code:**
```rust
Ok(response) => {
    error!("Worker unhealthy");
    // TODO: Mark worker as unhealthy in registry
}
Err(e) => {
    error!("Worker unreachable");
    // TODO: Mark worker as unhealthy in registry
}
```

**Problem:**
- Failed workers logged forever but never removed
- Registry fills with dead workers
- Port allocation breaks (see Root Cause 1)

### Root Cause 3: Insufficient Narration

**Problem:**
- No logging about port allocation decisions
- No health check summary logs
- Hard to debug port conflicts

## Changes Made

### Fix 1: Smart Port Allocation (TEAM-096)

**File:** `bin/rbee-hive/src/http/workers.rs` lines 144-167

```rust
// TEAM-096: Determine port - find first available port
// Check existing workers to avoid address conflicts
let workers = state.registry.list().await;
let mut port = 8081u16;
let used_ports: std::collections::HashSet<u16> = workers
    .iter()
    .filter_map(|w| {
        // Extract port from URL like "http://127.0.0.1:8081"
        w.url.split(':').last().and_then(|p| p.parse().ok())
    })
    .collect();

// Find first unused port starting from 8081
while used_ports.contains(&port) {
    port += 1;
    if port > 9000 {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "No available ports (8081-9000 all in use)".to_string(),
        ));
    }
}

info!("🔍 Port allocation: {} workers registered, using port {}", workers.len(), port);
```

**How it works:**
1. Extract ports from all registered workers
2. Find first unused port starting from 8081
3. Fail gracefully if all ports 8081-9000 are used
4. Log allocation decision for debugging

### Fix 2: Fail-Fast Protocol (TEAM-096)

**File:** `bin/rbee-hive/src/registry.rs` lines 51-95

Added `failed_health_checks` counter to `WorkerInfo` and `increment_failed_health_checks()` method.

**File:** `bin/rbee-hive/src/monitor.rs` lines 54-92

```rust
Ok(response) => {
    // TEAM-096: Increment fail counter and remove after 3 failures
    let fail_count = registry.increment_failed_health_checks(&worker.id).await.unwrap_or(0);
    error!(
        worker_id = %worker.id,
        url = %worker.url,
        status = %response.status(),
        failed_checks = fail_count,
        "❌ Worker unhealthy"
    );
    
    if fail_count >= 3 {
        error!(
            worker_id = %worker.id,
            url = %worker.url,
            "🚨 FAIL-FAST: Removing worker after 3 failed health checks"
        );
        registry.remove(&worker.id).await;
    }
}
```

**How it works:**
1. Track failed health checks per worker
2. Reset counter on successful health check
3. Remove worker after 3 consecutive failures
4. Apply to both HTTP errors and connection failures

### Fix 3: Enhanced Narration (TEAM-096)

**File:** `bin/rbee-hive/src/monitor.rs` lines 27-33

```rust
let workers = registry.list().await;
if workers.is_empty() {
    info!("🔍 Health monitor: No workers to check");
    continue;
}

info!("🔍 Health monitor: Checking {} workers", workers.len());
```

**Added logging:**
- Port allocation decision with worker count
- Health monitor summary (N workers to check)
- Fail-fast removal notices
- Failed check counter in error logs

## Files Modified

1. `bin/rbee-hive/src/registry.rs` - Added fail counter field and methods
2. `bin/rbee-hive/src/http/workers.rs` - Smart port allocation
3. `bin/rbee-hive/src/monitor.rs` - Fail-fast protocol
4. `bin/rbee-hive/src/timeout.rs` - Updated test (field addition)
5. All registry tests - Added `failed_health_checks: 0` field

## Verification Plan

### Tests Run

- [x] Compilation: `cargo check -p rbee-hive` ✅ SUCCESS
- [x] Unit tests: `cargo test -p rbee-hive` ✅ 42/43 passed
  - Registry tests: ALL PASS ✅
  - Monitor tests: ALL PASS ✅
  - 1 failure in provisioner::catalog (pre-existing, unrelated)
- [ ] End-to-end: Spawn 3 workers, kill one, verify port reuse works
- [ ] Fail-fast: Let worker fail health checks 3 times, verify removal
- [ ] Port exhaustion: Verify graceful error when all ports used

### Expected Behavior

**Before fix:**
```
Worker 1 spawns on 8081 ✅
Worker 1 crashes ❌
Worker 2 spawns on 8082 ✅
Worker 3 tries 8081 (collision!) ❌
ERROR: Address already in use
ERROR: Worker unhealthy (forever...)
```

**After fix:**
```
Worker 1 spawns on 8081 ✅
Worker 1 crashes ❌
🚨 FAIL-FAST: Removing worker after 3 failed health checks
Worker 2 spawns on 8081 (reused!) ✅
Worker 3 spawns on 8082 ✅
```

## Next Steps

1. Run end-to-end test: `./ASK_SKY_BLUE.sh` multiple times
2. Verify port allocation logs show correct decisions
3. Verify failed workers get removed after 3 health checks (90s)
4. Test edge case: spawn 1000 workers (should fail gracefully at port 9000)

## References

- Engineering rules: `.windsurf/rules/engineering-rules.md`
- Debugging rules: `ENGINEERING_DEBUGGING_RULES.md`
- Related: TEAM-095 investigated question mark bug (different issue)

---

**TEAM-096 | 2025-10-18 | Status: 🟡 TESTING**
