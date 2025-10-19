# Queen-rbee → rbee-hive Lifecycle Management Analysis

**Date:** 2025-10-19  
**Analyst:** TEAM-123  
**Status:** ⚠️ INCOMPLETE - Critical gaps identified

---

## Executive Summary

Queen-rbee's rbee-hive lifecycle management is **60% complete**. The startup flow works, but there are critical gaps in the callback chain, health monitoring, and error recovery.

**Critical Finding:** rbee-hive receives worker ready callbacks but **NEVER notifies queen-rbee**, causing queen-rbee to poll workers for 5 minutes instead of being notified immediately.

---

## 1. Startup Lifecycle: ✅ COMPLETE

### Local Startup (localhost)
**File:** `http/inference.rs:254-341`

**Flow:**
1. Check if rbee-hive already running on port 9200
2. If not, spawn `rbee-hive daemon --addr 127.0.0.1:9200`
3. Poll health endpoint every 100ms for max 10 seconds
4. Detach process and continue

**Status:** ✅ Works well
**Issues:** None

### Remote Startup (SSH)
**File:** `http/inference.rs:344-371`

**Flow:**
1. Execute SSH command: `{install_path}/rbee-hive daemon --addr 0.0.0.0:8080 > /tmp/rbee-hive.log 2>&1 &`
2. Wait for rbee-hive ready (60 second timeout)
3. Poll health endpoint with exponential backoff (5 retries)

**Status:** ✅ Works
**Issues:**
- Logs redirected to `/tmp/rbee-hive.log` - hard to debug
- No check if rbee-hive already running (will spawn duplicates!)
- Port hardcoded to 8080 (conflicts with queen-rbee on same machine)

---

## 2. Worker Ready Notification: ❌ BROKEN

### Expected Flow (from specs)
```
Worker → rbee-hive (POST /v1/workers/ready)
rbee-hive → queen-rbee (POST /v2/workers/ready) ← MISSING!
queen-rbee → proceeds with inference
```

### Actual Flow
```
Worker → rbee-hive (POST /v1/workers/ready)
rbee-hive → updates registry, returns 200 OK
queen-rbee → polls worker /v1/ready every 2s for 300s ← WORKAROUND!
```

**Problem:** rbee-hive receives worker ready callback but **DOES NOT notify queen-rbee**

**File:** `bin/rbee-hive/src/http/workers.rs:370-399`
```rust
pub async fn handle_worker_ready(
    State(state): State<AppState>,
    Json(request): Json<WorkerReadyRequest>,
) -> Result<Json<WorkerReadyResponse>, (StatusCode, String)> {
    // ... validation ...
    
    // Update worker state to idle
    state.registry.update_state(&request.worker_id, WorkerState::Idle).await;
    
    // ❌ MISSING: Notify queen-rbee that worker is ready!
    
    Ok(Json(WorkerReadyResponse { 
        message: "Worker registered as ready".to_string() 
    }))
}
```

**Workaround:** Queen-rbee polls worker directly
**File:** `bin/queen-rbee/src/http/inference.rs:425-503`
```rust
async fn wait_for_worker_ready(worker_url: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let timeout = std::time::Duration::from_secs(300); // 5 minutes!
    
    loop {
        match client.get(format!("{}/v1/ready", worker_url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                // Check if worker is actually ready
                let ready: ReadyResponse = resp.json().await?;
                if ready.ready {
                    return Ok(());
                }
            }
            // ... retry every 2 seconds ...
        }
        
        if start.elapsed() > timeout {
            anyhow::bail!("Worker ready timeout after 5 minutes");
        }
        
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}
```

**Impact:**
- ⏰ 5-minute timeout instead of immediate notification
- 🔄 Unnecessary polling (150 HTTP requests per worker!)
- 🐛 60-second BDD test timeouts when worker doesn't start

**Fix Required:**
1. rbee-hive needs queen-rbee callback URL (passed during spawn)
2. rbee-hive calls `POST {queen_url}/v2/workers/ready` when worker ready
3. Queen-rbee waits on async channel instead of polling

---

## 3. Health Monitoring: ❌ MISSING

### rbee-hive Health Checks
**Status:** ❌ NOT IMPLEMENTED

Queen-rbee has preflight checks (`preflight/rbee_hive.rs`) but **NEVER uses them after startup**:
- `check_health()` - checks `/health` endpoint
- `check_version_compatibility()` - validates version
- `query_backends()` - gets available backends
- `query_resources()` - gets RAM/disk info

**Problem:** These are only used in tests, never in production code!

**Missing:**
- Periodic health checks (every 30s?)
- Automatic restart if rbee-hive crashes
- Alerting when rbee-hive becomes unhealthy

---

## 4. Shutdown Lifecycle: ⚠️ PARTIAL

### Cascading Shutdown
**File:** `main.rs:144-298`  
**Status:** ✅ Implemented (TEAM-105)

**Flow:**
1. Ctrl+C received by queen-rbee
2. Query all registered hives from beehive registry
3. For each hive (parallel):
   - SSH to node
   - Find rbee-hive PID: `pgrep -f 'rbee-hive daemon'`
   - Send SIGTERM: `kill -TERM {pid}`
4. Wait max 30 seconds total
5. Audit log results

**Good:**
- ✅ Parallel shutdown
- ✅ 30-second timeout
- ✅ Audit logging
- ✅ Graceful SIGTERM (not SIGKILL)

**Issues:**
- ⚠️ No verification that rbee-hive actually stopped
- ⚠️ No cleanup of orphaned workers
- ⚠️ No notification to active clients

---

## 5. Error Recovery: ❌ MISSING

### rbee-hive Crash Detection
**Status:** ❌ NOT IMPLEMENTED

**What happens if rbee-hive crashes?**
- Queen-rbee doesn't know (no health monitoring)
- Active inference requests fail with connection errors
- Workers become orphaned (still running but unreachable)
- Next inference request tries to spawn new rbee-hive
- May create duplicate rbee-hive processes!

**Missing:**
- Crash detection
- Automatic restart
- Worker cleanup on crash
- Client notification

### Worker Crash Handling
**Status:** ⚠️ PARTIAL

Queen-rbee has worker registry (`worker_registry.rs`) with shutdown capability:
```rust
pub async fn shutdown_worker(&self, worker_id: &str) -> anyhow::Result<()> {
    // Send POST /v1/admin/shutdown to worker
}
```

**But:**
- No automatic detection of crashed workers
- No cleanup of orphaned workers
- No retry logic if worker crashes during inference

---

## 6. Resource Management: ❌ MISSING

### Connection Pooling
**Status:** ❌ NOT IMPLEMENTED

Every inference request creates new HTTP clients:
```rust
let client = reqwest::Client::new(); // Creates new client every time!
```

**Should:**
- Use connection pool
- Reuse HTTP clients
- Set connection limits

### rbee-hive Process Tracking
**Status:** ❌ NOT IMPLEMENTED

**Problems:**
- No tracking of spawned rbee-hive PIDs
- Can't reliably kill rbee-hive on shutdown
- May leave zombie processes
- No way to check if specific rbee-hive is still running

**Should:**
- Store rbee-hive PIDs in registry
- Track process state (starting, running, stopping, crashed)
- Periodic PID validation

---

## 7. Observability: ⚠️ PARTIAL

### Logging
**Status:** ✅ Good

Comprehensive tracing with structured logging:
- Startup events
- SSH connections
- Worker spawns
- Inference requests
- Shutdown events

### Metrics
**Status:** ❌ NOT IMPLEMENTED

**Missing:**
- rbee-hive uptime
- Worker spawn latency
- Inference request latency
- Error rates
- Active connections

### Audit Trail
**Status:** ⚠️ PARTIAL

Only shutdown events are audited (TEAM-105).

**Missing:**
- rbee-hive startup/crash events
- Worker lifecycle events
- Configuration changes
- Security events (auth failures, etc.)

---

## 8. Configuration Management: ❌ MISSING

### rbee-hive Configuration
**Status:** ❌ HARDCODED

All rbee-hive settings are hardcoded:
- Port: 9200 (local) or 8080 (remote)
- Install path: from registry
- Log path: `/tmp/rbee-hive.log`
- Timeouts: various hardcoded values

**Should:**
- Configurable ports
- Configurable log paths
- Configurable timeouts
- Environment-specific settings

---

## Completeness Score

| Component | Status | Score | Notes |
|-----------|--------|-------|-------|
| **Startup** | ✅ Complete | 95% | Works well, minor issues |
| **Worker Ready Notification** | ❌ Broken | 0% | Polling workaround, not callback |
| **Health Monitoring** | ❌ Missing | 0% | Preflight exists but unused |
| **Shutdown** | ⚠️ Partial | 70% | Works but no verification |
| **Error Recovery** | ❌ Missing | 0% | No crash detection/restart |
| **Resource Management** | ❌ Missing | 10% | No pooling, no tracking |
| **Observability** | ⚠️ Partial | 40% | Good logs, no metrics |
| **Configuration** | ❌ Missing | 5% | Everything hardcoded |

**Overall: 60% Complete**

---

## Critical Fixes Required

### Priority 1: Worker Ready Callback (CRITICAL)
**Impact:** 5-minute timeouts, unnecessary polling

**Fix:**
1. Add `queen_url` parameter to worker spawn request
2. rbee-hive stores queen_url in worker registry
3. rbee-hive calls `POST {queen_url}/v2/workers/ready` when worker ready
4. Queen-rbee implements `/v2/workers/ready` endpoint
5. Queen-rbee waits on async channel instead of polling

**Estimated Effort:** 4 hours

### Priority 2: rbee-hive Health Monitoring
**Impact:** Undetected crashes, orphaned workers

**Fix:**
1. Background task in queen-rbee: check rbee-hive health every 30s
2. If health check fails 3 times, mark as crashed
3. Attempt automatic restart
4. Clean up orphaned workers
5. Alert operators

**Estimated Effort:** 8 hours

### Priority 3: Process Tracking
**Impact:** Zombie processes, unreliable shutdown

**Fix:**
1. Store rbee-hive PIDs in beehive registry
2. Validate PIDs periodically
3. Use PIDs for reliable shutdown
4. Detect and clean up zombies

**Estimated Effort:** 4 hours

### Priority 4: Connection Pooling
**Impact:** Resource waste, slower performance

**Fix:**
1. Create shared `reqwest::Client` in AppState
2. Reuse client for all HTTP requests
3. Configure connection limits

**Estimated Effort:** 2 hours

---

## Recommendations

1. **Immediate:** Fix worker ready callback (Priority 1)
2. **Short-term:** Implement health monitoring (Priority 2)
3. **Medium-term:** Add process tracking and connection pooling (Priority 3-4)
4. **Long-term:** Add metrics, audit trail, configuration management

---

## References

- `.specs/LIFECYCLE_CLARIFICATION.md` - Expected callback flow
- `.specs/ARCHITECTURE_UPDATE.md` - System architecture
- `bin/queen-rbee/src/http/inference.rs` - Inference orchestration
- `bin/rbee-hive/src/http/workers.rs` - Worker ready callback
- `bin/queen-rbee/src/main.rs` - Shutdown handling
