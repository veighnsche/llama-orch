# rbee-hive Worker Lifecycle Management Gaps

**TEAM-096 Analysis | 2025-10-18**

**Status:** 🔴 UNDERDEVELOPED - Major lifecycle gaps identified

## User Observation

> "The bee hive lifecycle management of the worker bee is still very underdeveloped is it not?"

**Answer: YES. Critically underdeveloped.**

## Current Lifecycle State

### ✅ What EXISTS (Historical Team Work)

Based on historical team comments in the code:

1. **Spawn** (TEAM-027, TEAM-029, TEAM-035, TEAM-087, TEAM-088)
   - Binary path resolution
   - Port allocation (TEAM-096: Now fixed)
   - Model provisioning integration (TEAM-029)
   - Enhanced spawn diagnostics (TEAM-087)
   - Stdout/stderr inheritance (TEAM-088)
   - 100ms startup check

2. **Health Monitoring** (TEAM-027, TEAM-096)
   - 30s interval health checks
   - Fail-fast protocol (TEAM-096: After 3 failures)

3. **Idle Timeout** (TEAM-027)
   - 5 minute idle threshold
   - Graceful shutdown via POST /v1/admin/shutdown

4. **Graceful Shutdown** (TEAM-030)
   - Server: SIGINT/SIGTERM handling
   - Cascading shutdown to workers
   - Registry cleanup

### ❌ What's MISSING (Critical Gaps)

#### 1. **No Process Management**

**File:** `src/http/workers.rs` line 230

```rust
match spawn_result {
    Ok(mut child) => {
        // TEAM-087: Check if process started successfully
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        // ❌ PROBLEM: child.id() is NEVER stored!
        // ❌ PROBLEM: Process handle is DROPPED immediately!
        // ❌ PROBLEM: No way to kill/signal worker later!
        
        if let Ok(Some(status)) = child.try_wait() {
            // Handle immediate exit
        }
        
        // Register worker in loading state
        let worker = WorkerInfo { ... };
        state.registry.register(worker).await;
        
        // ❌ PROBLEM: `child` goes out of scope here - process becomes orphaned!
```

**Impact:**
- Can't send SIGTERM to workers
- Can't check if process is still running
- Rely 100% on HTTP health checks (30s delay)
- Orphaned processes on rbee-hive crash

#### 2. **No Crash Detection Beyond Health Checks**

**File:** `src/monitor.rs` line 22

```rust
pub async fn health_monitor_loop(registry: Arc<WorkerRegistry>) {
    let mut interval = tokio::time::interval(Duration::from_secs(30));
    // ❌ PROBLEM: 30 second interval = 30-90s to detect crashes!
```

**Impact:**
- Worker crashes at 0s
- First health check fails at 30s
- Second fails at 60s
- Third fails at 90s (finally removed)
- **90 seconds of "zombie" worker in registry!**

#### 3. **No Resource Limits**

**Missing:**
- Memory limits (GGUF models can be huge)
- CPU limits (inference can peg cores)
- Disk space checks (model downloads)
- VRAM limits (GPU workers)

**Current state:** Workers can OOM the entire system.

#### 4. **No Restart Policy**

**Missing:**
- Automatic restart on crash
- Exponential backoff
- Max restart attempts
- Crash loop detection

**Current state:** Worker dies = permanent loss until manual respawn.

#### 5. **No State Transition Validation**

**File:** `src/registry.rs` line 75

```rust
pub async fn update_state(&self, worker_id: &str, state: WorkerState) {
    let mut workers = self.workers.write().await;
    if let Some(worker) = workers.get_mut(worker_id) {
        worker.state = state;  // ❌ No validation!
        // Can go from Loading → Busy (skipping Idle)
        // Can go from any state → any state
    }
}
```

**Missing:**
- State machine with valid transitions
- Loading → Idle (only via ready callback)
- Idle ↔ Busy (only)
- Any → Failed (terminal)

#### 6. **No Graceful Shutdown Coordination**

**File:** `src/commands/daemon.rs` line 104

```rust
async fn shutdown_all_workers(registry: Arc<WorkerRegistry>) {
    let workers = registry.list().await;
    
    for worker in workers {
        // ❌ PROBLEM: Sequential shutdown (slow for many workers)
        // ❌ PROBLEM: No timeout enforcement
        // ❌ PROBLEM: If worker hangs, blocks entire shutdown
        if let Err(e) = shutdown_worker(&worker.url).await {
            tracing::warn!("Failed to shutdown worker");
        }
    }
}
```

**Should have:**
- Parallel shutdown with timeout
- Force kill after grace period
- Progress reporting

#### 7. **No Worker Ready Confirmation**

**File:** `src/http/workers.rs` line 272

```rust
// Register worker in loading state
let worker = WorkerInfo {
    state: WorkerState::Loading,  // ❌ Assumes it will become ready!
    // ...
};

state.registry.register(worker).await;
// ❌ No verification that worker actually calls /ready callback
```

**Missing:**
- Timeout for ready callback
- Automatic removal if not ready within N seconds
- Model load progress tracking

#### 8. **No Heartbeat Mechanism**

**Current:** Only HTTP health checks every 30s.

**Missing:**
- Worker-initiated heartbeats (faster detection)
- Bidirectional keepalive
- Network partition detection

#### 9. **No Worker Metrics Collection**

**Missing:**
- Inference latency tracking
- Token throughput
- Memory usage trends
- GPU utilization
- Queue depth

**Impact:** No data for autoscaling/load balancing decisions.

#### 10. **No Port Cleanup on Failure**

**Problem:** If worker fails AFTER binding port but BEFORE calling /ready:
- Port is bound
- Worker crashes
- Port stays bound for OS timeout (typically 60s)
- Can't reuse port immediately

**Solution needed:**
- SO_REUSEADDR on worker sockets
- Port release tracking
- Force unbind on cleanup

## Comparison: What SHOULD Exist

### Production-Grade Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│ SPAWN                                                    │
│  1. Validate resources available (ports, memory, disk) │
│  2. Spawn process with PID tracking                    │
│  3. Set resource limits (cgroups/rlimits)              │
│  4. Monitor stdout/stderr in background task           │
│  5. Set ready timeout (30s)                            │
├─────────────────────────────────────────────────────────┤
│ LOADING → READY                                         │
│  1. Worker loads model (progress reported)             │
│  2. Worker calls /ready callback (before timeout)      │
│  3. Registry validates transition Loading → Idle       │
│  4. Start heartbeat monitoring (5s interval)           │
├─────────────────────────────────────────────────────────┤
│ RUNNING                                                 │
│  1. Heartbeat every 5s (faster than 30s health check)  │
│  2. Metrics collected every 10s                        │
│  3. State transitions validated                        │
│  4. Process liveness checked                           │
├─────────────────────────────────────────────────────────┤
│ FAILURE DETECTION                                       │
│  1. Heartbeat timeout (15s = 3 missed beats)           │
│  2. Process exit detected immediately (wait4)          │
│  3. Health check fails (30s backup)                    │
│  4. OOM kill detected (cgroup notifications)           │
├─────────────────────────────────────────────────────────┤
│ CLEANUP                                                 │
│  1. Mark worker as Failed                              │
│  2. Send SIGTERM (graceful)                            │
│  3. Wait 10s                                           │
│  4. Send SIGKILL (force)                               │
│  5. Release port                                       │
│  6. Clean up resources                                 │
│  7. Remove from registry                               │
│  8. Decide restart policy                              │
├─────────────────────────────────────────────────────────┤
│ GRACEFUL SHUTDOWN                                       │
│  1. Stop accepting new requests                        │
│  2. Drain in-flight requests                           │
│  3. Send SIGTERM to all workers (parallel)             │
│  4. Wait 30s for graceful exit                         │
│  5. SIGKILL any remaining                              │
│  6. Release all resources                              │
└─────────────────────────────────────────────────────────┘
```

### Current Implementation

```
┌─────────────────────────────────────────────────────────┐
│ SPAWN                                                    │
│  1. ✅ Spawn process                                    │
│  2. ❌ No PID stored                                    │
│  3. ❌ Process handle dropped                           │
│  4. ✅ Stdout/stderr inherited (TEAM-088)               │
│  5. ❌ No ready timeout                                 │
├─────────────────────────────────────────────────────────┤
│ LOADING → READY                                         │
│  1. ❌ No progress reporting                            │
│  2. ✅ /ready callback exists                           │
│  3. ❌ No transition validation                         │
│  4. ❌ No heartbeat monitoring                          │
├─────────────────────────────────────────────────────────┤
│ RUNNING                                                 │
│  1. ❌ No heartbeat (only 30s health checks)            │
│  2. ❌ No metrics collection                            │
│  3. ❌ No state validation                              │
│  4. ❌ Can't check process liveness                     │
├─────────────────────────────────────────────────────────┤
│ FAILURE DETECTION                                       │
│  1. ❌ No heartbeat                                     │
│  2. ❌ Can't detect process exit                        │
│  3. ✅ Health check (but 30-90s delay)                  │
│  4. ❌ No OOM detection                                 │
├─────────────────────────────────────────────────────────┤
│ CLEANUP                                                 │
│  1. ✅ Fail-fast removal (TEAM-096)                     │
│  2. ❌ Can't send signals (no PID)                      │
│  3. ❌ Can't force kill                                 │
│  4. ❌ Port might stay bound                            │
│  5. ❌ No resource cleanup                              │
│  6. ✅ Registry removal                                 │
│  7. ❌ No restart policy                                │
├─────────────────────────────────────────────────────────┤
│ GRACEFUL SHUTDOWN                                       │
│  1. ✅ SIGINT/SIGTERM handling (TEAM-030)               │
│  2. ❌ No request draining                              │
│  3. ✅ HTTP shutdown to workers                         │
│  4. ❌ Sequential (slow), no timeout                    │
│  5. ❌ Can't force kill                                 │
│  6. ✅ Registry cleanup                                 │
└─────────────────────────────────────────────────────────┘
```

## Priority Fixes Needed

### P0 - Critical (Blocks Production)

1. **Store PIDs** - Can't manage workers without PIDs
2. **Ready timeout** - Workers stuck in Loading forever
3. **Force kill capability** - Hung workers block shutdown
4. **State machine** - Invalid transitions cause bugs

### P1 - High (Needed for Reliability)

5. **Heartbeat mechanism** - 30-90s crash detection too slow
6. **Resource limits** - Workers can OOM system
7. **Parallel shutdown** - Sequential shutdown too slow
8. **Restart policy** - Manual respawn not sustainable

### P2 - Medium (Needed for Operations)

9. **Metrics collection** - Can't debug/optimize without data
10. **Progress tracking** - Model load status unknown
11. **Port cleanup** - Port conflicts on rapid restart
12. **Crash loop detection** - Prevent infinite restart loops

## Files That Need Work

1. `src/http/workers.rs` - Store PIDs, ready timeout
2. `src/registry.rs` - Add PID field, state machine
3. `src/monitor.rs` - Add heartbeat, faster detection
4. `src/commands/daemon.rs` - Parallel shutdown, force kill
5. **NEW:** `src/lifecycle.rs` - Centralized lifecycle logic
6. **NEW:** `src/heartbeat.rs` - Worker heartbeat system

## Historical Team Context (To Preserve)

- TEAM-027: Initial spawn, monitoring, timeout (foundational)
- TEAM-029: Model provisioning integration
- TEAM-030: Graceful shutdown, registry cleanup
- TEAM-035: Worker CLI args
- TEAM-087: Enhanced spawn diagnostics
- TEAM-088: Stdout/stderr inheritance
- TEAM-092: Model metadata in callback
- TEAM-096: Port allocation fix, fail-fast protocol

**All this work was good foundation. But lifecycle management needs comprehensive overhaul.**

## Recommendation

**Create new epic:** "Worker Lifecycle Management v2"

**Scope:**
- Phase 1: PID tracking + ready timeout + force kill (P0)
- Phase 2: Heartbeat + state machine + restart policy (P1)
- Phase 3: Metrics + resource limits + parallel shutdown (P1-P2)

**Estimated:** 3-4 teams (each team = 1 phase)

**Why separate from TEAM-096:**
- TEAM-096 fixed immediate bug (port conflicts)
- Lifecycle overhaul is much larger scope
- Needs architectural design, not just bug fix

---

**TEAM-096 | 2025-10-18 | Respecting historical work, but acknowledging gaps**
