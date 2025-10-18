# Component: Worker Lifecycle Management

**Location:** `bin/rbee-hive/src/` (multiple files)  
**Type:** Process management system  
**Language:** Rust  
**Created by:** TEAM-027, 096  
**Status:** 🟡 PARTIAL (critical gaps)

## Overview

Manages complete lifecycle of worker processes from spawn to termination. Currently functional but missing critical PID tracking and force-kill capabilities.

## Lifecycle States

```
┌─────────────────────────────────────────────────────────┐
│ Worker Lifecycle States                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  SPAWN → LOADING → IDLE → BUSY → IDLE → TERMINATED     │
│    │        │       │      │      │         │          │
│    │        │       │      │      │         └─ Shutdown│
│    │        │       │      │      └─ Request complete  │
│    │        │       │      └─ Processing request       │
│    │        │       └─ Ready for requests              │
│    │        └─ Model loading                           │
│    └─ Process started                                  │
│                                                         │
│  SPAWN → LOADING → FAILED (if startup fails)           │
│  ANY STATE → FAILED (if health checks fail 3x)         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Spawn (`http/workers.rs` - TEAM-027, 096)

**Current Implementation:**
```rust
// ✅ Port allocation (TEAM-096: Smart allocation)
let port = find_available_port(&registry).await?;

// ✅ Spawn process
let child = Command::new("llm-worker-rbee")
    .arg("--worker-id").arg(&worker_id)
    .arg("--model").arg(&model_path)
    .arg("--port").arg(port.to_string())
    .stdout(Stdio::inherit())  // TEAM-088
    .stderr(Stdio::inherit())
    .spawn()?;

// ✅ Check immediate failure
tokio::time::sleep(Duration::from_millis(100)).await;
if let Ok(Some(status)) = child.try_wait() {
    return Err("Worker failed to start");
}

// ✅ Register in registry
registry.register(WorkerInfo {
    id: worker_id,
    state: WorkerState::Loading,
    failed_health_checks: 0,  // TEAM-096
    ...
}).await;

// ❌ CRITICAL GAP: child.id() not stored!
// child goes out of scope - PID lost!
```

**Gaps:**
- ❌ No PID storage
- ❌ No ready timeout (worker stuck in Loading forever)
- ❌ No resource limits

### 2. Health Monitoring (`monitor.rs` - TEAM-027, 096)

**Current Implementation:**
```rust
// ✅ 30s interval health checks
let mut interval = tokio::time::interval(Duration::from_secs(30));

loop {
    interval.tick().await;
    
    for worker in registry.list().await {
        match http_health_check(&worker.url).await {
            Ok(_) => {
                // ✅ Reset fail counter
                registry.update_state(&worker.id, worker.state).await;
            }
            Err(_) => {
                // ✅ TEAM-096: Fail-fast protocol
                let count = registry.increment_failed_health_checks(&worker.id).await;
                if count >= 3 {
                    registry.remove(&worker.id).await;
                }
            }
        }
    }
}
```

**Gaps:**
- ❌ 30-90s crash detection delay
- ❌ No heartbeat mechanism
- ❌ No process liveness check (only HTTP)

### 3. Idle Timeout (`timeout.rs` - TEAM-027)

**Current Implementation:**
```rust
// ✅ 60s interval, 5min timeout
for worker in registry.get_idle_workers().await {
    if idle_duration(&worker) > Duration::from_secs(300) {
        // ✅ Graceful shutdown request
        http_shutdown(&worker.url).await?;
        
        // ✅ Remove from registry
        registry.remove(&worker.id).await;
    }
}
```

**Gaps:**
- ❌ Can't force kill if HTTP shutdown fails
- ❌ No verification that process actually terminated

### 4. Graceful Shutdown (`commands/daemon.rs` - TEAM-030)

**Current Implementation:**
```rust
// ✅ SIGINT/SIGTERM handler
tokio::signal::ctrl_c().await?;

// ✅ Shutdown all workers
for worker in registry.list().await {
    let _ = http_shutdown(&worker.url).await;
}

// ✅ Clear registry
registry.clear().await;
```

**Gaps:**
- ❌ Sequential shutdown (slow for many workers)
- ❌ No timeout enforcement
- ❌ Can't force kill hung workers

## Critical Gaps Summary

### P0 - Blocks Production
1. **No PID Tracking**
   - Can't send SIGTERM/SIGKILL
   - Can't check process liveness
   - Can't force kill hung workers

2. **No Force Kill**
   - Hung workers block shutdown
   - Failed HTTP shutdown leaves orphans

3. **No Ready Timeout**
   - Workers stuck in Loading forever
   - No automatic cleanup

### P1 - Needed for Reliability
4. **No Restart Policy**
   - Crashed workers stay dead
   - Manual respawn required

5. **No Heartbeat**
   - 30-90s crash detection delay
   - Only HTTP health checks

6. **No Resource Limits**
   - Workers can OOM system
   - No CPU/memory limits

## Recommended Implementation

### Add PID Tracking
```rust
pub struct WorkerInfo {
    // ... existing fields ...
    pub pid: Option<u32>,  // TEAM-096: Store process ID
}

// In spawn:
let child = Command::new("llm-worker-rbee").spawn()?;
let pid = child.id();

registry.register(WorkerInfo {
    pid: Some(pid),
    ...
}).await;
```

### Add Force Kill
```rust
pub async fn force_kill_worker(&self, worker_id: &str) -> Result<()> {
    let worker = self.registry.get(worker_id).await
        .ok_or("Worker not found")?;
    
    if let Some(pid) = worker.pid {
        // Try SIGTERM first
        nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGTERM,
        )?;
        
        // Wait 10s
        tokio::time::sleep(Duration::from_secs(10)).await;
        
        // Check if still alive
        if process_exists(pid) {
            // Force SIGKILL
            nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                nix::sys::signal::Signal::SIGKILL,
            )?;
        }
    }
    
    self.registry.remove(worker_id).await;
    Ok(())
}
```

### Add Ready Timeout
```rust
// In spawn:
let worker_id = worker_id.clone();
let registry = registry.clone();

tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(30)).await;
    
    if let Some(worker) = registry.get(&worker_id).await {
        if worker.state == WorkerState::Loading {
            // Still loading after 30s - kill it
            force_kill_worker(&worker_id).await;
        }
    }
});
```

## Maturity Assessment

**Status:** 🟡 **FUNCTIONAL BUT INCOMPLETE**

**Implemented:**
- ✅ Spawn with smart port allocation (TEAM-096)
- ✅ Health monitoring (30s interval)
- ✅ Fail-fast removal (3 failures, TEAM-096)
- ✅ Idle timeout (5min, TEAM-027)
- ✅ Graceful shutdown (TEAM-030)
- ✅ Stdout/stderr inheritance (TEAM-088)

**Critical Gaps:**
- ❌ PID tracking
- ❌ Force kill
- ❌ Ready timeout
- ❌ Restart policy
- ❌ Heartbeat mechanism
- ❌ Resource limits

## Related Components

- **Worker Registry** - Stores worker state
- **Monitor** - Health checks
- **Timeout** - Idle enforcement
- **HTTP API** - Worker management

---

**Created by:** TEAM-027, 096  
**Last Updated:** 2025-10-18  
**Maturity:** 🟡 Functional but incomplete (see LIFECYCLE_MANAGEMENT_GAPS.md)
