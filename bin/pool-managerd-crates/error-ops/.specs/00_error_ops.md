# Error Ops SPEC — Operational Cleanup & Reporting (ERROPS-16xxx)

**Status**: Draft  
**Applies to**: `bin/pool-managerd-crates/error-ops/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `error-ops` crate handles operational cleanup when workers fail or crash. It is NOT policy-driven - it performs deterministic system cleanup and reports facts to orchestrator.

**Why it exists:**
- Pool manager must clean up system resources when workers fail
- Need to update VRAM accounting, remove registry entries, kill zombie processes
- Report failures to orchestratord (facts, not decisions)

**What it does:**
- Clean up crashed worker resources (VRAM accounting, registry, processes)
- Kill zombie processes
- Close file handles and network connections
- Report worker failures to orchestratord with exit codes/errors
- Track error patterns for observability

**What it does NOT do:**
- ❌ Decide to retry (orchestratord decides)
- ❌ Decide to fail over to another pool (orchestratord decides)
- ❌ Make policy decisions (this is operational cleanup only)

**Key distinction:**
- **error-ops** (pool manager): System cleanup + report facts
- **retry-policy** (orchestrator): Decide what to do about failures

---

## 1. Core Responsibilities

### [ERROPS-16001] Worker Crash Cleanup
The crate MUST clean up resources when workers crash or fail.

### [ERROPS-16002] VRAM Accounting Update
The crate MUST update GPU inventory when workers fail (release allocated VRAM).

### [ERROPS-16003] Registry Cleanup
The crate MUST remove failed workers from worker registry.

### [ERROPS-16004] Failure Reporting
The crate MUST report worker failures to orchestratord with details (exit code, error message).

---

## 2. Cleanup Operations

### [ERROPS-16010] Worker Crash Handler
When worker crashes (process exits unexpectedly):
```rust
pub fn handle_worker_crash(&mut self, worker_id: String, exit_code: i32) {
    // 1. Remove from registry
    let worker = self.registry.remove(&worker_id);
    
    // 2. Release VRAM
    self.gpu_inventory.release_worker_allocation(
        worker.gpu_id,
        worker.vram_bytes
    );
    
    // 3. Kill zombie processes (if any)
    self.kill_zombie_processes(&worker_id);
    
    // 4. Close file handles
    self.close_worker_resources(&worker_id);
    
    // 5. Report to orchestrator
    self.report_worker_failed(worker_id, exit_code, "Process crashed");
}
```

### [ERROPS-16011] Worker Startup Failure
When worker fails to start (timeout or error during load):
```rust
pub fn handle_startup_failure(&mut self, worker_id: String, error: String) {
    // 1. Kill process if still running
    self.kill_worker_process(&worker_id);
    
    // 2. Remove from registry
    self.registry.remove(&worker_id);
    
    // 3. Don't update VRAM (never allocated)
    
    // 4. Report to orchestrator
    self.report_worker_failed(worker_id, -1, error);
}
```

### [ERROPS-16012] Worker Hang Detection
When worker stops responding (health check fails):
```rust
pub fn handle_worker_hang(&mut self, worker_id: String) {
    // 1. Mark as unhealthy in registry
    self.registry.mark_unhealthy(&worker_id);
    
    // 2. Kill hanging process (SIGKILL)
    self.kill_worker_process(&worker_id);
    
    // 3. Clean up resources (same as crash)
    self.handle_worker_crash(worker_id, -1);
}
```

---

## 3. Process Management

### [ERROPS-16020] Zombie Process Reaping
The crate MUST reap zombie processes:
```rust
pub fn kill_zombie_processes(&self, worker_id: &str) {
    // Check for zombie child processes
    let zombies = find_zombie_children(worker.pid);
    for zombie_pid in zombies {
        // SIGKILL zombies
        kill(zombie_pid, SIGKILL);
        // Wait for cleanup
        waitpid(zombie_pid, None);
    }
}
```

### [ERROPS-16021] Graceful Kill Fallback
If worker doesn't respond to SIGTERM:
1. Send SIGTERM
2. Wait 5 seconds
3. If still alive, send SIGKILL
4. Wait for process exit

---

## 4. Resource Cleanup

### [ERROPS-16030] VRAM Accounting
When worker fails, update GPU inventory:
```rust
gpu_inventory.release_worker_allocation(gpu_id, vram_bytes);
```

This decrements `allocated_vram` and updates `available_vram`.

### [ERROPS-16031] Registry Cleanup
Remove worker from active registry:
```rust
registry.remove(&worker_id);
```

### [ERROPS-16032] File Handle Cleanup
Close any open file handles associated with worker:
- Model file handles
- Log file handles
- Socket connections

---

## 5. Failure Reporting

### [ERROPS-16040] Report to Orchestrator
Report worker failure with details:
```rust
POST http://orchestrator/v2/internal/workers/failed
{
  "worker_id": "worker-abc",
  "pool_id": "pool-1",
  "gpu_id": 0,
  "exit_code": -11,  // SIGSEGV
  "error_message": "Process crashed with segmentation fault",
  "vram_released_bytes": 16000000000,
  "uptime_seconds": 3600
}
```

### [ERROPS-16041] Failure Categories
The crate MUST categorize failures:
```rust
pub enum FailureCategory {
    Crash,           // Process crashed (SIGSEGV, SIGABRT)
    OOM,             // Out of memory
    Timeout,         // Startup timeout
    Hang,            // Stopped responding
    ExplicitStop,    // Graceful stop command
}
```

### [ERROPS-16042] Error Context
Include diagnostic context in report:
- Last log lines from worker
- Exit code or signal
- Uptime at failure
- VRAM usage at failure

---

## 6. Error Tracking

### [ERROPS-16050] Error Patterns
The crate SHOULD track error patterns for observability:
- Crash rate per GPU
- Most common failure types
- Mean time between failures (MTBF)

### [ERROPS-16051] Metrics
The crate MUST emit metrics:
- `worker_failures_total{category, gpu_id}`
- `worker_crashes_total{signal}`
- `zombie_processes_reaped_total`
- `cleanup_duration_ms`

---

## 7. Logging

### [ERROPS-16060] Structured Logging
The crate MUST log cleanup events:
```rust
tracing::error!(
    worker_id = %worker_id,
    gpu_id = gpu_id,
    exit_code = exit_code,
    vram_released = vram_bytes,
    "Worker crashed, cleanup complete"
);
```

### [ERROPS-16061] Log Levels
- **ERROR**: Worker crash, hang detection
- **WARN**: Zombie process detected
- **INFO**: Cleanup complete, resources released

---

## 8. Dependencies

### [ERROPS-16070] Required Crates
```toml
[dependencies]
gpu-inventory = { path = "../gpu-inventory" }
tokio = { workspace = true }
tracing = { workspace = true }
thiserror = { workspace = true }
nix = { workspace = true }  # For process signals
```

---

## 9. Testing

### [ERROPS-16080] Unit Tests
The crate MUST include tests for:
- VRAM accounting updates on failure
- Registry cleanup
- Failure reporting format

### [ERROPS-16081] Integration Tests
The crate SHOULD test:
- Zombie process reaping
- Graceful kill fallback (SIGTERM → SIGKILL)

---

## 10. Traceability

**Code**: `bin/pool-managerd-crates/error-ops/src/`  
**Tests**: `bin/pool-managerd-crates/error-ops/tests/`  
**Parent**: `bin/pool-managerd/.specs/00_pool-managerd.md`  
**Used by**: `pool-managerd`, `worker-lifecycle`  
**Depends on**: `gpu-inventory`  
**Spec IDs**: ERROPS-16001 to ERROPS-16081

---

**End of Specification**
