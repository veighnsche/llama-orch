# TEAM-272: Worker Lifecycle - Management

**Phase:** 6 of 9  
**Estimated Effort:** 24-32 hours  
**Prerequisites:** TEAM-271 complete  
**Blocks:** TEAM-273 (Job Router Integration)

---

## 🎯 Mission

Implement worker management operations: WorkerList, WorkerGet, WorkerDelete. Complete the worker lifecycle CRUD operations.

**Deliverables:**
1. ✅ WorkerList operation
2. ✅ WorkerGet operation
3. ✅ WorkerDelete operation with process cleanup
4. ✅ Narration events
5. ✅ All operations wired up in job_router.rs

---

## 📁 Files to Modify

```
bin/25_rbee_hive_crates/worker-lifecycle/
├── src/
│   ├── lib.rs          ← Export management module (optional)
│   └── management.rs   ← Process cleanup logic (optional)
└── Cargo.toml          ← Add dependencies if needed

bin/20_rbee_hive/
└── src/
    └── job_router.rs   ← Wire up 3 operations
```

---

## 🏗️ Implementation Guide

### Step 1: Add Dependencies (if needed)

For process killing on Unix:
```toml
[dependencies]
# Existing dependencies...

# For process management (Unix only)
[target.'cfg(unix)'.dependencies]
nix = { version = "0.27", features = ["signal"] }
```

### Step 2: Wire Up WorkerList in job_router.rs

```rust
Operation::WorkerList { hive_id } => {
    // TEAM-272: Implemented worker list
    NARRATE
        .action("worker_list_start")
        .job_id(&job_id)
        .context(&hive_id)
        .human("📋 Listing workers on hive '{}'")
        .emit();

    let workers = state.worker_registry.list();

    NARRATE
        .action("worker_list_result")
        .job_id(&job_id)
        .context(&workers.len().to_string())
        .human("Found {} worker(s)")
        .emit();

    if workers.is_empty() {
        NARRATE
            .action("worker_list_empty")
            .job_id(&job_id)
            .human("No workers found")
            .emit();
    } else {
        for worker in &workers {
            let status = match &worker.status {
                rbee_hive_worker_lifecycle::WorkerStatus::Starting => "starting",
                rbee_hive_worker_lifecycle::WorkerStatus::Ready => "ready",
                rbee_hive_worker_lifecycle::WorkerStatus::Busy => "busy",
                rbee_hive_worker_lifecycle::WorkerStatus::Stopped => "stopped",
                rbee_hive_worker_lifecycle::WorkerStatus::Failed { .. } => "failed",
            };

            NARRATE
                .action("worker_list_entry")
                .job_id(&job_id)
                .context(&worker.id)
                .context(&worker.model_id)
                .context(&worker.device)
                .context(&worker.port.to_string())
                .context(status)
                .human("  {} | {} | {} | port {} | {}")
                .emit();
        }
    }
}
```

### Step 3: Wire Up WorkerGet in job_router.rs

```rust
Operation::WorkerGet { hive_id, id } => {
    // TEAM-272: Implemented worker get
    NARRATE
        .action("worker_get_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&id)
        .human("🔍 Getting worker '{}' on hive '{}'")
        .emit();

    match state.worker_registry.get(&id) {
        Ok(worker) => {
            NARRATE
                .action("worker_get_found")
                .job_id(&job_id)
                .context(&worker.id)
                .context(&worker.model_id)
                .context(&worker.device)
                .context(&worker.port.to_string())
                .human("✅ Worker: {} | Model: {} | Device: {} | Port: {}")
                .emit();

            // Emit worker details as JSON
            let json = serde_json::to_string_pretty(&worker)
                .unwrap_or_else(|_| "Failed to serialize".to_string());

            NARRATE
                .action("worker_get_details")
                .job_id(&job_id)
                .human(&json)
                .emit();
        }
        Err(e) => {
            NARRATE
                .action("worker_get_error")
                .job_id(&job_id)
                .context(&id)
                .context(&e.to_string())
                .human("❌ Worker '{}' not found: {}")
                .emit();
            return Err(e);
        }
    }
}
```

### Step 4: Wire Up WorkerDelete in job_router.rs

```rust
Operation::WorkerDelete { hive_id, id } => {
    // TEAM-272: Implemented worker delete
    NARRATE
        .action("worker_delete_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&id)
        .human("🗑️  Deleting worker '{}' on hive '{}'")
        .emit();

    match state.worker_registry.get(&id) {
        Ok(worker) => {
            // Kill process
            NARRATE
                .action("worker_delete_kill")
                .job_id(&job_id)
                .context(&worker.pid.to_string())
                .human("Killing process PID {}")
                .emit();

            // Attempt to kill the process
            #[cfg(unix)]
            {
                use nix::sys::signal::{kill, Signal};
                use nix::unistd::Pid;

                // Try SIGTERM first (graceful)
                match kill(Pid::from_raw(worker.pid as i32), Signal::SIGTERM) {
                    Ok(_) => {
                        NARRATE
                            .action("worker_delete_sigterm")
                            .job_id(&job_id)
                            .human("Sent SIGTERM to worker process")
                            .emit();

                        // Wait a bit for graceful shutdown
                        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

                        // Check if process still exists, if so, use SIGKILL
                        if kill(Pid::from_raw(worker.pid as i32), Signal::SIGKILL).is_ok() {
                            NARRATE
                                .action("worker_delete_sigkill")
                                .job_id(&job_id)
                                .human("Sent SIGKILL to worker process")
                                .emit();
                        }
                    }
                    Err(e) => {
                        NARRATE
                            .action("worker_delete_kill_error")
                            .job_id(&job_id)
                            .context(&e.to_string())
                            .human("⚠️  Failed to kill process: {} (may already be dead)")
                            .emit();
                    }
                }
            }

            #[cfg(not(unix))]
            {
                NARRATE
                    .action("worker_delete_kill_unsupported")
                    .job_id(&job_id)
                    .human("⚠️  Process killing not implemented for this platform")
                    .emit();
            }

            // Remove from registry
            state.worker_registry.remove(&id)?;

            NARRATE
                .action("worker_delete_complete")
                .job_id(&job_id)
                .context(&id)
                .human("✅ Worker deleted: {}")
                .emit();
        }
        Err(e) => {
            NARRATE
                .action("worker_delete_error")
                .job_id(&job_id)
                .context(&id)
                .context(&e.to_string())
                .human("❌ Delete failed for '{}': {}")
                .emit();
            return Err(e);
        }
    }
}
```

### Step 5: Optional - Create Management Module

If you want to separate process cleanup logic:

**File:** `worker-lifecycle/src/management.rs`

```rust
// TEAM-272: Worker management utilities
use anyhow::Result;

/// Kill a worker process by PID
///
/// Tries SIGTERM first (graceful), then SIGKILL if needed.
#[cfg(unix)]
pub async fn kill_worker_process(pid: u32) -> Result<()> {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    // Try SIGTERM first
    kill(Pid::from_raw(pid as i32), Signal::SIGTERM)?;

    // Wait for graceful shutdown
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Force kill if still running
    let _ = kill(Pid::from_raw(pid as i32), Signal::SIGKILL);

    Ok(())
}

#[cfg(not(unix))]
pub async fn kill_worker_process(_pid: u32) -> Result<()> {
    anyhow::bail!("Process killing not implemented for this platform")
}
```

Then use in job_router.rs:
```rust
use rbee_hive_worker_lifecycle::management::kill_worker_process;

// In WorkerDelete operation
kill_worker_process(worker.pid).await?;
```

---

## ✅ Acceptance Criteria

- [ ] WorkerList operation implemented
- [ ] WorkerGet operation implemented
- [ ] WorkerDelete operation implemented
- [ ] Process cleanup working (Unix) or documented (Windows)
- [ ] All operations emit narration with `.job_id()`
- [ ] `cargo check --bin rbee-hive` passes
- [ ] Manual testing shows operations work

---

## 🧪 Testing Commands

```bash
# Check compilation
cargo check --bin rbee-hive

# Manual testing
cargo run --bin rbee-hive -- --port 8600

# In another terminal:

# List workers
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "worker_list", "hive_id": "localhost"}'

# Get worker
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "worker_get", "hive_id": "localhost", "id": "worker-1"}'

# Delete worker
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "worker_delete", "hive_id": "localhost", "id": "worker-1"}'
```

---

## 📝 Handoff Checklist

Create `TEAM_272_HANDOFF.md` with:

- [ ] All 3 operations working
- [ ] Example narration output
- [ ] Process cleanup demonstrated
- [ ] Known limitations (platform-specific)
- [ ] Notes for TEAM-273

---

## 🚨 Known Limitations

### 1. Platform-Specific Process Killing

**Unix:** Uses `nix` crate for SIGTERM/SIGKILL.

**Windows:** Not implemented - document this limitation.

**Workaround:** Use conditional compilation:
```rust
#[cfg(unix)]
{
    // Unix-specific code
}

#[cfg(not(unix))]
{
    // Log warning
}
```

### 2. No Process Monitoring

**Current:** No verification that process actually died.

**Future:** Check process status after kill:
```rust
// Check if process still exists
if std::fs::read_to_string(format!("/proc/{}/status", pid)).is_ok() {
    // Process still alive
}
```

### 3. No Graceful Shutdown Protocol

**Current:** Just sends SIGTERM, no coordination with worker.

**Future:** Implement graceful shutdown:
1. Send shutdown request to worker HTTP API
2. Wait for acknowledgment
3. Fall back to SIGTERM if no response

---

## 🎓 Example Narration Output

### WorkerList (Empty)
```
[hv-router] worker_list_start: 📋 Listing workers on hive 'localhost'
[hv-router] worker_list_result: Found 0 worker(s)
[hv-router] worker_list_empty: No workers found
```

### WorkerList (With Workers)
```
[hv-router] worker_list_start: 📋 Listing workers on hive 'localhost'
[hv-router] worker_list_result: Found 2 worker(s)
[hv-router] worker_list_entry:   worker-1 | test-model | GPU-0 | port 9100 | ready
[hv-router] worker_list_entry:   worker-2 | test-model | CPU-0 | port 9101 | ready
```

### WorkerGet
```
[hv-router] worker_get_start: 🔍 Getting worker 'worker-1' on hive 'localhost'
[hv-router] worker_get_found: ✅ Worker: worker-1 | Model: test-model | Device: GPU-0 | Port: 9100
[hv-router] worker_get_details: {
  "id": "worker-1",
  "model_id": "test-model",
  "device": "GPU-0",
  "pid": 12345,
  "port": 9100,
  "status": "Ready",
  "started_at": "2025-10-23T15:00:00Z"
}
```

### WorkerDelete
```
[hv-router] worker_delete_start: 🗑️  Deleting worker 'worker-1' on hive 'localhost'
[hv-router] worker_delete_kill: Killing process PID 12345
[hv-router] worker_delete_sigterm: Sent SIGTERM to worker process
[hv-router] worker_delete_complete: ✅ Worker deleted: worker-1
```

---

## 📚 Reference Implementations

- **TEAM-268:** Model operations (similar CRUD pattern)
- **hive-lifecycle/stop.rs:** Process killing patterns
- **TEAM-211:** Simple operations (narration patterns)

---

## 🎯 Success Metrics

**Code:**
- WorkerList: ~40 lines
- WorkerGet: ~35 lines
- WorkerDelete: ~60 lines (with process cleanup)
- Total: ~135 lines

**Operations:**
- ✅ All 3 operations working
- ✅ Full narration support
- ✅ Process cleanup (Unix)
- ✅ Error handling

---

**TEAM-272: Complete the worker lifecycle! 🎯🚀**

**After this, TEAM-273 will integrate everything!**
