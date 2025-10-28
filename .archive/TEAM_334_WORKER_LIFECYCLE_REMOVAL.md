# TEAM-334: Worker-Lifecycle Removal

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025

## Problem

The `worker-lifecycle` crate was:
1. ❌ **Broken** - Using non-existent `DaemonManager` API
2. ❌ **Redundant** - Duplicating functionality from `daemon-lifecycle`
3. ❌ **Unmaintained** - Compilation errors not caught

## Solution

**Removed `worker-lifecycle` crate entirely** and wired `daemon-lifecycle` directly into `rbee-hive`.

## Changes Made

### 1. Deleted Crate

```bash
rm -rf /home/vince/Projects/llama-orch/bin/25_rbee_hive_crates/worker-lifecycle
```

### 2. Updated Workspace

**File:** `Cargo.toml`

```diff
- "bin/25_rbee_hive_crates/worker-lifecycle",
+ # TEAM-334: worker-lifecycle removed - use daemon-lifecycle directly
```

### 3. Updated rbee-hive Dependencies

**File:** `bin/20_rbee_hive/Cargo.toml`

```diff
- rbee-hive-worker-lifecycle = { path = "../25_rbee_hive_crates/worker-lifecycle" }
+ # TEAM-334: Use daemon-lifecycle directly (worker-lifecycle removed)
+ daemon-lifecycle = { path = "../99_shared_crates/daemon-lifecycle" }
+
+ # TEAM-334: For process management (Unix only)
+ [target.'cfg(unix)'.dependencies]
+ nix = { version = "0.27", features = ["signal"] }
```

### 4. Rewrote Worker Operations

**File:** `bin/20_rbee_hive/src/job_router.rs`

#### WorkerSpawn

**Before (using worker-lifecycle):**
```rust
use rbee_hive_worker_lifecycle::{start_worker, WorkerStartConfig};

let config = WorkerStartConfig {
    worker_id: request.worker.clone(),
    model_id: request.model.clone(),
    device: request.device.to_string(),
    port,
    queen_url,
    job_id: job_id.clone(),
};

let result = start_worker(config).await?;
```

**After (using daemon-lifecycle):**
```rust
use daemon_lifecycle::{start_daemon, StartConfig, HttpDaemonConfig, SshConfig};
use rbee_hive_worker_catalog::{WorkerType, Platform};

// Determine worker type
let worker_type = match request.worker.as_str() {
    "cuda" => WorkerType::CudaLlm,
    "cpu" => WorkerType::CpuLlm,
    "metal" => WorkerType::MetalLlm,
    _ => return Err(anyhow::anyhow!("Unsupported worker type")),
};

// Find worker binary
let worker_binary = state.worker_catalog
    .find_by_type_and_platform(worker_type, Platform::current())?;

// Start worker using daemon-lifecycle
let daemon_config = HttpDaemonConfig::new(&worker_id, &base_url).with_args(args);
let config = StartConfig {
    ssh_config: SshConfig::localhost(),
    daemon_config,
    job_id: Some(job_id.clone()),
};

let pid = start_daemon(config).await?;
```

#### WorkerProcessList

**Before (using worker-lifecycle):**
```rust
use rbee_hive_worker_lifecycle::list_workers;
let processes = list_workers(&job_id).await?;
```

**After (direct ps command):**
```rust
let output = tokio::process::Command::new("ps")
    .args(&["aux"])
    .output()
    .await?;

let worker_lines: Vec<_> = stdout.lines()
    .filter(|line| line.contains("llm-worker") || line.contains("worker-rbee"))
    .collect();
```

#### WorkerProcessGet

**Before (using worker-lifecycle):**
```rust
use rbee_hive_worker_lifecycle::get_worker;
let proc_info = get_worker(&job_id, pid).await?;
```

**After (direct ps command):**
```rust
let output = tokio::process::Command::new("ps")
    .args(&["-p", &pid.to_string(), "-o", "pid,command"])
    .output()
    .await?;
```

#### WorkerProcessDelete

**Before (using worker-lifecycle):**
```rust
use rbee_hive_worker_lifecycle::stop_worker;
stop_worker(&job_id, &worker_id, pid).await?;
```

**After (direct nix signal):**
```rust
#[cfg(unix)]
{
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    let pid_nix = Pid::from_raw(pid as i32);
    match kill(pid_nix, Signal::SIGTERM) {
        Ok(_) => {
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            let _ = kill(pid_nix, Signal::SIGKILL);
        }
        Err(_) => { /* already dead */ }
    }
}
```

## Benefits

1. ✅ **Simpler** - One less crate to maintain
2. ✅ **Working** - No more broken `DaemonManager` references
3. ✅ **Consistent** - Same lifecycle management as queen/hive daemons
4. ✅ **Direct** - No unnecessary abstraction layers
5. ✅ **Cleaner** - Process operations use standard Unix tools

## Files Changed

### Deleted
- `bin/25_rbee_hive_crates/worker-lifecycle/` (entire crate, ~1500 LOC)

### Modified
- `Cargo.toml` (workspace member removed)
- `bin/20_rbee_hive/Cargo.toml` (dependency updated, nix added)
- `bin/20_rbee_hive/src/job_router.rs` (worker operations rewritten)

## Verification

```bash
✅ cargo check -p rbee-hive
✅ Compilation successful
✅ All worker operations implemented
✅ No broken dependencies
```

## Architecture

**Before:**
```text
rbee-hive
    ↓
worker-lifecycle (broken)
    ↓
daemon-lifecycle (via DaemonManager - doesn't exist!)
```

**After:**
```text
rbee-hive
    ↓
daemon-lifecycle (direct)
    ↓
SshConfig::localhost() → local_exec() (no SSH)
```

## Key Decisions

1. **Direct ps commands** - Simpler than maintaining catalog of processes
2. **Direct nix signals** - Standard Unix process management
3. **daemon-lifecycle** - Reuse existing, working lifecycle management
4. **No abstraction** - Worker operations are simple enough to inline

## Related Work

- **TEAM-331:** Implemented localhost bypass in daemon-lifecycle
- **TEAM-332:** Implemented SSH config resolver middleware
- **TEAM-333:** Simplified queen localhost handling
- **TEAM-334:** Removed worker-lifecycle (this document)

## Summary

The `worker-lifecycle` crate was a broken abstraction that duplicated `daemon-lifecycle` functionality. By removing it and using `daemon-lifecycle` directly, we:

- ✅ Fixed compilation errors
- ✅ Reduced code complexity
- ✅ Improved maintainability
- ✅ Aligned with existing patterns

**Result:** Cleaner, working, maintainable code! 🎉
