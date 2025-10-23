# TEAM-259: daemon-lifecycle Extensions Proposal

**Status:** 📋 PROPOSAL

**Date:** Oct 23, 2025

**Mission:** Identify patterns from hive-lifecycle that should be generalized into daemon-lifecycle for reuse across queen, hive, and worker lifecycles.

---

## Current State

### daemon-lifecycle (Shared)
```
daemon-lifecycle/
├── manager.rs - DaemonManager, spawn_daemon()
├── health.rs - is_daemon_healthy()
└── ensure.rs - ensure_daemon_running()
```

### Specialized Lifecycles
```
queen-lifecycle/    (rbee-keeper → queen-rbee)
├── Uses: daemon-lifecycle ✅
└── Queen-specific logic

hive-lifecycle/     (queen-rbee → rbee-hive)
├── Uses: daemon-lifecycle ✅
├── SSH remote operations
├── Graceful shutdown (SIGTERM → SIGKILL)
└── Capabilities caching

worker-lifecycle/   (rbee-hive → llm-worker)
├── Status: STUB (not implemented)
└── Will need: spawn, stop, health check
```

---

## Patterns to Extract

### 1. ✅ Already in daemon-lifecycle

**Pattern:** Basic daemon spawning
- `spawn_daemon()` - Spawn with Stdio::null()
- `DaemonManager::find_in_target()` - Find binary

**Pattern:** Ensure running
- `ensure_daemon_running()` - Check → spawn → wait
- `is_daemon_healthy()` - HTTP /health check

**Used by:** queen-lifecycle ✅, hive-lifecycle ✅

---

### 2. 🎯 Should Add: Graceful Shutdown

**Location:** `hive-lifecycle/src/stop.rs` (lines 77-198)

**Pattern:**
```rust
// 1. Send SIGTERM (graceful shutdown)
// 2. Wait 5 seconds for process to exit
// 3. If still running, send SIGKILL (force kill)
// 4. Verify process is dead
```

**Current Implementation:**
```rust
// hive-lifecycle/src/stop.rs
if is_remote {
    // SSH: pkill -TERM rbee-hive
    ssh_exec(&format!("pkill -TERM {}", binary_name), ...)?;
    sleep(Duration::from_secs(5)).await;
    
    // Check if still running
    if still_running {
        // Force kill
        ssh_exec(&format!("pkill -KILL {}", binary_name), ...)?;
    }
} else {
    // Local: same pattern with local commands
}
```

**Proposed Addition:**
```rust
// daemon-lifecycle/src/shutdown.rs

pub async fn graceful_shutdown(
    daemon_name: &str,
    grace_period: Option<Duration>,
) -> Result<()> {
    // 1. SIGTERM
    // 2. Wait grace_period (default 5s)
    // 3. SIGKILL if needed
}

pub async fn graceful_shutdown_remote(
    daemon_name: &str,
    ssh_config: &SshConfig,
    grace_period: Option<Duration>,
) -> Result<()> {
    // Same but via SSH
}
```

**Benefits:**
- ✅ Reusable across queen, hive, worker
- ✅ Consistent shutdown behavior
- ✅ Single place to fix bugs

**Would be used by:**
- hive-lifecycle (stop.rs)
- worker-lifecycle (future)
- queen-lifecycle (if needed)

---

### 3. 🎯 Should Add: Remote Daemon Operations

**Location:** `hive-lifecycle/src/ssh_helper.rs` (5748 bytes)

**Pattern:**
```rust
// SSH-based daemon operations
pub fn is_remote_hive(config: &HiveEntry) -> bool
pub fn get_remote_binary_path(config: &HiveEntry) -> String
pub async fn ssh_exec(cmd: &str, config: &SshConfig) -> Result<String>
```

**Current Usage:**
```rust
// hive-lifecycle/src/start.rs
if is_remote_hive(hive_config) {
    let binary_path = get_remote_binary_path(hive_config);
    let start_cmd = format!("nohup {} --port {} > /dev/null 2>&1 & echo $!", ...);
    let pid = ssh_exec(&start_cmd, ...)?;
}
```

**Proposed Addition:**
```rust
// daemon-lifecycle/src/remote.rs

pub struct SshConfig {
    pub hostname: String,
    pub ssh_user: String,
    pub ssh_port: u16,
}

pub async fn spawn_daemon_remote(
    binary_path: &str,
    args: Vec<String>,
    ssh_config: &SshConfig,
) -> Result<u32> {
    // Spawn daemon via SSH
    // Returns PID
}

pub async fn is_daemon_healthy_remote(
    base_url: &str,
    ssh_config: &SshConfig,
) -> bool {
    // Health check via SSH tunnel or direct
}

pub async fn stop_daemon_remote(
    daemon_name: &str,
    ssh_config: &SshConfig,
    graceful: bool,
) -> Result<()> {
    // Stop daemon via SSH
}
```

**Benefits:**
- ✅ Reusable for any remote daemon
- ✅ Consistent SSH handling
- ✅ Single place for SSH logic

**Would be used by:**
- hive-lifecycle (start.rs, stop.rs)
- worker-lifecycle (if workers are remote)

**Note:** This is more complex because it requires SSH dependencies. Might want to make this a separate feature flag or crate (`daemon-lifecycle-ssh`).

---

### 4. 🎯 Should Add: Poll with Exponential Backoff

**Location:** `queen-lifecycle/src/health.rs` (lines 52-116)

**Pattern:**
```rust
// Exponential backoff: 100ms → 200ms → 400ms → 800ms → 1600ms → 3200ms
let mut delay = Duration::from_millis(100);
let max_delay = Duration::from_millis(3200);

loop {
    if is_healthy().await {
        return Ok(());
    }
    
    if timeout_exceeded() {
        return Err(...);
    }
    
    sleep(delay).await;
    delay = std::cmp::min(delay * 2, max_delay);
}
```

**Proposed Addition:**
```rust
// daemon-lifecycle/src/health.rs

pub async fn poll_until_healthy(
    base_url: &str,
    timeout: Duration,
    initial_delay: Option<Duration>,
    max_delay: Option<Duration>,
) -> Result<()> {
    // Exponential backoff polling
    // Default: 100ms → 3200ms
}
```

**Benefits:**
- ✅ Consistent backoff strategy
- ✅ Configurable delays
- ✅ Reusable pattern

**Would be used by:**
- queen-lifecycle (already has this)
- hive-lifecycle (start.rs)
- worker-lifecycle (future)

---

### 5. ❌ Should NOT Add: Domain-Specific Logic

**Keep in specialized crates:**

#### hive-lifecycle specific:
- `capabilities.rs` - Fetch/cache hive capabilities
- `install.rs` / `uninstall.rs` - Binary installation
- `validation.rs` - Hive config validation
- `hive_client.rs` - HTTP client for hive API

#### queen-lifecycle specific:
- Preflight validation (config, hives)
- TimeoutEnforcer with progress bar
- QueenHandle type

#### worker-lifecycle specific (future):
- Worker type detection (vLLM, llama.cpp, etc.)
- Model loading
- Device allocation

---

## Proposed daemon-lifecycle Structure

### Current
```
daemon-lifecycle/
├── lib.rs
├── manager.rs - DaemonManager, spawn
├── health.rs - is_daemon_healthy
└── ensure.rs - ensure_daemon_running
```

### Proposed
```
daemon-lifecycle/
├── lib.rs
├── manager.rs - DaemonManager, spawn
├── health.rs - is_daemon_healthy, poll_until_healthy
├── ensure.rs - ensure_daemon_running
├── shutdown.rs - graceful_shutdown (NEW)
└── remote.rs - SSH operations (NEW, optional feature)
```

---

## Implementation Priority

### Phase 1: Core Extensions (High Priority)

**1. Graceful Shutdown** 🔥
- **Complexity:** Low
- **Impact:** High
- **LOC:** ~80 lines
- **Reusability:** All lifecycles
- **Dependencies:** None (just process signals)

**2. Poll with Exponential Backoff** 🔥
- **Complexity:** Low
- **Impact:** Medium
- **LOC:** ~40 lines
- **Reusability:** All lifecycles
- **Dependencies:** None

### Phase 2: Advanced Extensions (Medium Priority)

**3. Remote Operations** ⚠️
- **Complexity:** High
- **Impact:** Medium
- **LOC:** ~200 lines
- **Reusability:** hive-lifecycle, maybe worker-lifecycle
- **Dependencies:** SSH crate, feature flag
- **Consideration:** Might be better as separate `daemon-lifecycle-ssh` crate

---

## Recommendation

### ✅ Add to daemon-lifecycle NOW:

1. **Graceful Shutdown** (`shutdown.rs`)
   ```rust
   pub async fn graceful_shutdown(daemon_name: &str, grace_period: Option<Duration>) -> Result<()>
   pub async fn graceful_shutdown_by_pid(pid: u32, grace_period: Option<Duration>) -> Result<()>
   ```

2. **Enhanced Poll** (extend `health.rs`)
   ```rust
   pub async fn poll_until_healthy(
       base_url: &str,
       timeout: Duration,
       initial_delay: Option<Duration>,
       max_delay: Option<Duration>,
   ) -> Result<()>
   ```

### 🤔 Consider for Future:

3. **Remote Operations** (separate crate or feature flag)
   - Only if worker-lifecycle needs it
   - Could be `daemon-lifecycle-ssh` crate
   - Or feature flag: `daemon-lifecycle = { features = ["ssh"] }`

---

## Benefits

### Code Reduction
- **hive-lifecycle:** Can remove ~80 LOC (graceful shutdown)
- **queen-lifecycle:** Can remove ~40 LOC (poll backoff)
- **worker-lifecycle:** Won't need to implement these patterns

### Consistency
- ✅ Same shutdown behavior everywhere
- ✅ Same backoff strategy everywhere
- ✅ Same error handling

### Maintainability
- ✅ Fix bugs in one place
- ✅ Improve patterns in one place
- ✅ Test once, use everywhere

---

## Worker Lifecycle Considerations

### Multiple Worker Types

**Current workers:**
- vLLM workers
- llama.cpp workers
- (future) Other LLM backends

**Common patterns (should use daemon-lifecycle):**
- ✅ Spawn worker process
- ✅ Health check (HTTP /health)
- ✅ Graceful shutdown
- ✅ Poll until ready

**Worker-specific (keep in worker-lifecycle):**
- Model loading
- Device allocation (GPU/CPU)
- Worker type detection
- Backend-specific configuration

**Proposed worker-lifecycle structure:**
```
worker-lifecycle/
├── lib.rs
├── types.rs - WorkerHandle, WorkerType enum
├── spawn.rs - Uses daemon-lifecycle::spawn_daemon
├── health.rs - Uses daemon-lifecycle::is_daemon_healthy
├── ensure.rs - Uses daemon-lifecycle::ensure_daemon_running
├── stop.rs - Uses daemon-lifecycle::graceful_shutdown
├── vllm.rs - vLLM-specific logic
└── llamacpp.rs - llama.cpp-specific logic
```

---

## Summary

**Patterns to extract:**
1. ✅ **Graceful shutdown** - High priority, low complexity
2. ✅ **Poll with backoff** - High priority, low complexity
3. 🤔 **Remote operations** - Medium priority, high complexity

**Benefits:**
- Reduce duplication across queen/hive/worker lifecycles
- Consistent behavior
- Single source of truth

**Next steps:**
1. Implement graceful shutdown in daemon-lifecycle
2. Enhance poll_until_healthy with backoff
3. Refactor hive-lifecycle to use new functions
4. Implement worker-lifecycle using daemon-lifecycle

**This will make worker lifecycle implementation much easier!** 🎯
