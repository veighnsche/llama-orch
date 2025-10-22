# TEAM-259: daemon-lifecycle Extensions (REVISED)

**Status:** 📋 PROPOSAL (REVISED)

**Date:** Oct 23, 2025

**Mission:** Identify patterns that should be shared between hive-lifecycle and worker-lifecycle.

---

## Architecture Clarifications

### ✅ Confirmed Architecture

```
rbee-keeper (CLI)
    ↓ [polling + ensure running]
queen-rbee (daemon on control machine)
    ↓ [SSH to remote machines]
rbee-hive (daemon on worker machine)
    ↓ [local spawn, heartbeat monitoring]
llm-worker (process on same machine as hive)
```

### Key Points

1. **SSH is ONLY for queen → hive**
   - Workers are NEVER remote from hive
   - Workers run on the same machine as their hive
   - Hive controls the machine, workers are local processes

2. **Polling is ONLY for rbee-keeper → queen**
   - Queen uses heartbeat for hive health
   - Hive uses heartbeat for worker health
   - No exponential backoff needed in hive/worker

3. **Multiple Worker Types**
   - LLM workers (vLLM, llama.cpp)
   - Stable Diffusion workers
   - Whisper workers
   - CPU and CUDA variants

4. **Install/Uninstall is Common**
   - Install stable diffusion worker
   - Install whisper worker
   - Install CUDA LLM worker
   - Install CPU LLM worker
   - Fast operations

---

## What Should Be Shared?

### ❌ NOT Shared (Keep Separate)

**SSH Operations** - Only queen → hive
- Location: `hive-lifecycle/src/ssh_helper.rs`
- Used by: hive-lifecycle ONLY
- Reason: Workers are local to hive

**Polling with Backoff** - Only rbee-keeper → queen
- Location: `queen-lifecycle/src/health.rs`
- Used by: queen-lifecycle ONLY
- Reason: Hive/workers use heartbeat

**Capabilities** - Hive-specific
- Location: `hive-lifecycle/src/capabilities.rs`
- Used by: hive-lifecycle ONLY
- Reason: Hive-specific hardware detection

---

### ✅ SHOULD Be Shared

#### 1. 🎯 Install/Uninstall Pattern

**Current:**
- `hive-lifecycle/src/install.rs` (306 lines)
- `hive-lifecycle/src/uninstall.rs` (4305 bytes)

**Pattern:**
```rust
// Install: Find binary, copy if remote, configure
// Uninstall: Remove config, cleanup

// For workers:
// - Install stable diffusion worker
// - Install whisper worker
// - Install CUDA LLM worker
// - Install CPU LLM worker
```

**Proposed:** `daemon-lifecycle/src/install.rs`
```rust
pub struct InstallConfig {
    pub binary_name: String,
    pub binary_path: Option<String>,
    pub target_path: Option<String>,
}

pub async fn install_daemon(config: InstallConfig) -> Result<String> {
    // 1. Find binary (local or provided path)
    // 2. Verify binary exists
    // 3. Return installation path
}

pub async fn uninstall_daemon(daemon_name: &str) -> Result<()> {
    // 1. Check if running (error if yes)
    // 2. Remove configuration
    // 3. Cleanup
}
```

**Would be used by:**
- hive-lifecycle (install/uninstall hive)
- worker-lifecycle (install/uninstall workers)

---

#### 2. 🎯 List Pattern

**Current:**
- `hive-lifecycle/src/list.rs` (82 lines)

**Pattern:**
```rust
// List all configured instances
// Returns: Vec<Info>

// For hives: List all configured hives
// For workers: List all workers on a hive
```

**Proposed:** `daemon-lifecycle/src/list.rs`
```rust
pub trait ListableConfig {
    type Info;
    fn list_all(&self) -> Vec<Self::Info>;
}

pub async fn list_daemons<T: ListableConfig>(
    config: &T,
    job_id: &str,
) -> Result<Vec<T::Info>> {
    // Generic list implementation
}
```

**Would be used by:**
- hive-lifecycle (list hives)
- worker-lifecycle (list workers)

---

#### 3. 🎯 Get Pattern

**Current:**
- `hive-lifecycle/src/get.rs` (1685 bytes)

**Pattern:**
```rust
// Get details for a specific instance by ID/alias
// Returns: Info struct

// For hives: Get hive by alias
// For workers: Get worker by ID
```

**Proposed:** `daemon-lifecycle/src/get.rs`
```rust
pub trait GettableConfig {
    type Info;
    fn get_by_id(&self, id: &str) -> Option<Self::Info>;
}

pub async fn get_daemon<T: GettableConfig>(
    config: &T,
    id: &str,
    job_id: &str,
) -> Result<T::Info> {
    // Generic get implementation
}
```

**Would be used by:**
- hive-lifecycle (get hive)
- worker-lifecycle (get worker)

---

#### 4. 🎯 Status Pattern

**Current:**
- `hive-lifecycle/src/status.rs` (81 lines)

**Pattern:**
```rust
// Check if daemon is running via HTTP health check
// Returns: StatusResponse { running: bool, health_url: String }

// For hives: Check hive status
// For workers: Check worker status
```

**Proposed:** `daemon-lifecycle/src/status.rs`
```rust
pub struct StatusRequest {
    pub id: String,
    pub health_url: String,
}

pub struct StatusResponse {
    pub id: String,
    pub running: bool,
    pub health_url: String,
}

pub async fn check_daemon_status(
    request: StatusRequest,
    job_id: &str,
) -> Result<StatusResponse> {
    // HTTP health check with 5s timeout
    // Returns running status
}
```

**Would be used by:**
- hive-lifecycle (hive status)
- worker-lifecycle (worker status)

---

#### 5. ✅ Already Shared (Keep)

**Spawn Daemon** - `daemon-lifecycle/src/manager.rs`
```rust
pub async fn spawn_daemon(binary_path: &str, args: Vec<String>) -> Result<Child>
```
- Used by: hive-lifecycle (start.rs), worker-lifecycle (future)

**Health Check** - `daemon-lifecycle/src/health.rs`
```rust
pub async fn is_daemon_healthy(base_url: &str) -> bool
```
- Used by: hive-lifecycle (start.rs), worker-lifecycle (future)

**Ensure Running** - `daemon-lifecycle/src/ensure.rs`
```rust
pub async fn ensure_daemon_running(...) -> Result<bool>
```
- Used by: queen-lifecycle ONLY (rbee-keeper → queen)
- NOT used by hive/worker (they use heartbeat)

---

## Proposed daemon-lifecycle Structure

### Current
```
daemon-lifecycle/
├── lib.rs
├── manager.rs - spawn_daemon, find_in_target
├── health.rs - is_daemon_healthy
└── ensure.rs - ensure_daemon_running (queen-only)
```

### Proposed
```
daemon-lifecycle/
├── lib.rs
├── manager.rs - spawn_daemon, find_in_target
├── health.rs - is_daemon_healthy
├── ensure.rs - ensure_daemon_running (queen-only)
├── install.rs - install/uninstall daemon (NEW)
├── list.rs - list daemons (NEW)
├── get.rs - get daemon by ID (NEW)
└── status.rs - check daemon status (NEW)
```

---

## Comparison: hive-lifecycle vs worker-lifecycle

### hive-lifecycle (queen → hive)
```
hive-lifecycle/
├── install.rs - Install hive ✅ SHARED
├── uninstall.rs - Uninstall hive ✅ SHARED
├── start.rs - Start hive (uses daemon-lifecycle::spawn_daemon)
├── stop.rs - Stop hive (uses daemon-lifecycle::graceful_shutdown)
├── list.rs - List hives ✅ SHARED
├── get.rs - Get hive ✅ SHARED
├── status.rs - Hive status ✅ SHARED
├── capabilities.rs - Fetch capabilities ❌ HIVE-ONLY
├── ssh_helper.rs - SSH operations ❌ HIVE-ONLY
├── ssh_test.rs - SSH testing ❌ HIVE-ONLY
├── hive_client.rs - HTTP client ❌ HIVE-ONLY
├── types.rs - Hive types
└── validation.rs - Hive validation
```

### worker-lifecycle (hive → worker)
```
worker-lifecycle/
├── install.rs - Install worker ✅ SHARED
├── uninstall.rs - Uninstall worker ✅ SHARED
├── spawn.rs - Spawn worker (uses daemon-lifecycle::spawn_daemon)
├── stop.rs - Stop worker (uses daemon-lifecycle::graceful_shutdown)
├── list.rs - List workers ✅ SHARED
├── get.rs - Get worker ✅ SHARED
├── status.rs - Worker status ✅ SHARED
├── vllm.rs - vLLM-specific ❌ WORKER-ONLY
├── llamacpp.rs - llama.cpp-specific ❌ WORKER-ONLY
├── stable_diffusion.rs - SD-specific ❌ WORKER-ONLY
├── whisper.rs - Whisper-specific ❌ WORKER-ONLY
├── types.rs - Worker types
└── validation.rs - Worker validation
```

---

## Implementation Priority

### Phase 1: Core CRUD Operations (High Priority)

**1. Install/Uninstall** 🔥
- **Complexity:** Medium
- **Impact:** High
- **LOC:** ~150 lines
- **Reusability:** hive-lifecycle, worker-lifecycle
- **Benefit:** Fast worker installation (SD, Whisper, LLM variants)

**2. List** 🔥
- **Complexity:** Low
- **Impact:** Medium
- **LOC:** ~50 lines
- **Reusability:** hive-lifecycle, worker-lifecycle
- **Benefit:** Consistent list interface

**3. Get** 🔥
- **Complexity:** Low
- **Impact:** Medium
- **LOC:** ~40 lines
- **Reusability:** hive-lifecycle, worker-lifecycle
- **Benefit:** Consistent get interface

**4. Status** 🔥
- **Complexity:** Low
- **Impact:** Medium
- **LOC:** ~60 lines
- **Reusability:** hive-lifecycle, worker-lifecycle
- **Benefit:** Consistent status checking

---

## Benefits

### Code Reduction
- **hive-lifecycle:** Can remove ~250 LOC (install, list, get, status)
- **worker-lifecycle:** Won't need to implement these patterns (~250 LOC saved)
- **Total:** ~500 LOC saved

### Consistency
- ✅ Same install/uninstall behavior
- ✅ Same list/get/status interface
- ✅ Same error handling
- ✅ Same narration patterns

### Worker Type Flexibility
- ✅ Easy to add new worker types
- ✅ Shared CRUD operations
- ✅ Type-specific logic in worker-lifecycle

---

## Worker Lifecycle Structure

### Proposed worker-lifecycle
```
worker-lifecycle/
├── lib.rs
├── types.rs
│   ├── WorkerType enum (VLlm, LlamaCpp, StableDiffusion, Whisper)
│   ├── WorkerHandle
│   └── WorkerInfo
├── spawn.rs - Uses daemon-lifecycle::spawn_daemon
├── stop.rs - Uses daemon-lifecycle::graceful_shutdown
├── install.rs - Uses daemon-lifecycle::install_daemon
├── uninstall.rs - Uses daemon-lifecycle::uninstall_daemon
├── list.rs - Uses daemon-lifecycle::list_daemons
├── get.rs - Uses daemon-lifecycle::get_daemon
├── status.rs - Uses daemon-lifecycle::check_daemon_status
├── vllm.rs - vLLM-specific config
├── llamacpp.rs - llama.cpp-specific config
├── stable_diffusion.rs - SD-specific config
└── whisper.rs - Whisper-specific config
```

---

## Summary

### ✅ Should Be Shared (daemon-lifecycle)
1. **Install/Uninstall** - Binary installation pattern
2. **List** - List all instances
3. **Get** - Get instance by ID
4. **Status** - Health check status
5. **Spawn** - Already shared ✅
6. **Health Check** - Already shared ✅

### ❌ Should NOT Be Shared
1. **SSH Operations** - Only queen → hive
2. **Polling with Backoff** - Only rbee-keeper → queen
3. **Capabilities** - Hive-specific hardware detection
4. **Worker Type Logic** - Worker-specific (vLLM, llama.cpp, etc.)

### Benefits
- ✅ ~500 LOC saved
- ✅ Consistent CRUD operations
- ✅ Easy to add new worker types
- ✅ Fast worker installation (SD, Whisper, LLM variants)

**This will make worker-lifecycle implementation much cleaner!** 🎯
