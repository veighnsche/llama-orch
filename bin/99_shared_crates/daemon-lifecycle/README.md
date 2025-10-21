# daemon-lifecycle

**Status:** ✅ Production Ready (TEAM-197 narration v0.5.0 migration)  
**Purpose:** Shared daemon lifecycle management for all rbee binaries  
**Location:** `bin/99_shared_crates/daemon-lifecycle/` (SHARED - used by 3 binaries)

---

## Overview

The `daemon-lifecycle` crate provides **shared daemon lifecycle management** functionality for managing daemon processes across the rbee system. It eliminates code duplication by providing common operations for starting, stopping, monitoring, and restarting daemon processes.

### System Context

In the llama-orch architecture, there is a **three-level daemon management chain**:

```
┌─────────────────┐
│  rbee-keeper    │  ← CLI tool
│     (CLI)       │  ├─ rbee-keeper-crates/queen-lifecycle
│                 │  │  └─> daemon-lifecycle (THIS CRATE)
└────────┬────────┘
         │ Manages
         ↓
┌─────────────────┐
│   queen-rbee    │  ← Orchestrator daemon
│ (orchestratord) │  ├─ queen-rbee-crates/hive-lifecycle
│                 │  │  └─> daemon-lifecycle (THIS CRATE)
└────────┬────────┘
         │ Manages
         ↓
┌─────────────────┐
│   rbee-hive     │  ← Pool manager daemon
│ (pool-managerd) │  ├─ rbee-hive-crates/worker-lifecycle
│                 │  │  └─> daemon-lifecycle (THIS CRATE)
└────────┬────────┘
         │ Manages
         ↓
┌─────────────────┐
│ llm-worker-rbee │  ← Worker process (no lifecycle management)
│  (worker-orcd)  │
└─────────────────┘
```

**Key Responsibilities:**
- Start daemon processes
- Stop daemon processes gracefully
- Check daemon status (running/stopped/failed)
- Restart daemons with config reload
- Manage PID files
- Handle process monitoring
- Cleanup on failure

---

## Why This is Shared

### Code Duplication Problem

**Before (duplicated across 3 binaries):**
- `rbee-keeper` → `queen-rbee` lifecycle: ~132 LOC
- `queen-rbee` → `rbee-hive` lifecycle: ~800 LOC
- `rbee-hive` → `llm-worker` lifecycle: ~386 LOC
- **Total:** ~1,318 LOC (duplicated!)

**After (shared crate):**
- `daemon-lifecycle`: ~500 LOC (shared)
- Binary-specific wrappers: ~50 LOC each × 3 = ~150 LOC
- **Total:** ~650 LOC

**Savings:** ~668 LOC (~50% reduction)

### Used By

**Three binary-specific lifecycle crates:**
1. `rbee-keeper-crates/queen-lifecycle` - Keeper manages queen
2. `queen-rbee-crates/hive-lifecycle` - Queen manages hives
3. `rbee-hive-crates/worker-lifecycle` - Hive manages workers

---

## Architecture Principles

### 1. Generic Daemon Management

**Common operations:**
- Process spawning (fork/exec or tokio::process)
- PID file management (create, read, delete)
- Status checking (is process running?)
- Graceful shutdown (SIGTERM → wait → SIGKILL)
- Health checks (HTTP ping, process alive)

### 2. Binary-Agnostic Design

**This crate doesn't know about:**
- What binary it's managing (queen, hive, worker)
- Binary-specific configuration
- Binary-specific health checks

**This crate only knows:**
- How to start a process
- How to stop a process
- How to check if a process is running
- How to manage PID files

### 3. Extensible via Traits

**Binary-specific crates provide:**
- Binary path
- Command-line arguments
- Configuration file paths
- Health check endpoints
- Custom startup/shutdown logic

---

## API Design

### Core Types

```rust
/// Daemon manager
pub struct DaemonManager {
    config: DaemonConfig,
}

/// Daemon configuration
pub struct DaemonConfig {
    /// Path to daemon binary
    pub binary_path: PathBuf,
    
    /// Command-line arguments
    pub args: Vec<String>,
    
    /// PID file path
    pub pid_file: PathBuf,
    
    /// Working directory
    pub working_dir: Option<PathBuf>,
    
    /// Environment variables
    pub env: HashMap<String, String>,
    
    /// Startup timeout (seconds)
    pub startup_timeout_secs: u64,
    
    /// Shutdown timeout (seconds)
    pub shutdown_timeout_secs: u64,
}

/// Daemon status
pub enum DaemonStatus {
    /// Daemon is running with PID
    Running(u32),
    
    /// Daemon is stopped
    Stopped,
    
    /// Daemon failed to start or crashed
    Failed(String),
    
    /// PID file exists but process is not running (stale)
    Stale(u32),
}
```

### Core API

```rust
impl DaemonManager {
    /// Create new daemon manager
    pub fn new(config: DaemonConfig) -> Self;
    
    /// Start daemon
    pub fn start(&self) -> Result<u32>;
    
    /// Stop daemon gracefully
    pub fn stop(&self) -> Result<()>;
    
    /// Force kill daemon
    pub fn kill(&self) -> Result<()>;
    
    /// Check daemon status
    pub fn status(&self) -> Result<DaemonStatus>;
    
    /// Restart daemon
    pub fn restart(&self) -> Result<u32>;
    
    /// Wait for daemon to be ready
    pub fn wait_ready(&self, timeout_secs: u64) -> Result<()>;
    
    /// Get daemon PID
    pub fn get_pid(&self) -> Result<Option<u32>>;
    
    /// Check if daemon is running
    pub fn is_running(&self) -> bool;
}
```

---

## Usage Examples

### Basic Start/Stop

```rust
use daemon_lifecycle::{DaemonManager, DaemonConfig};
use std::path::PathBuf;

// Configure daemon
let config = DaemonConfig {
    binary_path: PathBuf::from("/usr/local/bin/queen-rbee"),
    args: vec!["--config".to_string(), "/etc/rbee/queen.toml".to_string()],
    pid_file: PathBuf::from("/var/run/queen-rbee.pid"),
    working_dir: None,
    env: HashMap::new(),
    startup_timeout_secs: 30,
    shutdown_timeout_secs: 10,
};

// Create manager
let manager = DaemonManager::new(config);

// Start daemon
let pid = manager.start()?;
println!("Started daemon with PID {}", pid);

// Check status
let status = manager.status()?;
println!("Status: {:?}", status);

// Stop daemon
manager.stop()?;
println!("Stopped daemon");
```

### Check Status

```rust
use daemon_lifecycle::{DaemonManager, DaemonStatus};

let manager = DaemonManager::new(config);

match manager.status()? {
    DaemonStatus::Running(pid) => {
        println!("Daemon is running with PID {}", pid);
    }
    DaemonStatus::Stopped => {
        println!("Daemon is stopped");
    }
    DaemonStatus::Failed(error) => {
        eprintln!("Daemon failed: {}", error);
    }
    DaemonStatus::Stale(pid) => {
        eprintln!("Stale PID file found (PID {}), cleaning up", pid);
        manager.cleanup_stale()?;
    }
}
```

### Restart with Config Reload

```rust
use daemon_lifecycle::DaemonManager;

let manager = DaemonManager::new(config);

// Restart daemon (stop + start)
let new_pid = manager.restart()?;
println!("Restarted daemon with new PID {}", new_pid);
```

### Wait for Daemon to be Ready

```rust
use daemon_lifecycle::DaemonManager;

let manager = DaemonManager::new(config);

// Start daemon
manager.start()?;

// Wait for daemon to be ready (e.g., HTTP server listening)
manager.wait_ready(30)?; // Wait up to 30 seconds

println!("Daemon is ready!");
```

---

## PID File Management

### PID File Format

**Simple text file with PID:**
```
12345
```

### PID File Operations

**Create PID file:**
```rust
// Automatically created on start()
manager.start()?; // Creates /var/run/daemon.pid
```

**Read PID file:**
```rust
let pid = manager.get_pid()?;
if let Some(pid) = pid {
    println!("Daemon PID: {}", pid);
}
```

**Delete PID file:**
```rust
// Automatically deleted on stop()
manager.stop()?; // Deletes /var/run/daemon.pid
```

**Cleanup stale PID file:**
```rust
// If PID file exists but process is not running
manager.cleanup_stale()?;
```

---

## Graceful Shutdown

### Shutdown Sequence

1. **Send SIGTERM** - Request graceful shutdown
2. **Wait** - Give process time to cleanup (default: 10s)
3. **Check** - Is process still running?
4. **Send SIGKILL** - Force kill if still running
5. **Cleanup** - Delete PID file

### Example

```rust
use daemon_lifecycle::DaemonManager;

let manager = DaemonManager::new(config);

// Graceful shutdown (SIGTERM → wait → SIGKILL)
manager.stop()?;

// Force kill (SIGKILL immediately)
manager.kill()?;
```

---

## Process Monitoring

### Health Checks

**Binary-specific crates can implement custom health checks:**

```rust
use daemon_lifecycle::{DaemonManager, DaemonStatus};

let manager = DaemonManager::new(config);

// Start daemon
manager.start()?;

// Custom health check (HTTP ping)
async fn check_health(url: &str) -> Result<bool> {
    let response = reqwest::get(url).await?;
    Ok(response.status().is_success())
}

// Wait for daemon to be healthy
let mut attempts = 0;
while attempts < 30 {
    if check_health("http://localhost:8080/health").await? {
        println!("Daemon is healthy!");
        break;
    }
    tokio::time::sleep(Duration::from_secs(1)).await;
    attempts += 1;
}
```

---

## Dependencies

### Required

- **`tokio`**: Async runtime for process spawning
- **`anyhow`**: Error handling
- **`observability-narration-core`**: Structured observability (v0.5.0+)

### Optional

- **`serde`**: Configuration serialization
- **`reqwest`**: HTTP health checks (for binary-specific crates)

---

## Implementation Status

### Phase 1: Core Lifecycle (M1)
- [ ] `DaemonManager` implementation
- [ ] Start/stop/status operations
- [ ] PID file management
- [ ] Graceful shutdown (SIGTERM → SIGKILL)
- [ ] Unit tests

### Phase 2: Advanced Features (M2)
- [ ] Health check support
- [ ] Restart with config reload
- [ ] Process monitoring
- [ ] Crash recovery
- [ ] Integration tests

### Phase 3: Production Features (M3)
- [ ] Log rotation integration
- [ ] Resource limits (CPU, memory)
- [ ] Systemd integration
- [ ] Docker support

---

## Error Handling

### Error Types

```rust
pub enum DaemonError {
    /// Failed to start daemon
    StartFailed(String),
    
    /// Failed to stop daemon
    StopFailed(String),
    
    /// PID file error
    PidFileError(String),
    
    /// Process not found
    ProcessNotFound(u32),
    
    /// Timeout waiting for daemon
    Timeout(String),
    
    /// Permission denied
    PermissionDenied(String),
}
```

---

## Testing Strategy

### Unit Tests

- PID file creation/deletion
- Status checking
- Graceful shutdown logic
- Error handling

### Integration Tests

- Start/stop real processes
- PID file management
- Timeout handling
- Stale PID cleanup

### Mock Tests

- Mock process spawning
- Mock signal sending
- Simulate failures

---

## Related Crates

### Used By
- **`rbee-keeper-crates/queen-lifecycle`**: Keeper manages queen
- **`queen-rbee-crates/hive-lifecycle`**: Queen manages hives
- **`rbee-hive-crates/worker-lifecycle`**: Hive manages workers

### Related Documentation
- **Usage guide:** `USAGE.md`
- **Architecture:** `bin/.plan/TEAM_130G_FINAL_ARCHITECTURE.md`
- **Consolidation:** `bin/.plan/TEAM_130E_CONSOLIDATION_SUMMARY.md`

---

## Specification References

- **SYS-6.1.x**: Orchestrator (manages hives)
- **SYS-6.2.x**: Pool Manager (manages workers)
- **Process isolation:** SYS-2.4.x

See: `/home/vince/Projects/llama-orch/bin/.specs/00_llama-orch.md`

---

## Team History

- **TEAM-130E**: Identified lifecycle duplication across binaries
- **TEAM-135**: Scaffolding for new crate-based architecture
- **TEAM-152**: Implemented core daemon spawning functionality with narration
- **TEAM-197**: Migrated to narration-core v0.5.0 pattern (NarrationFactory, fixed-width format)

---

**Next Steps:**
1. Implement `DaemonManager` core functionality
2. Add PID file management
3. Implement graceful shutdown
4. Add tests
5. Integrate with binary-specific lifecycle crates
