# TEAM-261 Pivot Analysis: Hive as CLI (Not Daemon)

## PROS: Why Hive Should Be a CLI Tool

**Date:** Oct 23, 2025  
**Proposal:** Convert rbee-hive from daemon to CLI tool  
**Status:** 🟢 ANALYSIS - PROS

---

## Executive Summary

**Proposal:** Make `rbee-hive` a CLI tool (like `rbee-keeper`) instead of a long-running daemon.

**Key Insight:** Hive only manages worker lifecycle. It doesn't need to be always-running.

---

## 1. 🎯 Architectural Simplicity

### Current (Daemon)
```
queen-rbee (daemon)
    ↓ HTTP POST /v1/jobs
rbee-hive (daemon) ← Always running, waiting for requests
    ↓ Spawns workers
llm-worker-rbee (daemon)
    ↓ Heartbeat
queen-rbee (receives heartbeat)
```

### Proposed (CLI)
```
queen-rbee (daemon)
    ↓ SSH + CLI invocation
rbee-hive (CLI) ← Runs only when needed, exits immediately
    ↓ Spawns workers
llm-worker-rbee (daemon)
    ↓ Heartbeat
queen-rbee (receives heartbeat)
```

**Benefits:**
- ✅ One less daemon to manage
- ✅ No hive heartbeat needed
- ✅ No hive HTTP server needed
- ✅ No hive process monitoring needed

---

## 2. 🚀 Operational Simplicity

### No Daemon Management Overhead

**Current Problems:**
- Hive daemon must be started/stopped
- Hive daemon must be monitored for crashes
- Hive daemon must send heartbeats
- Hive daemon must maintain HTTP server
- Hive daemon must handle graceful shutdown

**With CLI:**
- ✅ No daemon to start/stop
- ✅ No daemon to monitor
- ✅ No heartbeats needed
- ✅ No HTTP server needed
- ✅ Process exits immediately after task

### Cleaner State Management

**Current (Daemon):**
```rust
// Hive maintains state in memory
struct HiveState {
    registry: Arc<JobRegistry<String>>,
    worker_registry: Arc<WorkerRegistry>,  // In-memory state
    model_catalog: Arc<ModelCatalog>,      // In-memory state
}
```

**Proposed (CLI):**
```rust
// No in-memory state! Everything on disk or in queen
// CLI reads config, performs action, exits
// Worker registry lives in queen (via heartbeats)
// Model catalog lives on disk
```

**Benefits:**
- ✅ No state synchronization issues
- ✅ No memory leaks
- ✅ No stale state
- ✅ Simpler debugging (no long-running state)

---

## 3. 💰 Resource Efficiency

### Memory Savings

**Current:**
- Hive daemon: ~50-100 MB RAM (always running)
- Hive HTTP server: Additional overhead
- Hive job registry: In-memory state

**Proposed:**
- Hive CLI: 0 MB when not running
- Hive CLI: ~10-20 MB for 1-2 seconds during execution
- No persistent memory footprint

**Impact:**
- ✅ 50-100 MB saved per machine
- ✅ Better for resource-constrained environments
- ✅ More memory for actual inference

### CPU Savings

**Current:**
- Hive daemon: Constant event loop
- Hive daemon: Heartbeat task (every 5s)
- Hive daemon: HTTP server polling

**Proposed:**
- Hive CLI: 0% CPU when not running
- Hive CLI: Brief CPU spike during execution
- No background tasks

---

## 4. 🔧 Simpler Deployment

### No Service Management

**Current:**
```bash
# Install hive
./rbee hive install localhost

# Start hive daemon
./rbee hive start localhost

# Stop hive daemon
./rbee hive stop localhost

# Uninstall hive
./rbee hive uninstall localhost
```

**Proposed:**
```bash
# Install hive (just copy binary)
./rbee hive install localhost

# No start/stop needed!
# Hive runs on-demand via SSH

# Uninstall hive (just delete binary)
./rbee hive uninstall localhost
```

**Benefits:**
- ✅ No systemd/init.d integration needed
- ✅ No daemon lifecycle management
- ✅ No "is hive running?" checks
- ✅ Simpler install/uninstall

### Easier Remote Execution

**Current:**
```rust
// Queen must ensure hive daemon is running
ensure_hive_running(&hive_id).await?;

// Then forward request via HTTP
let client = JobClient::new(&hive_url);
client.submit_and_stream(operation, ...).await?;
```

**Proposed:**
```rust
// Queen just runs CLI via SSH
ssh_exec(
    &hive_host,
    &format!("rbee-hive worker spawn --model {} --device {}", model, device)
).await?;
```

**Benefits:**
- ✅ No HTTP client needed
- ✅ No job-server pattern needed
- ✅ Direct SSH execution
- ✅ Simpler error handling

---

## 5. 🎭 Better Separation of Concerns

### Clear Responsibilities

**Current (Blurred):**
- Hive daemon: Manages workers + HTTP server + heartbeats + state
- Queen: Manages hives + workers (via hive) + inference routing

**Proposed (Clear):**
- Hive CLI: ONLY worker lifecycle (spawn/stop/list)
- Queen: Manages everything else (tracking, routing, scheduling)
- Workers: ONLY inference + heartbeats to queen

**Benefits:**
- ✅ Single source of truth (queen)
- ✅ No distributed state
- ✅ Clearer ownership
- ✅ Easier to reason about

### Worker Registry in Queen

**Current:**
- Hive maintains worker registry
- Hive sends aggregated heartbeats to queen
- Queen maintains separate hive registry
- Two sources of truth!

**Proposed:**
- Workers send heartbeats DIRECTLY to queen
- Queen maintains THE worker registry
- Hive doesn't track anything
- One source of truth!

**Benefits:**
- ✅ No state synchronization
- ✅ No aggregation overhead
- ✅ Real-time worker status in queen
- ✅ Simpler failure detection

---

## 6. 🔒 Better Security

### No Open Ports

**Current:**
- Hive daemon listens on port (e.g., 9000)
- Port must be accessible from queen
- Firewall rules needed
- Attack surface: HTTP endpoints

**Proposed:**
- Hive CLI has no open ports
- Only SSH port needed (already required)
- No additional firewall rules
- Attack surface: SSH only

**Benefits:**
- ✅ Smaller attack surface
- ✅ No HTTP vulnerabilities
- ✅ Leverage SSH security
- ✅ Simpler network setup

### No Authentication Needed

**Current:**
- Hive HTTP endpoints need authentication
- Token management between queen and hive
- Token rotation complexity

**Proposed:**
- SSH handles authentication
- No additional auth layer needed
- SSH key management (already solved)

**Benefits:**
- ✅ No custom auth
- ✅ Leverage SSH keys
- ✅ Simpler security model

---

## 7. 📊 Simpler Monitoring

### No Daemon Health Checks

**Current:**
- Queen must check if hive is running
- Queen must restart crashed hives
- Queen must handle hive failures
- Hive must send heartbeats

**Proposed:**
- No hive to monitor
- CLI either succeeds or fails
- SSH handles connection failures
- Workers send heartbeats directly

**Benefits:**
- ✅ No hive health checks
- ✅ No hive restart logic
- ✅ No hive heartbeat processing
- ✅ Simpler failure modes

### Better Observability

**Current:**
- Hive logs to its own stdout
- Queen logs to its stdout
- Worker logs to its stdout
- Three log streams to correlate

**Proposed:**
- Hive CLI output captured by SSH
- Queen logs everything
- Worker logs to its stdout
- Two log streams (queen + workers)

**Benefits:**
- ✅ Centralized logging in queen
- ✅ Easier correlation
- ✅ Simpler debugging

---

## 8. 🚄 Faster Operations

### No HTTP Overhead

**Current:**
```
Queen → HTTP POST → Hive → Spawn worker
        (network)    (parse)  (execute)
```

**Proposed:**
```
Queen → SSH exec → Hive CLI → Spawn worker
        (network)   (execute)
```

**Benefits:**
- ✅ No HTTP parsing
- ✅ No job-server overhead
- ✅ No SSE streaming
- ✅ Direct execution

### No Startup Delay

**Current:**
- Hive daemon must be started first
- Wait for health check to pass
- Then send request

**Proposed:**
- CLI runs immediately
- No startup delay
- No health check needed

---

## 9. 🧪 Easier Testing

### No Integration Test Complexity

**Current:**
```rust
#[tokio::test]
async fn test_worker_spawn() {
    // Start mock hive daemon
    let hive = start_mock_hive().await;
    
    // Start mock queen daemon
    let queen = start_mock_queen().await;
    
    // Send HTTP request
    let client = JobClient::new(&hive.url());
    client.submit_and_stream(operation, ...).await?;
    
    // Cleanup daemons
}
```

**Proposed:**
```rust
#[tokio::test]
async fn test_worker_spawn() {
    // Just run CLI
    let output = Command::new("rbee-hive")
        .args(&["worker", "spawn", "--model", "test"])
        .output()
        .await?;
    
    assert!(output.status.success());
}
```

**Benefits:**
- ✅ No mock servers
- ✅ No daemon lifecycle
- ✅ Simpler test setup
- ✅ Faster tests

---

## 10. 🔄 Better for Remote Execution

### SSH is Already Required

**Current:**
- SSH needed for: install, capabilities, remote start
- HTTP needed for: worker operations
- Two protocols to manage

**Proposed:**
- SSH needed for: everything
- One protocol to manage

**Benefits:**
- ✅ Consistent protocol
- ✅ No HTTP client needed
- ✅ No port forwarding needed
- ✅ Simpler remote execution

### Easier Debugging

**Current:**
```bash
# Debug hive daemon
ssh hive-host "journalctl -u rbee-hive -f"
```

**Proposed:**
```bash
# Debug hive CLI (output captured by queen)
# Just look at queen logs
```

**Benefits:**
- ✅ Centralized debugging
- ✅ No remote log tailing
- ✅ Easier troubleshooting

---

## 11. 💡 Aligns with Unix Philosophy

### Do One Thing Well

**Current:**
- Hive daemon: HTTP server + worker manager + heartbeat sender + state manager
- Too many responsibilities

**Proposed:**
- Hive CLI: Worker lifecycle ONLY
- Queen: Orchestration ONLY
- Workers: Inference ONLY

**Benefits:**
- ✅ Single responsibility
- ✅ Composable tools
- ✅ Unix-style design

### Stateless Tools

**Current:**
- Hive maintains state
- State can become stale
- State synchronization issues

**Proposed:**
- Hive is stateless
- Reads config, performs action, exits
- No state to manage

**Benefits:**
- ✅ Idempotent operations
- ✅ No state drift
- ✅ Easier to reason about

---

## 12. 🎯 Matches Actual Usage Pattern

### Hive Operations are Infrequent

**Reality:**
- Worker spawn: Once per model/device
- Worker stop: Rarely (only on cleanup)
- Worker list: Occasionally (for debugging)
- Model download: Once per model

**Current:**
- Hive daemon runs 24/7 waiting for rare events
- Wasted resources

**Proposed:**
- Hive CLI runs only when needed
- No wasted resources

**Benefits:**
- ✅ Resource efficiency
- ✅ Matches usage pattern
- ✅ No idle daemon

---

## 13. 🔧 Simpler Code Architecture

### No HTTP Layer Needed

**Current:**
```
bin/20_rbee_hive/
├── src/
│   ├── http/
│   │   ├── jobs.rs      (135 LOC)
│   │   └── mod.rs       (5 LOC)
│   ├── job_router.rs    (267 LOC)
│   ├── heartbeat.rs     (80 LOC)
│   └── main.rs          (219 LOC)
```

**Proposed:**
```
bin/20_rbee_hive/
├── src/
│   ├── commands/
│   │   ├── worker_spawn.rs
│   │   ├── worker_stop.rs
│   │   └── worker_list.rs
│   └── main.rs          (CLI parser)
```

**Benefits:**
- ✅ ~400 LOC removed (HTTP + job_router)
- ✅ No HTTP dependencies
- ✅ No job-server dependency
- ✅ Simpler codebase

---

## 14. 🌐 Better for Heterogeneous Environments

### No Port Conflicts

**Current:**
- Each hive needs unique port
- Port management complexity
- Port conflicts possible

**Proposed:**
- No ports needed
- No port management
- No conflicts

**Benefits:**
- ✅ Simpler configuration
- ✅ No port allocation
- ✅ Works anywhere

---

## 15. 📦 Smaller Binary

### No HTTP Dependencies

**Current Dependencies:**
```toml
axum = "0.8"
tower = "0.5"
tower-http = "0.6"
job-server = { path = "..." }
futures = "0.3"
async-stream = "0.3"
```

**Proposed Dependencies:**
```toml
clap = { version = "4", features = ["derive"] }
anyhow = "1.0"
# That's it!
```

**Benefits:**
- ✅ Smaller binary size
- ✅ Faster compilation
- ✅ Fewer dependencies
- ✅ Simpler dependency tree

---

## Summary of PROS

| Category | Benefit | Impact |
|----------|---------|--------|
| **Architecture** | One less daemon | High |
| **Resources** | 50-100 MB saved per machine | Medium |
| **Complexity** | ~400 LOC removed | High |
| **Security** | Smaller attack surface | Medium |
| **Deployment** | No service management | High |
| **Monitoring** | No daemon health checks | Medium |
| **Testing** | Simpler integration tests | High |
| **Remote Execution** | Single protocol (SSH) | High |
| **State Management** | No distributed state | High |
| **Usage Pattern** | Matches actual usage | High |

**Overall Assessment:** 🟢 STRONG PROS

---

## Key Enablers

This pivot is possible because:

1. ✅ **Workers send heartbeats directly to queen** (no aggregation needed)
2. ✅ **Worker operations are infrequent** (no need for always-running daemon)
3. ✅ **SSH is already required** (no new protocol needed)
4. ✅ **Queen is the source of truth** (no distributed state needed)
5. ✅ **Hive only manages lifecycle** (no complex logic needed)

---

**TEAM-261 Pivot Analysis - PROS**  
**Date:** Oct 23, 2025  
**Verdict:** 🟢 STRONG CASE FOR CLI APPROACH
