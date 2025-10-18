# Component: Cascading Shutdown

**Location:** Multiple components  
**Type:** Lifecycle management protocol  
**Language:** Rust  
**Created by:** TEAM-030  
**Status:** ✅ IMPLEMENTED (with gaps)

## Overview

Cascading shutdown protocol ensures clean termination of the entire rbee ecosystem when any parent component dies. Prevents orphaned processes and ensures VRAM cleanup.

**Critical Rule:** When queen-rbee dies, ALL rbee-hive instances and ALL workers STOP GRACEFULLY.

---

## Shutdown Cascade Flow

```
┌─────────────────────────────────────────────────────────┐
│ Cascading Shutdown Protocol                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User (Ctrl+C)                                          │
│       │                                                 │
│       ▼                                                 │
│  rbee-keeper (SIGTERM)                                  │
│       │                                                 │
│       ▼                                                 │
│  queen-rbee (receives SIGTERM)                          │
│       │                                                 │
│       ├─► SSH: SIGTERM to rbee-hive-1                   │
│       ├─► SSH: SIGTERM to rbee-hive-2                   │
│       └─► SSH: SIGTERM to rbee-hive-N                   │
│                │           │           │                │
│                ▼           ▼           ▼                │
│           rbee-hive   rbee-hive   rbee-hive            │
│                │           │           │                │
│                ├─► HTTP: POST /v1/shutdown worker-1     │
│                ├─► HTTP: POST /v1/shutdown worker-2     │
│                └─► HTTP: POST /v1/shutdown worker-N     │
│                         │           │           │       │
│                         ▼           ▼           ▼       │
│                    worker-1    worker-2    worker-N     │
│                         │           │           │       │
│                         └───────────┴───────────┘       │
│                                     │                   │
│                                     ▼                   │
│                            VRAM Released ✅              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation by Component

### 1. rbee-hive → Workers (TEAM-030)

**Location:** `bin/rbee-hive/src/commands/daemon.rs`

**Implementation:**
```rust
// TEAM-030: Setup graceful shutdown handler
let registry_shutdown = registry.clone();
tokio::spawn(async move {
    tokio::signal::ctrl_c().await.ok();
    tracing::info!("Shutdown signal received - cleaning up workers");
    shutdown_all_workers(registry_shutdown).await;
});

/// Shutdown all workers (TEAM-030: Cascading shutdown)
async fn shutdown_all_workers(registry: Arc<WorkerRegistry>) {
    let workers = registry.list().await;
    tracing::info!("Shutting down {} workers", workers.len());

    for worker in workers {
        tracing::info!("Sending shutdown to worker {}", worker.id);

        // Try to gracefully shutdown worker via HTTP
        if let Err(e) = shutdown_worker(&worker.url).await {
            tracing::warn!("Failed to shutdown worker {} gracefully: {}", worker.id, e);
        }
    }

    // Clear registry
    registry.clear().await;
    tracing::info!("All workers shutdown complete");
}

/// Shutdown a single worker via HTTP
async fn shutdown_worker(worker_url: &str) -> Result<()> {
    let client = reqwest::Client::new();

    // Try POST /v1/shutdown endpoint
    let response = client
        .post(format!("{}/v1/shutdown", worker_url))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await?;

    if response.status().is_success() {
        tracing::info!("Worker at {} acknowledged shutdown", worker_url);
    }

    Ok(())
}
```

**Status:** ✅ Implemented  
**Gaps:**
- ⚠️ Sequential shutdown (slow for many workers)
- ⚠️ No force-kill if HTTP fails (workers can hang)
- ⚠️ No timeout enforcement (can wait forever)

---

### 2. queen-rbee → Hives (TEAM-030)

**Location:** `bin/queen-rbee/src/main.rs` (TEAM-030 scaffold)

**Expected Implementation:**
```rust
// TEAM-030: Add cascading shutdown support
impl QueenRbee {
    async fn shutdown(&self) {
        tracing::info!("Queen shutting down - cascading to hives");
        
        let hives = self.hives.read().await;
        for hive in hives.iter() {
            // SSH to remote hive and send SIGTERM
            if let Err(e) = ssh_shutdown_hive(&hive.ssh_host, &hive.ssh_user).await {
                tracing::warn!("Failed to shutdown hive {}: {}", hive.node_name, e);
            }
        }
        
        tracing::info!("All hives shutdown complete");
    }
}

async fn ssh_shutdown_hive(host: &str, user: &str) -> Result<()> {
    // SSH command: pkill -TERM rbee-hive
    let output = Command::new("ssh")
        .arg(format!("{}@{}", user, host))
        .arg("pkill")
        .arg("-TERM")
        .arg("rbee-hive")
        .output()
        .await?;
    
    Ok(())
}
```

**Status:** 🟡 Scaffolded (TEAM-030 TODO notes)  
**Gaps:**
- ⚠️ Not fully implemented (scaffold only)
- ⚠️ No SSH connection handling
- ⚠️ No timeout enforcement
- ⚠️ Sequential shutdown (slow)

---

### 3. rbee-keeper → queen-rbee

**Location:** `bin/rbee-keeper/src/commands/infer.rs`

**Implementation:**
```rust
// After inference completes
tracing::info!("Shutting down queen...");
queen_process.kill().await?;
```

**Status:** ✅ Implemented  
**Method:** Direct process kill (rbee-keeper spawns queen)

---

## Shutdown Timeouts

### Current Timeouts
- **Worker HTTP shutdown:** 5 seconds
- **Hive → Workers:** No overall timeout (sequential)
- **Queen → Hives:** Not implemented
- **Force kill:** Not implemented

### Recommended Timeouts
```rust
// Per-worker shutdown
const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

// Overall hive shutdown (all workers)
const HIVE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

// Overall queen shutdown (all hives)
const QUEEN_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(60);

// Force kill delay after graceful attempt
const FORCE_KILL_DELAY: Duration = Duration::from_secs(10);
```

---

## Shutdown Modes

### 1. Graceful Shutdown (Current)
**Method:** HTTP POST /v1/shutdown  
**Timeout:** 5s per worker  
**Pros:** Clean VRAM release, proper cleanup  
**Cons:** Can hang if worker unresponsive

### 2. Force Kill (Missing - Critical Gap!)
**Method:** SIGKILL via PID  
**Timeout:** Immediate  
**Pros:** Guaranteed termination  
**Cons:** Potential VRAM leak, dirty shutdown

**Recommended Flow:**
```rust
// 1. Try graceful shutdown
let graceful_result = shutdown_worker_http(&worker.url).await;

// 2. Wait for confirmation or timeout
tokio::time::timeout(
    Duration::from_secs(5),
    wait_for_worker_exit(worker.pid)
).await;

// 3. Force kill if still alive
if is_worker_alive(worker.pid) {
    tracing::warn!("Worker {} unresponsive, force killing", worker.id);
    force_kill_worker(worker.pid)?;
}
```

---

## Parallel vs Sequential Shutdown

### Current: Sequential (Slow)
```rust
for worker in workers {
    shutdown_worker(&worker.url).await?;  // Waits for each
}
// Total time: N workers × 5s = potentially minutes
```

### Recommended: Parallel (Fast)
```rust
let shutdown_tasks: Vec<_> = workers
    .iter()
    .map(|w| shutdown_worker(&w.url))
    .collect();

// Shutdown all workers concurrently
let results = futures::future::join_all(shutdown_tasks).await;

// Total time: max(worker shutdown times) ≈ 5s
```

---

## Integration with HTTP Server

### rbee-hive HTTP Server (TEAM-030)

**Location:** `bin/rbee-hive/src/http/server.rs`

```rust
/// Run server with graceful shutdown
axum::serve(listener, self.router)
    .with_graceful_shutdown(async move {
        let _ = shutdown_rx.recv().await;
        warn!("rbee-hive HTTP server shutting down gracefully");
    })
    .await
```

**Features:**
- ✅ SIGINT/SIGTERM handler
- ✅ Graceful shutdown (completes in-flight requests)
- ✅ Broadcast channel for shutdown signal

**Gaps:**
- ⚠️ No timeout (can wait forever for in-flight requests)
- ⚠️ No force shutdown after timeout

---

## Maturity Assessment

**Status:** 🟡 **PARTIAL IMPLEMENTATION**

### Implemented (TEAM-030)
- ✅ rbee-hive → workers HTTP shutdown
- ✅ SIGTERM/SIGINT handlers
- ✅ Registry cleanup
- ✅ Graceful HTTP server shutdown
- ✅ Worker shutdown endpoint

### Critical Gaps
- ❌ **No force-kill capability** (workers can hang shutdown)
- ❌ **Sequential shutdown** (slow for many workers)
- ❌ **No timeout enforcement** (can wait forever)
- ❌ **queen-rbee → hives** not fully implemented
- ❌ **No PID tracking** (can't force-kill)

### Recommended Improvements

**P0 - Critical:**
1. Add PID tracking to enable force-kill
2. Implement parallel worker shutdown
3. Add shutdown timeout enforcement
4. Complete queen-rbee → hives shutdown

**P1 - High Priority:**
5. Add force-kill after graceful timeout
6. Add shutdown progress reporting
7. Add shutdown metrics (time, success rate)

**P2 - Nice to Have:**
8. Add shutdown hooks for cleanup
9. Add graceful degradation (partial shutdown)
10. Add shutdown audit logging

---

## Testing Scenarios

### Basic Shutdown
```bash
# Start hive
cargo run -p rbee-hive -- daemon &
HIVE_PID=$!

# Spawn workers
curl -X POST http://localhost:8081/v1/workers/spawn ...

# Trigger shutdown
kill -TERM $HIVE_PID

# Verify all workers stopped
ps aux | grep llm-worker-rbee  # Should be empty
```

### Hung Worker Scenario
```bash
# Start hive
cargo run -p rbee-hive -- daemon &

# Spawn worker
curl -X POST http://localhost:8081/v1/workers/spawn ...

# Simulate hung worker (block /v1/shutdown endpoint)
# ... modify worker to ignore shutdown ...

# Trigger shutdown
kill -TERM $HIVE_PID

# Expected: Hive hangs waiting for worker
# Actual: Hive waits 5s, then logs warning, continues
# Desired: Hive force-kills after 10s
```

### Cascade Shutdown (Full Stack)
```bash
# Start queen
cargo run -p queen-rbee &
QUEEN_PID=$!

# Start hives (via queen)
# ... queen spawns hives via SSH ...

# Spawn workers
# ... hives spawn workers ...

# Trigger cascade
kill -TERM $QUEEN_PID

# Verify cascade:
# 1. Queen sends SIGTERM to all hives
# 2. Hives send shutdown to all workers
# 3. Workers exit cleanly
# 4. VRAM released
```

---

## Shutdown Sequence Diagram

```
User                rbee-keeper    queen-rbee     rbee-hive-1    worker-1
 │                       │              │              │             │
 │  Ctrl+C               │              │              │             │
 ├──────────────────────►│              │              │             │
 │                       │  SIGTERM     │              │             │
 │                       ├─────────────►│              │             │
 │                       │              │  SSH SIGTERM │             │
 │                       │              ├─────────────►│             │
 │                       │              │              │ POST /shutdown
 │                       │              │              ├────────────►│
 │                       │              │              │             │
 │                       │              │              │  Cleanup    │
 │                       │              │              │  Release VRAM
 │                       │              │              │◄────────────┤
 │                       │              │              │  200 OK     │
 │                       │              │  SSH exit    │             │
 │                       │              │◄─────────────┤             │
 │                       │  Exit        │              │             │
 │                       │◄─────────────┤              │             │
 │  Done                 │              │              │             │
 │◄──────────────────────┤              │              │             │
```

---

## Configuration

### Environment Variables
```bash
# Shutdown timeouts
RBEE_WORKER_SHUTDOWN_TIMEOUT=5      # Seconds to wait for worker
RBEE_HIVE_SHUTDOWN_TIMEOUT=30       # Seconds to wait for all workers
RBEE_QUEEN_SHUTDOWN_TIMEOUT=60      # Seconds to wait for all hives
RBEE_FORCE_KILL_DELAY=10            # Seconds before force kill
```

### Config File (Future)
```toml
[shutdown]
worker_timeout = 5
hive_timeout = 30
queen_timeout = 60
force_kill_delay = 10
parallel_shutdown = true
max_concurrent_shutdowns = 10
```

---

## Related Components

- **Worker Lifecycle** - Manages worker processes, needs PID tracking for force-kill
- **Worker Registry** - Tracks workers, cleared on shutdown
- **HTTP Server** - Graceful shutdown support
- **SSH Module** - Used for queen → hive shutdown (not fully implemented)

---

## Comparison: Current vs Ideal

| Feature | Current (TEAM-030) | Ideal |
|---------|-------------------|-------|
| **Hive → Workers** | ✅ HTTP shutdown | ✅ HTTP + force-kill |
| **Queen → Hives** | 🟡 Scaffold only | ✅ SSH SIGTERM |
| **Shutdown Mode** | Sequential | Parallel |
| **Timeout** | 5s per worker | 30s total + force |
| **Force Kill** | ❌ No | ✅ After timeout |
| **PID Tracking** | ❌ No | ✅ Yes |
| **Progress** | Logs only | Metrics + logs |

---

**Created by:** TEAM-030  
**Last Updated:** 2025-10-18 (TEAM-096 documentation)  
**Maturity:** 🟡 Partial (rbee-hive complete, queen-rbee scaffolded)  
**Critical Gap:** No force-kill capability (requires PID tracking)
