# Component: queen-rbee (Orchestrator Daemon)

**Location:** `bin/queen-rbee/`  
**Type:** HTTP daemon / Orchestrator  
**Language:** Rust  
**Purpose:** Central orchestrator for distributed inference

## Overview

`queen-rbee` is the central orchestrator daemon that manages rbee-hive instances (worker pool managers) across local and remote machines. It receives inference requests, routes them to appropriate hives, and coordinates model provisioning.

## Responsibilities

### 1. Hive Registry Management (SQLite)

**Persistent Storage:**
- Stores registered rbee-hive instances
- Tracks hive capabilities (GPUs, models)
- Maintains connection info (host, port, SSH keys)

**Schema:**
```sql
CREATE TABLE hives (
    id TEXT PRIMARY KEY,
    hostname TEXT NOT NULL,
    port INTEGER NOT NULL,
    mode TEXT NOT NULL,  -- 'local' or 'network'
    ssh_key_path TEXT,   -- For network mode
    capabilities JSON,   -- GPU info, available models
    last_seen TIMESTAMP,
    status TEXT          -- 'online', 'offline', 'error'
);
```

### 2. Worker Registry (RAM)

**Ephemeral Storage:**
- Tracks active workers across all hives
- Maps worker_id → hive_id → worker_info
- Used for request routing

**Structure:**
```rust
struct WorkerRegistry {
    workers: HashMap<WorkerId, WorkerInfo>,
}

struct WorkerInfo {
    worker_id: String,
    hive_id: String,      // Which hive owns this worker
    model_ref: String,
    state: WorkerState,
    url: String,          // Worker's HTTP endpoint
}
```

### 3. Hive Lifecycle Management

**Local Hives:**
```rust
// Start local hive
async fn start_local_hive(config: HiveConfig) -> Result<HiveId> {
    // 1. Spawn rbee-hive process
    let child = Command::new("rbee-hive")
        .arg("--port").arg(config.port)
        .spawn()?;
    
    // 2. Store PID for lifecycle management
    // 3. Wait for health check
    // 4. Register in SQLite
}
```

**Network Hives (SSH):**
```rust
// Start remote hive via SSH
async fn start_network_hive(config: HiveConfig) -> Result<HiveId> {
    // 1. SSH to remote machine
    let ssh = SshSession::connect(&config.hostname, &config.ssh_key)?;
    
    // 2. Upload rbee-hive binary if needed
    // 3. Start rbee-hive via SSH
    ssh.exec("rbee-hive --port 8080")?;
    
    // 4. Verify health check over network
    // 5. Register in SQLite
}
```

**Shutdown:**
```rust
async fn shutdown_hive(hive_id: &str) {
    // 1. Send graceful shutdown to hive
    // 2. Wait for workers to drain
    // 3. Force kill if timeout
    // 4. Update SQLite status
}
```

### 4. Request Routing

**Inference Request Flow:**
```
User Request
    │
    ▼
queen-rbee receives POST /v1/infer
    │
    ▼
Find worker with requested model
    │
    ├─ Worker exists? → Route to worker
    │
    └─ No worker? → Spawn worker
           │
           ├─ Find hive with model
           │
           ├─ No hive has model? → Download model
           │
           └─ Spawn worker on hive
```

### 5. Model Provisioning Coordination

**Centralized Model Catalog:**
- Queen tracks which hives have which models
- Coordinates downloads across hives
- Avoids duplicate downloads

**Flow:**
```rust
async fn ensure_model_available(model_ref: &str, hive_id: &str) -> Result<()> {
    // 1. Check if hive has model
    if hive_has_model(hive_id, model_ref).await? {
        return Ok(());
    }
    
    // 2. Trigger download on hive
    let hive_url = get_hive_url(hive_id).await?;
    http_client.post(format!("{}/v1/models/download", hive_url))
        .json(&DownloadRequest { model_ref })
        .send().await?;
    
    // 3. Wait for download completion
    // 4. Update queen's model catalog
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ queen-rbee (Orchestrator)                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ HTTP API                                         │  │
│  │  - POST /v1/infer                                │  │
│  │  - GET /v1/hives/list                            │  │
│  │  - POST /v1/hives/register                       │  │
│  │  - GET /v1/workers/list (all hives)              │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Hive Registry (SQLite)                           │  │
│  │  - Persistent hive registration                  │  │
│  │  - Hive capabilities                             │  │
│  │  - Connection info (SSH keys)                    │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Worker Registry (RAM)                            │  │
│  │  - Ephemeral worker tracking                     │  │
│  │  - Maps worker → hive                            │  │
│  │  - Request routing                               │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Hive Lifecycle Manager                           │  │
│  │  - Start/stop local hives                        │  │
│  │  - SSH to remote hives                           │  │
│  │  - Health monitoring                             │  │
│  └──────────────────────────────────────────────────┘  │
│                        │                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Request Router                                   │  │
│  │  - Find worker for model                         │  │
│  │  - Spawn worker if needed                        │  │
│  │  - Load balancing (future)                       │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │ rbee-hive│        │ rbee-hive│        │ rbee-hive│
   │ (local)  │        │ (network)│        │ (network)│
   └──────────┘        └──────────┘        └──────────┘
```

## Local vs Network Mode

### Local Mode
**Characteristics:**
- Hive runs on same machine as queen
- Direct process spawning
- No SSH required
- Shared filesystem (model cache)

**Configuration:**
```toml
[[hive]]
mode = "local"
port = 8080
```

**Implementation:**
```rust
if hive.mode == "local" {
    // Spawn rbee-hive as child process
    let child = Command::new("rbee-hive").spawn()?;
    store_pid(child.id());
}
```

### Network Mode
**Characteristics:**
- Hive runs on remote machine
- SSH for management
- Separate filesystem (separate model cache)
- Network latency

**Configuration:**
```toml
[[hive]]
mode = "network"
hostname = "192.168.1.100"
port = 8080
ssh_key = "/home/user/.ssh/id_rsa"
```

**Implementation:**
```rust
if hive.mode == "network" {
    // SSH to remote machine
    let ssh = SshSession::connect(&hive.hostname, &hive.ssh_key)?;
    
    // Start rbee-hive remotely
    ssh.exec("rbee-hive --port 8080")?;
    
    // Health check over network
    let url = format!("http://{}:{}/v1/health", hive.hostname, hive.port);
    reqwest::get(&url).await?;
}
```

### Mode Detection

**Automatic Detection:**
```rust
fn detect_hive_mode(hostname: &str) -> HiveMode {
    if hostname == "localhost" || hostname == "127.0.0.1" {
        HiveMode::Local
    } else {
        HiveMode::Network
    }
}
```

**Explicit Configuration:**
```rust
// User can override auto-detection
[hive.local]
mode = "local"  # Force local even if hostname is IP

[hive.remote]
mode = "network"  # Force network even if hostname is localhost
```

## Key Files

- `src/main.rs` - Daemon entry point
- `src/http/routes.rs` - HTTP API endpoints
- `src/registry/hive_registry.rs` - SQLite hive registry
- `src/registry/worker_registry.rs` - RAM worker registry
- `src/lifecycle/hive_manager.rs` - Hive lifecycle (start/stop)
- `src/lifecycle/ssh.rs` - SSH client for remote hives
- `src/router.rs` - Request routing logic
- `src/scheduler.rs` - Load balancing (future)

## Current Status

**Implemented:**
- ✅ Basic HTTP API structure
- ✅ Inference request handling
- ✅ Worker registry (RAM)

**Missing:**
- ❌ Hive registry (SQLite)
- ❌ Hive lifecycle management
- ❌ SSH support for network mode
- ❌ Mode detection (local vs network)
- ❌ Model provisioning coordination
- ❌ Request routing logic
- ❌ Health monitoring of hives
- ❌ Graceful shutdown coordination

## Related Components

- **rbee-keeper** - CLI that starts/stops queen-rbee
- **rbee-hive** - Worker pool manager (managed by queen)
- **llm-worker-rbee** - Worker processes (managed by hive)

## Future Enhancements

1. **Scheduler** - Intelligent load balancing across hives
2. **Auto-scaling** - Spawn/shutdown hives based on load
3. **Multi-region** - Geographic distribution
4. **Failover** - Automatic hive failover
5. **Metrics** - Centralized metrics collection

---

**Created by:** Multiple teams  
**Last Updated:** TEAM-096 | 2025-10-18  
**Status:** 🔴 INCOMPLETE - Core structure exists, hive management missing
