# queen-rbee

**Status:** 🚧 MIGRATION TARGET (TEAM-135)  
**Purpose:** The orchestrator brain - scheduling, routing, and lifecycle management  
**Binary Name:** `queen-rbee`  
**Source:** `bin/old.queen-rbee/` (~2,015 LOC)

---

## 🎯 CORE PRINCIPLE

**queen-rbee is THE BRAIN of the entire rbee infrastructure**

```
rbee-keeper (UI) → queen-rbee (BRAIN) → rbee-hive (execution) → llm-worker (inference)
```

**This daemon is THE ORCHESTRATOR that:**
- ✅ Makes ALL intelligent scheduling decisions
- ✅ Manages ALL hive lifecycle (start/stop local & remote hives)
- ✅ Tracks ALL workers across ALL hives
- ✅ Routes ALL inference requests to appropriate workers
- ✅ Provides HTTP API for rbee-keeper CLI
- ✅ Uses SSH to manage remote hives

**This daemon does NOT:**
- ❌ Have a CLI interface (daemon only, no CLI commands)
- ❌ Execute inference (that's llm-worker-rbee)
- ❌ Spawn workers directly (that's rbee-hive)
- ❌ Provide user-facing commands (that's rbee-keeper)

---

## 🧠 CORE RESPONSIBILITIES

### 1. **Orchestrator Intelligence (THE BRAIN)**

**Makes ALL intelligent decisions:**

**Admission Control:**
- Accept/reject inference requests based on capacity
- Queue management (persistent job queue)
- Priority handling (interactive vs batch)

**Scheduling & Load Balancing:**
- Worker selection algorithms (least loaded, round-robin, etc.)
- Load balancing across multiple hives
- GPU VRAM optimization
- Model affinity (prefer workers with model already loaded)

**Eviction Policies:**
- Worker eviction (idle timeout, memory pressure)
- Model eviction (LRU, usage-based)

**Retry & Timeout Logic:**
- Exponential backoff for failed requests
- Timeout management per request
- Cancellation handling

**Future: Rhai Scheduler Engine (M2):**
- User-programmable scheduling logic
- 40+ helper functions (workers.least_loaded(), gpu_vram_free(), etc.)
- Platform mode (immutable) vs Home/Lab mode (customizable)

---

### 2. **State Management (Dual Registry System)**

**A. Hive Registry (SQLite - PERSISTENT):**

Tracks all rbee-hive instances (local & remote):

```sql
CREATE TABLE hives (
    hive_id TEXT PRIMARY KEY,
    hostname TEXT NOT NULL,
    port INTEGER NOT NULL,
    mode TEXT NOT NULL,  -- 'local' or 'network'
    ssh_host TEXT,       -- For remote hives
    ssh_port INTEGER,
    ssh_user TEXT,
    ssh_key_path TEXT,
    capabilities JSON,   -- GPUs, backends, models
    last_seen TIMESTAMP,
    status TEXT          -- 'online', 'offline', 'error'
);
```

**Operations:**
- Register hive (local or remote)
- Update hive status (health monitoring)
- Query hives by capability (GPU type, backend, etc.)
- Remove hive

**B. Worker Registry (RAM - EPHEMERAL):**

Tracks all workers across all hives:

```rust
struct WorkerRegistry {
    workers: HashMap<WorkerId, WorkerInfo>,
}

struct WorkerInfo {
    worker_id: String,
    hive_id: String,      // Which hive owns this worker
    model_ref: String,
    backend: String,
    device: u32,
    state: WorkerState,   // Loading, Idle, Busy
    url: String,
    slots_total: u32,
    slots_available: u32,
    vram_bytes: Option<u64>,
}
```

**Operations:**
- Register worker (called by rbee-hive when worker is ready)
- Update worker state (idle → busy → idle)
- Query workers by model/backend/hive
- Remove worker (shutdown or crashed)

**C. Job Queue (SQLite - PERSISTENT):**

Tracks all inference jobs:

```sql
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    tenant_id TEXT,
    model_ref TEXT,
    priority TEXT,        -- 'interactive', 'batch'
    state TEXT,           -- 'queued', 'running', 'complete', 'failed'
    worker_id TEXT,
    submitted_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error TEXT
);
```

---

### 3. **Hive Lifecycle Management**

**queen-rbee manages ALL hive lifecycle (rbee-keeper does NOT)**

**A. Local Hives (Same Machine):**

```rust
async fn start_local_hive(config: HiveConfig) -> Result<HiveId> {
    // 1. Spawn rbee-hive daemon as child process
    let child = Command::new("rbee-hive")
        .arg("daemon")  // rbee-hive ONLY runs in daemon mode
        .arg("--port").arg(config.port)
        .spawn()?;
    
    // 2. Store PID for lifecycle management
    // 3. Wait for health check (GET /health)
    // 4. Register in SQLite hive registry
    // 5. Start health monitoring task
}

async fn stop_local_hive(hive_id: &str) {
    // 1. Send SIGTERM to hive process
    // 2. Wait for graceful shutdown (30s)
    // 3. Force kill if timeout
    // 4. Update SQLite status
}
```

**B. Network Hives (Remote Machines via SSH):**

```rust
async fn start_network_hive(config: HiveConfig) -> Result<HiveId> {
    // 1. SSH to remote machine
    let ssh = SshClient::connect(&config.ssh_host, &config.ssh_key)?;
    
    // 2. Check if rbee-hive is already running
    // 3. If not running: spawn rbee-hive daemon via SSH
    // 4. Wait for health check (HTTP to remote hive)
    // 5. Register in SQLite hive registry
    // 6. Start health monitoring task
}

async fn stop_network_hive(hive_id: &str) {
    // 1. SSH to remote machine
    // 2. Send shutdown command to rbee-hive
    // 3. Wait for graceful shutdown
    // 4. Update SQLite status
}
```

**Key Point:** queen-rbee uses SSH to manage remote hives. rbee-keeper does NOT use SSH.

---

### 4. **Request Routing & Orchestration**

**Inference Request Flow:**

```rust
async fn handle_inference_request(req: InferenceRequest) -> Result<Response> {
    // 1. ADMISSION CONTROL
    if !should_admit(&req) {
        return Err(Error::QueueFull);
    }
    
    // 2. SCHEDULING
    let worker = select_worker(&req.model_ref, &req.backend)?;
    
    // 3. WORKER PROVISIONING (if needed)
    if worker.is_none() {
        // Find hive with capacity
        let hive = select_hive_for_model(&req.model_ref)?;
        
        // Ensure hive is running
        ensure_hive_running(&hive.hive_id).await?;
        
        // Request hive to spawn worker
        let worker = request_worker_spawn(&hive, &req.model_ref).await?;
        
        // Wait for worker ready callback
        wait_for_worker_ready(&worker.worker_id, Duration::from_secs(300)).await?;
    }
    
    // 4. ROUTE REQUEST
    let response = route_to_worker(&worker, &req).await?;
    
    // 5. RELAY SSE STREAM
    relay_sse_stream(response)
}
```

**Key Decisions Made by queen-rbee:**
- Which hive to use for a model
- Which worker to route to
- When to spawn new workers
- When to evict idle workers
- How to handle failures (retry, backoff)

---

### 5. **Model Provisioning Coordination**

**queen-rbee coordinates model downloads (does NOT download directly):**

```rust
async fn ensure_model_available(hive_id: &str, model_ref: &str) -> Result<()> {
    // 1. Check if hive has model
    let has_model = check_hive_has_model(hive_id, model_ref).await?;
    
    if !has_model {
        // 2. Request hive to download model
        let download_id = request_model_download(hive_id, model_ref).await?;
        
        // 3. Poll download progress
        wait_for_download_complete(hive_id, download_id).await?;
    }
    
    Ok(())
}
```

**Key Point:** queen-rbee coordinates, rbee-hive executes the actual download.

---

### 6. **Health Monitoring**

**Continuous health monitoring of all hives:**

```rust
async fn health_monitor_loop() {
    loop {
        for hive in hive_registry.list_all() {
            match check_hive_health(&hive).await {
                Ok(_) => hive_registry.mark_online(&hive.hive_id),
                Err(_) => {
                    hive_registry.mark_offline(&hive.hive_id);
                    // Evict all workers from this hive
                    worker_registry.remove_by_hive(&hive.hive_id);
                }
            }
        }
        
        tokio::time::sleep(Duration::from_secs(30)).await;
    }
}
```

---

### 7. **HTTP API (for rbee-keeper)**

**Provides HTTP API that rbee-keeper uses:**

```
GET  /health                          # Health check
POST /v2/tasks                        # Submit inference request
GET  /v2/tasks/:id                    # Query job status

GET  /v2/registry/beehives/list       # List registered hives
POST /v2/registry/beehives/add        # Register new hive
POST /v2/registry/beehives/remove     # Remove hive

GET  /v2/workers/list                 # List all workers
GET  /v2/workers/health?node=X        # Worker health
POST /v2/workers/spawn                # Request worker spawn
POST /v2/workers/shutdown             # Shutdown worker

GET  /v2/models/list?node=X           # List models on hive
POST /v2/models/download              # Request model download

GET  /v2/logs?node=X&follow=true      # Stream logs from hive
```

**Key Point:** rbee-keeper submits requests to these endpoints. queen-rbee orchestrates everything.

---

## 🏗️ ARCHITECTURE

### Binary Structure

```
bin/10_queen_rbee/
├── src/
│   ├── main.rs          (~283 LOC - entry point, shutdown handler)
│   ├── http_server.rs   (HTTP routes & API - IN BINARY)
│   └── lib.rs           (Re-exports from crates)
└── Cargo.toml

Dependencies:
├── queen-rbee-hive-registry   (Hive registry - SQLite)
├── queen-rbee-worker-registry (Worker registry - RAM)
├── queen-rbee-ssh-client      (SSH client for remote hives)
├── queen-rbee-hive-lifecycle  (Hive lifecycle management)
├── queen-rbee-preflight       (Preflight checks)
├── queen-rbee-health          (Health check endpoint)
├── queen-rbee-hive-catalog    (Persistent hive storage)
└── queen-rbee-scheduler       (Device selection & scheduling)
```

**IMPORTANT:** HTTP server entry point is implemented DIRECTLY in the binary,
not as a separate crate. This keeps the HTTP routing logic tightly coupled to the binary.

### Orchestration Flow

```
┌─────────────────────────────────────────────────────┐
│ rbee-keeper: rbee infer --node gpu-0 --model llama  │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP POST /v2/tasks
                   ▼
┌─────────────────────────────────────────────────────┐
│ queen-rbee (THE BRAIN)                              │
│  1. Admission control (accept/reject)               │
│  2. Check worker registry for available worker      │
│  3. If no worker:                                   │
│     a. Select hive for model (scheduling)           │
│     b. ensure_hive_running(hive_id)                 │
│        - Local: spawn rbee-hive daemon              │
│        - Remote: SSH + spawn rbee-hive daemon       │
│     c. Request hive to spawn worker                 │
│     d. Wait for worker ready callback               │
│  4. Route request to worker                         │
│  5. Relay SSE stream back to rbee-keeper            │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP or SSH
                   ▼
┌─────────────────────────────────────────────────────┐
│ rbee-hive (EXECUTION)                               │
│  - Receives worker spawn request from queen         │
│  - Spawns llm-worker-rbee process                   │
│  - Worker calls back to queen /v1/workers/ready     │
│  - Routes inference requests to worker              │
└──────────────────┬──────────────────────────────────┘
                   │ spawns
                   ▼
┌─────────────────────────────────────────────────────┐
│ llm-worker-rbee (INFERENCE)                         │
│  - Loads model, generates tokens                    │
└─────────────────────────────────────────────────────┘
```

---

## 🚫 WHAT THIS DAEMON DOES NOT DO

### ❌ NO CLI Interface

queen-rbee is a daemon ONLY. It does NOT have CLI commands.

```bash
# ❌ WRONG (old architecture violation)
queen-rbee hive start gpu-0  # NO CLI!

# ✅ CORRECT
queen-rbee --port 8080 --database /path/to/db.sqlite  # Daemon only
```

**Why:** rbee-keeper is the CLI. queen-rbee is the daemon.

### ❌ NO Direct Inference Execution

queen-rbee routes requests. llm-worker-rbee executes inference.

### ❌ NO Direct Worker Spawning

queen-rbee requests worker spawns. rbee-hive spawns workers.

### ❌ NO Direct Model Downloads

queen-rbee coordinates downloads. rbee-hive downloads models.

---

## 🔗 DEPENDENCIES

### Internal Crates

```toml
[dependencies]
queen-rbee-registry = { path = "../15_queen_rbee_crates/registry" }
queen-rbee-http-server = { path = "../15_queen_rbee_crates/http-server" }
queen-rbee-orchestrator = { path = "../15_queen_rbee_crates/orchestrator" }
queen-rbee-remote = { path = "../15_queen_rbee_crates/remote" }
queen-rbee-lifecycle = { path = "../15_queen_rbee_crates/lifecycle" }
```

### Shared Crates

```toml
[dependencies]
auth-min = { path = "../../libs/auth-min" }
audit-logging = { path = "../../libs/audit-logging" }
input-validation = { path = "../../libs/input-validation" }
deadline-propagation = { path = "../../libs/deadline-propagation" }
secrets-management = { path = "../../libs/secrets-management" }
```

### External Crates

```toml
[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
sqlx = { version = "0.7", features = ["sqlite", "runtime-tokio"] }
reqwest = "0.11"
ssh2 = "0.9"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"
```

---

## 📊 MIGRATION STATUS

**Source:** `bin/old.queen-rbee/` (2,015 LOC)  
**Target:** `bin/10_queen_rbee/` (~283 LOC binary + 1,732 LOC in crates)

**Crate Decomposition:**
- `registry` - 353 LOC (Dual registry system)
- `http-server` - 897 LOC (HTTP routes, types, middleware)
- `orchestrator` - 610 LOC (Inference orchestration, scheduling)
- `remote` - 182 LOC (SSH client, preflight checks)
- `lifecycle` - ~800 LOC (NEW - Hive lifecycle management)
- **Binary** - 283 LOC (main.rs entry point, shutdown handler)

**Architecture Fixes:**
- ❌ Remove CLI interface (daemon only)
- ✅ Add hive lifecycle management (~800 LOC NEW)
- ✅ Add scheduling engine (basic version, Rhai in M2)
- ✅ Add job queue persistence
- ✅ Add health monitoring

---

## ✅ ACCEPTANCE CRITERIA

### Compilation

```bash
cd bin/10_queen_rbee
cargo check
cargo clippy -- -D warnings
cargo build --release
```

### Functionality

- [ ] Starts as daemon (no CLI commands)
- [ ] Manages local hive lifecycle (spawn/stop)
- [ ] Manages remote hive lifecycle (SSH spawn/stop)
- [ ] Tracks all workers across all hives
- [ ] Routes inference requests to appropriate workers
- [ ] Spawns workers on-demand when needed
- [ ] Health monitors all hives
- [ ] Provides HTTP API for rbee-keeper

### Architecture Compliance

- [ ] NO CLI interface (daemon only)
- [ ] Uses SSH to manage remote hives
- [ ] Makes ALL scheduling decisions
- [ ] Manages ALL hive lifecycle
- [ ] Tracks ALL workers
- [ ] rbee-keeper only submits requests (no orchestration in keeper)

---

## 📚 REFERENCES

### Planning Documents

- `.plan/.archive-130BC-134/TEAM_130C_QUEEN_RBEE_COMPLETE_RESPONSIBILITIES.md`
  - Lines 1-31: Critical findings (massive gaps identified)
  - Lines 34-117: Complete responsibilities
  - Lines 120-200: Hive lifecycle management

- `.plan/.archive-130BC-134/TEAM_132_queen-rbee_INVESTIGATION_REPORT.md`
  - Complete decomposition analysis
  - Crate structure and LOC breakdown

### Source Code

- `bin/old.queen-rbee/` - Original implementation
- `bin/15_queen_rbee_crates/` - Decomposed crates

---

## 🚨 CRITICAL: SEPARATION OF CONCERNS

**rbee-keeper (UI) responsibilities:**
- ✅ Parse CLI arguments
- ✅ Start/stop queen-rbee daemon
- ✅ Submit requests to queen-rbee HTTP API
- ✅ Display results to user
- ❌ NO scheduling
- ❌ NO SSH
- ❌ NO hive management
- ❌ NO worker management

**queen-rbee (BRAIN) responsibilities:**
- ✅ ALL scheduling decisions
- ✅ ALL hive lifecycle (local + remote via SSH)
- ✅ ALL worker tracking
- ✅ ALL request routing
- ✅ ALL orchestration logic
- ❌ NO CLI interface
- ❌ NO direct inference execution

**rbee-hive (EXECUTION) responsibilities:**
- ✅ Spawn workers
- ✅ Download models
- ✅ Route requests to workers
- ✅ Provide HTTP API to queen
- ❌ NO scheduling
- ❌ NO CLI interface
- ❌ NO cross-hive coordination

---

**Migration Status:** 🚧 NOT STARTED  
**Priority:** CRITICAL (blocking entire system)  
**Estimated Effort:** 2-3 weeks (massive missing functionality)

**Next Steps:**
1. Migrate existing crates in `15_queen_rbee_crates/`
2. Implement hive lifecycle management (~800 LOC NEW)
3. Implement basic scheduling engine
4. Implement job queue persistence
5. Implement health monitoring
6. Create `src/main.rs` entry point
7. Test full orchestration flow
