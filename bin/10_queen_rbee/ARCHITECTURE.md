# queen-rbee: The Orchestrator Brain 🧠👑

**Date:** Oct 29, 2025  
**Status:** ✅ ACTIVE  
**Binary:** `queen-rbee`  
**Port:** 7833 (default)

---

## 🎯 Core Purpose

**queen-rbee is THE BRAIN of the entire rbee infrastructure.**

```
rbee-keeper (CLI) → queen-rbee (BRAIN) → rbee-hive (worker lifecycle only)
                                      ↘ llm-worker (DIRECT for inference)
```

**Flow:**
1. Queen sends job to hive: "spawn worker with model X"
2. Hive spawns worker, worker sends heartbeat to queen
3. Queen routes inference DIRECTLY to worker (hive never sees inference requests)

The Queen makes ALL intelligent decisions about:
- **Scheduling** - Which worker handles which inference request
- **Job Coordination** - Sending jobs to hives (worker spawn, model download)
- **Worker Tracking** - Monitoring all workers via heartbeats
- **Hive Tracking** - Monitoring all hives via heartbeats
- **Direct Inference Routing** - Routes ALL infer requests directly to workers (hive NEVER involved)
- **OpenAI API Compatibility** - Provides OpenAI-compatible endpoints

---

## 🧠 Core Responsibilities

### 1. **Orchestration & Scheduling**

**Current (M0/M1):**
- Worker selection (least loaded, round-robin)
- Admission control (accept/reject requests)
- Load balancing across hives
- **Direct worker routing** - Infer requests go directly to worker (not through hive)

**Future (M2 - Rhai Scheduler):**
- User-programmable scheduling logic
- 40+ helper functions (`workers.least_loaded()`, `gpu_vram_free()`, etc.)
- Two modes:
  - **Platform Mode:** Immutable scheduler for multi-tenant marketplace
  - **Home/Lab Mode:** Custom Rhai scripts or YAML configs

See: `/home/vince/Projects/llama-orch/.business/stakeholders/RHAI_PROGRAMMABLE_SCHEDULER.md`

---

### 2. **Heartbeat Aggregation (Event-Driven)**

**The Queen is the CENTRAL HEARTBEAT HUB.**

All components send heartbeats TO the Queen:
- **Workers** → Queen: `POST /v1/worker-heartbeat` (every 30s)
- **Hives** → Queen: `POST /v1/hive-heartbeat` (every 30s)
- **Queen** → Clients: SSE stream (every 2.5s)

**CRITICAL:** Hives do NOT aggregate worker heartbeats. Workers send directly to Queen.

**Architecture (TEAM-288):**
```
Worker → POST /v1/worker-heartbeat → Queen → Broadcast Channel → SSE → Client
Hive   → POST /v1/hive-heartbeat   → Queen → Broadcast Channel → SSE → Client
                
Queen Timer → (every 2.5s) ────────────────────────────────────────────→ Client
```

**Event Types:**
```json
// Queen's own heartbeat (every 2.5 seconds)
{
  "type": "queen",
  "workers_online": 2,
  "workers_available": 1,
  "hives_online": 1,
  "hives_available": 1,
  "worker_ids": ["worker-1", "worker-2"],
  "hive_ids": ["gpu-0"],
  "timestamp": "2025-10-29T12:00:00Z"
}

// Worker heartbeat (real-time forwarding)
{
  "type": "worker",
  "worker_id": "worker-1",
  "status": "Ready",
  "timestamp": "2025-10-29T12:00:01Z"
}

// Hive heartbeat (real-time forwarding)
{
  "type": "hive",
  "hive_id": "gpu-0",
  "status": "Online",
  "timestamp": "2025-10-29T12:00:02Z"
}
```

**SSE Endpoint:**
```
GET /v1/heartbeats/stream
Accept: text/event-stream
```

See: `TEAM_288_EVENT_DRIVEN_HEARTBEAT.md`

---

### 3. **Job Coordination (Internal Only)**

**CRITICAL CORRECTION:** Queen does NOT expose worker/model operations through its job API.

**Queen's public job API:**
- `Status` - Query registries
- `Infer` - Schedule and route to workers

**Worker/Model management:**
- **Manual:** rbee-keeper talks directly to hive's job server
- **Automatic:** Queen internally sends jobs to hive when needed for inference

#### Queen's internal worker provisioning (for inference):

```rust
// When queen needs a worker for inference (INTERNAL operation)
async fn ensure_worker_available(model: &str) -> Result<WorkerId> {
    // 1. Check worker registry
    if let Some(worker) = find_available_worker(model) {
        return Ok(worker.id);
    }
    
    // 2. Select hive for model
    let hive = select_hive_for_model(model)?;
    
    // 3. Send job to HIVE's job server (internal, not exposed)
    let job_client = JobClient::new(&hive.url);
    job_client.submit(Operation::WorkerSpawn {
        hive_id: hive.id,
        model: model.to_string(),
        worker: "cpu",
        device: 0,
    }).await?;
    
    // 4. Wait for worker heartbeat
    wait_for_worker_heartbeat(model).await?;
}
```

**This is NOT exposed through queen's job API. It's internal orchestration.**

---

### 4. **Worker Registry (RAM - Heartbeat-Based)**

**The Queen tracks ALL workers via heartbeats.**

**Source:** `bin/15_queen_rbee_crates/worker-registry/`

```rust
use queen_rbee_worker_registry::WorkerRegistry;
use worker_contract::WorkerHeartbeat;

let registry = WorkerRegistry::new();

// Worker sends heartbeat directly to queen
registry.update_worker(heartbeat);

// Query workers for scheduling
let online_workers = registry.list_online_workers();
let available = registry.list_available_workers();
```

**Worker Registration Flow:**
1. Queen sends job to hive: `Operation::WorkerSpawn`
2. Hive spawns `llm-worker` process with `--queen-url`
3. Worker starts HTTP server
4. Worker calls back: `POST http://{queen_url}/v1/workers/ready`
5. Queen registers worker in RAM registry (from heartbeat)
6. Worker sends heartbeats: `POST http://{queen_url}/v1/worker-heartbeat` (every 30s)

**CRITICAL:** Registry is built from heartbeats, NOT SQLite.

---

### 5. **Hive Registry (RAM - Heartbeat-Based)**

**CRITICAL CORRECTION:** NO SQLite. Hive registry is built from heartbeats.

**Source:** `bin/15_queen_rbee_crates/hive-registry/`

```rust
use queen_rbee_hive_registry::HiveRegistry;
use hive_contract::HiveHeartbeat;

let registry = HiveRegistry::new();

// Hive sends heartbeat to queen
registry.update_hive(heartbeat);

// Query hives for scheduling
let online_hives = registry.list_online_hives();
let available = registry.list_available_hives();
let count = registry.count_online();

// Cleanup stale entries
let removed = registry.cleanup_stale();
```

**Hive Configuration:** Managed by rbee-keeper (not queen).

**Registry:** Built from heartbeats, stored in RAM, ephemeral.

---

### 6. **Request Routing & Orchestration**

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
        
        // Send job to hive to spawn worker
        let job_id = send_worker_spawn_job(&hive, &req.model_ref).await?;
        
        // Wait for worker ready callback
        wait_for_worker_ready(&worker_id, Duration::from_secs(300)).await?;
    }
    
    // 4. ROUTE REQUEST DIRECTLY TO WORKER (bypassing hive)
    let response = http_client.post(&worker.url)
        .json(&req)
        .send().await?;
    
    // 5. RELAY SSE STREAM
    relay_sse_stream(response)
}
```

**CRITICAL:** Infer requests go DIRECTLY to worker, not through hive.

---

### 7. **Model Provisioning Coordination**

**Queen coordinates model downloads via jobs:**

```rust
async fn ensure_model_available(hive_id: &str, model_ref: &str) -> Result<()> {
    // 1. Check if hive has model (query hive)
    let has_model = check_hive_has_model(hive_id, model_ref).await?;
    
    if !has_model {
        // 2. Send job to hive to download model
        let operation = Operation::ModelDownload {
            model: model_ref.to_string(),
        };
        let job_id = submit_job_to_hive(hive_id, operation).await?;
        
        // 3. Wait for download complete
        wait_for_job_complete(job_id).await?;
    }
    
    Ok(())
}
```

**Key Point:** Queen coordinates via jobs, hive executes the actual download.

---

### 8. **OpenAI API Compatibility**

**Queen provides OpenAI-compatible endpoints.**

**Source:** `bin/15_queen_rbee_crates/rbee-openai-adapter/`

**Endpoints:**
```
POST /openai/v1/chat/completions      # OpenAI-compatible chat endpoint
GET  /openai/v1/models                # List models (OpenAI format)
GET  /openai/v1/models/{model}        # Get model details
```

**Usage:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7833/openai",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="llama-3-8b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
```

**Translation:**
- OpenAI request → `Operation::Infer`
- Submit to queen's job router
- Transform response to OpenAI format

---

## 🚨 CRITICAL: Queen URL Configuration

### The Bug: Localhost vs Remote

**PROBLEM:** When rbee-keeper starts a remote hive, the hive and workers need to know WHERE to send heartbeats.

❌ **WRONG (causes bug):**
```bash
# Remote hive spawned with localhost queen URL
ssh remote-machine "rbee-hive --queen-url http://localhost:7833"
# ❌ Workers send heartbeat to localhost (their own machine!)
# ❌ Queen never receives heartbeat
# ❌ Queen thinks worker is offline
```

✅ **CORRECT:**
```bash
# Remote hive spawned with Queen's public IP
ssh remote-machine "rbee-hive --queen-url http://192.168.1.100:7833"
# ✅ Workers send heartbeat to Queen's IP
# ✅ Queen receives heartbeat
# ✅ Queen tracks worker correctly
```

### Implementation Requirements

**rbee-keeper manages hive lifecycle:**
1. Detect if hive is local or remote
2. For **local hives:** Use `http://localhost:7833`
3. For **remote hives:** Use Queen's public IP/hostname (from config)

**Configuration (rbee-keeper):**
```toml
# ~/.config/rbee/config.toml
[queen]
port = 7833
public_address = "192.168.1.100"  # ← REQUIRED for remote hives
# OR
public_hostname = "queen.local"   # ← Alternative
```

**Hive spawn (in rbee-keeper):**
```rust
async fn start_hive(config: HiveConfig) -> Result<HiveId> {
    let queen_url = if config.is_local {
        "http://localhost:7833".to_string()
    } else {
        // Remote hive - use public address
        format!("http://{}:7833", queen_config.public_address)
    };
    
    // Spawn hive with queen_url
    spawn_hive(&config, &queen_url).await?;
}
```

**Worker spawn (in rbee-hive):**
```rust
// Hive receives queen_url from CLI args, passes to worker
async fn spawn_worker(worker_id: &str, model: &str, queen_url: &str) -> Result<()> {
    Command::new("llm-worker")
        .arg("--worker-id").arg(worker_id)
        .arg("--model").arg(model)
        .arg("--queen-url").arg(queen_url)  // ← Pass through from hive CLI
        .spawn()?;
}
```

---

## 🔌 HTTP API

**The Queen provides HTTP API:**

```
# Health & Info
GET  /health                          # Health check
GET  /v1/info                         # Queen info (version, uptime)
GET  /v1/build-info                   # Build information

# Jobs (ONLY orchestration operations)
POST /v1/jobs                         # Submit job (Status, Infer only)
GET  /v1/jobs/:id/stream              # SSE stream for job narration

# Heartbeats (from workers and hives)
POST /v1/worker-heartbeat             # Worker heartbeat (from workers)
POST /v1/hive-heartbeat               # Hive heartbeat (from hives)
GET  /v1/heartbeats/stream            # SSE stream for all heartbeats

# Worker Registry (queen manages this)
POST /v1/workers/ready                # Worker ready callback (from workers)

# OpenAI Compatibility
POST /openai/v1/chat/completions      # OpenAI-compatible chat endpoint
GET  /openai/v1/models                # List models (OpenAI format)
GET  /openai/v1/models/{model}        # Get model details

# Web UI
GET  /ui/*                            # Queen's web UI
```

**Key Points:**
- **Queen's job API:** ONLY `Status` and `Infer` operations
- **Worker/Model management:** Talk to hive's job server directly (`http://localhost:7835/v1/jobs`)
- **rbee-keeper CLI:** Connects to BOTH queen AND hive job servers
- **rbee-keeper GUI:** Opens queen/hive/worker UIs in iframes
- **NO PROXYING:** Queen doesn't forward worker/model operations

---

## 🏗️ Architecture

### Binary Structure

```
bin/10_queen_rbee/
├── src/
│   ├── main.rs                    # Entry point, shutdown handler
│   ├── http/
│   │   ├── mod.rs                 # HTTP routes
│   │   ├── heartbeat.rs           # Heartbeat endpoints
│   │   ├── heartbeat_stream.rs    # SSE streaming
│   │   ├── info.rs                # Info endpoint
│   │   ├── build_info.rs          # Build info endpoint
│   │   └── jobs.rs                # Job submission (Status, Infer only)
│   ├── job_router.rs              # Operation routing (Status, Infer)
│   ├── hive_client.rs             # Internal client for hive job server
│   └── lib.rs                     # Re-exports
└── Cargo.toml

Dependencies (internal crates):
├── queen-rbee-hive-registry       # Hive registry (RAM, heartbeat-based)
├── queen-rbee-worker-registry     # Worker registry (RAM, heartbeat-based)
├── rbee-openai-adapter            # OpenAI API compatibility
├── scheduler                      # Worker selection
├── rbee-job-client                # Job submission client (for internal hive calls)
├── hive-contract                  # Hive types
├── worker-contract                # Worker types
└── observability-narration-core   # Narration system
```

**CRITICAL:** 
- NO SQLite. Both registries are RAM-based, built from heartbeats.
- NO proxying. Queen's job API only handles Status and Infer.
- Worker/model operations go directly to hive's job server.

---

## 🚫 What Queen Does NOT Do

### ❌ NO CLI Interface

Queen is a **daemon ONLY**. It does NOT have CLI commands.

```bash
# ❌ WRONG
queen-rbee hive start gpu-0  # NO CLI!

# ✅ CORRECT
queen-rbee --port 7833  # Daemon only
```

**Why:** rbee-keeper is the CLI. queen-rbee is the daemon.

### ❌ NO Direct Hive Lifecycle Management

**rbee-keeper** manages hive lifecycle (start/stop), NOT queen.

Queen sends jobs to hives, but doesn't spawn/stop them.

### ❌ NO Direct Inference Execution

Queen routes requests. `llm-worker` executes inference.

### ❌ NO Direct Worker Spawning

Queen sends jobs to hive. `rbee-hive` spawns workers.

### ❌ NO Direct Model Downloads

Queen sends jobs to hive. `rbee-hive` downloads models.

### ❌ NO Worker Heartbeat Aggregation by Hive

Workers send heartbeats DIRECTLY to queen, not through hive.

---

## 🔗 Component Communication

### Heartbeat Flow

```
┌─────────────────────────────────────────────────────────┐
│ llm-worker                                              │
│  - Starts HTTP server                                   │
│  - Calls: POST http://{queen_url}/v1/workers/ready      │
│  - Sends: POST http://{queen_url}/v1/worker-heartbeat   │
│           (every 30s)                                   │
└──────────────────┬──────────────────────────────────────┘
                   │ DIRECT (not through hive)
                   ▼
┌─────────────────────────────────────────────────────────┐
│ queen-rbee (THE BRAIN)                                  │
│  - Receives all heartbeats                              │
│  - Updates registries (RAM)                             │
│  - Broadcasts events via SSE                            │
│  - Sends own heartbeat every 2.5s                       │
└──────────────────┬──────────────────────────────────────┘
                   │ SSE
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Web UI / rbee-keeper                                    │
│  - Subscribes: GET /v1/heartbeats/stream                │
│  - Receives real-time updates                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ rbee-hive                                               │
│  - Receives queen_url from CLI args                     │
│  - Sends: POST http://{queen_url}/v1/hive-heartbeat     │
│           (every 30s)                                   │
│  - Does NOT aggregate worker heartbeats                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
                 Queen
```

### Inference Request Flow

```
┌─────────────────────────────────────────────────────────┐
│ rbee-keeper: rbee infer --model llama3                  │
└──────────────────┬──────────────────────────────────────┘
                   │ POST /v1/jobs (Operation::Infer)
                   ▼
┌─────────────────────────────────────────────────────────┐
│ queen-rbee (THE BRAIN)                                  │
│  1. Admission control (accept/reject)                   │
│  2. Check worker registry for available worker          │
│  3. If no worker exists:                                │
│     a. Select hive for model (scheduling)               │
│     b. Send job to hive: Operation::WorkerSpawn         │
│     c. Wait for worker heartbeat (worker is ready)      │
│  4. Route infer request DIRECTLY to worker              │
│  5. Relay SSE stream back to rbee-keeper                │
└──────────────────┬──────────────────────────────────────┘
                   │ 
                   │ (if worker spawn needed)
                   ▼
┌─────────────────────────────────────────────────────────┐
│ rbee-hive (WORKER LIFECYCLE ONLY)                       │
│  - Receives: Operation::WorkerSpawn job from queen      │
│  - Spawns llm-worker process with --queen-url           │
│  - Worker sends heartbeat to queen                      │
│  - Hive NEVER sees inference requests                   │
└──────────────────┬──────────────────────────────────────┘
                   │ spawns
                   ▼
┌─────────────────────────────────────────────────────────┐
│ llm-worker (INFERENCE)                                  │
│  - Loads model, generates tokens                        │
│  - Sends heartbeat to queen (via --queen-url)           │
│  - Receives infer requests DIRECTLY from queen          │
│  - Hive is NOT in the inference path                    │
└─────────────────────────────────────────────────────────┘
                   ▲
                   │ POST /v1/infer (DIRECT)
                   │
                 Queen
```

**CRITICAL:** 
- Hive ONLY does worker lifecycle (spawn/stop)
- Inference requests ALWAYS go directly from queen to worker
- Hive NEVER sees or routes inference requests

---

## 🎭 GUI vs API Usage

### GUI Mode (Personal Use)

**rbee-keeper GUI opens web UIs in iframes:**

```
rbee-keeper GUI
  ├─→ Queen Web UI (iframe: http://localhost:7833/ui)
  ├─→ Hive Web UI (iframe: http://localhost:7835/ui)
  └─→ Worker Web UI (iframe: http://localhost:8080/ui)
```

- User manages workers/models through hive's UI
- User runs inference through queen's UI
- Direct SDK access (no HTTP proxying)
- Each component has its own web UI

**Use Case:** Personal laptop, single user, interactive use

### CLI Mode (Scripting/Automation)

**rbee-keeper CLI connects to multiple job servers:**

```
rbee-keeper CLI
  ├─→ Queen Job Server (http://localhost:7833/v1/jobs)
  │   ├─ Status
  │   └─ Infer
  │
  └─→ Hive Job Server (http://localhost:7835/v1/jobs)
      ├─ WorkerSpawn, WorkerProcessList, WorkerProcessGet, WorkerProcessDelete
      └─ ModelDownload, ModelList, ModelGet, ModelDelete
```

**NO PROXYING** - CLI talks directly to queen AND hive.

**Use Case:** 
- Scripting and automation
- CI/CD pipelines
- Remote management
- Multi-machine setup (homelab)

---

## 📊 State Management

### Dual Registry System

**A. Hive Registry (RAM - EPHEMERAL)**
- Tracks all rbee-hive instances via heartbeats
- Lost on Queen restart (hives re-register via heartbeat)
- Updated in real-time via heartbeats
- Used for scheduling decisions

**B. Worker Registry (RAM - EPHEMERAL)**
- Tracks all workers across all hives via heartbeats
- Lost on Queen restart (workers re-register via heartbeat)
- Updated in real-time via heartbeats
- Used for scheduling decisions

**CRITICAL:** NO SQLite. Both registries are RAM-based, built from heartbeats.

---

## 🚨 Critical Implementation Notes

### 1. Queen URL Propagation

**The queen_url MUST be propagated through the entire chain:**

```
rbee-keeper Config → Hive Spawn → Worker Spawn → Worker Heartbeat
```

**Implementation checklist:**
- [ ] rbee-keeper config has `public_address` or `public_hostname`
- [ ] rbee-keeper detects local vs remote hive
- [ ] rbee-keeper passes correct queen_url when spawning hive
- [ ] Hive receives queen_url as CLI argument
- [ ] Hive passes queen_url to workers
- [ ] Workers use queen_url for heartbeat and ready callback

### 2. Heartbeat Endpoints

**Workers and hives MUST use the correct endpoints:**

```rust
// Worker ready callback (once, at startup)
POST http://{queen_url}/v1/workers/ready
{
  "worker": {
    "id": "worker-1",
    "hive_id": "gpu-0",
    "model_ref": "meta-llama/Llama-3.2-1B",
    "status": "Ready",
    "url": "http://localhost:9001"
  }
}

// Worker heartbeat (every 30s)
POST http://{queen_url}/v1/worker-heartbeat
{
  "worker": {
    "id": "worker-1",
    "status": "Ready",
    "slots_available": 1
  }
}

// Hive heartbeat (every 30s)
POST http://{queen_url}/v1/hive-heartbeat
{
  "hive_id": "gpu-0",
  "status": "Online",
  "workers": ["worker-1", "worker-2"]
}
```

### 3. SSE Streaming

**Queen provides TWO SSE endpoints:**

1. **Job-specific streaming:** `GET /v1/jobs/{job_id}/stream`
   - Narration events for specific job
   - Scoped to single operation
   - Ends when job completes

2. **Heartbeat streaming:** `GET /v1/heartbeats/stream`
   - Real-time heartbeat events
   - System-wide monitoring
   - Persistent connection

---

## 📚 References

### Planning Documents
- `bin/10_queen_rbee/TEAM_288_EVENT_DRIVEN_HEARTBEAT.md` - Heartbeat architecture
- `.business/stakeholders/RHAI_PROGRAMMABLE_SCHEDULER.md` - Future scheduling
- `PORT_CONFIGURATION.md` - Port assignments

### Implementation
- `bin/10_queen_rbee/src/` - Queen source code
- `bin/15_queen_rbee_crates/` - Queen-specific crates
- `bin/99_shared_crates/` - Shared infrastructure

---

## ✅ Summary

**The Queen is THE BRAIN:**
1. ✅ Makes ALL scheduling decisions
2. ✅ Sends jobs to hives (worker spawn, model download)
3. ✅ Tracks ALL workers via heartbeats (direct from workers)
4. ✅ Tracks ALL hives via heartbeats
5. ✅ Provides unified HTTP API for rbee-keeper
6. ✅ Routes infer requests DIRECTLY to workers
7. ✅ Provides OpenAI-compatible API

**The Queen does NOT:**
1. ❌ Have a CLI interface (daemon only)
2. ❌ Manage hive lifecycle (that's rbee-keeper)
3. ❌ Execute inference directly (that's llm-worker)
4. ❌ Expose worker/model operations through its job API (talk to hive directly)
5. ❌ Proxy operations to hive (no forwarding)
6. ❌ Use SQLite (registries are RAM-based, heartbeat-driven)

**Critical Bug Fix Needed:**
- [ ] Implement `public_address` config in rbee-keeper
- [ ] Fix queen_url propagation for remote hives
- [ ] Ensure workers/hives send heartbeat to correct IP (not localhost)

---

**Document Version:** 2.0  
**Last Updated:** Oct 29, 2025  
**Status:** ✅ COMPLETE - CORRECTED
