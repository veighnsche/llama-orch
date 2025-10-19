# TEAM-130D: rbee-hive - PART 1: METRICS & CRATES (CORRECTED)

**Binary:** `bin/rbee-hive`  
**Phase:** Phase 2, Day 5-6 (REWRITE)  
**Date:** 2025-10-19  
**Team:** 130D (Architectural Corrections Applied)

---

## 🎯 EXECUTIVE SUMMARY

**Current:** Pool manager daemon (4,184 LOC code-only, 48 files)  
**Violations:** CLI commands (297 LOC models + workers + status)  
**Corrected:** Daemon-only with 9 crates (3,887 LOC)  
**Risk:** LOW (remove CLI, keep HTTP API)  
**Timeline:** 2 weeks (80 hours, 2 developers)

**TEAM-130D Corrections Applied:**
- ✅ Removed models command (118 LOC violation)
- ✅ Removed workers command (105 LOC violation)
- ✅ Removed status command (74 LOC violation)
- ✅ Simplified main.rs to daemon-only (~50 LOC)
- ✅ ALL functionality via HTTP API (no CLI)

**Reference:** `TEAM_130C_ARCHITECTURAL_VIOLATIONS_SUMMARY.md`

---

## 📊 GROUND TRUTH METRICS

```bash
$ cloc bin/rbee-hive/src --quiet
Files: 48 | Code: 4,184 | Comments: 1,089 | Blanks: 869
Total Lines: 6,142
```

**TEAM-130D Analysis:**
- **Violations:** 297 LOC (models 118 + workers 105 + status 74)
- **Keep:** 3,887 LOC (HTTP API + daemon functionality)
- **Corrected Total:** 3,887 LOC

**Files with Violations:**
1. commands/models.rs - 118 LOC ❌ REMOVE (never planned)
2. commands/workers.rs - 105 LOC ❌ REMOVE (never planned)
3. commands/status.rs - 74 LOC ❌ REMOVE (never planned)
4. cli.rs CLI parsing - partial ❌ SIMPLIFY (only daemon args)

**Files to Keep:**
- commands/daemon.rs - 348 LOC ✅ KEEP (daemon mode)
- All HTTP API files ✅ KEEP
- All worker lifecycle files ✅ KEEP

---

## 🏗️ 9 CORRECTED CRATES (No CLI Violations)

| # | Crate | LOC | Purpose | Status |
|---|-------|-----|---------|--------|
| 1 | registry | 644 | Worker state management | ✅ Keep |
| 2 | http-server | 878 | HTTP API endpoints | ✅ Keep |
| 3 | http-middleware | 89 | Auth + logging | ✅ Keep |
| 4 | provisioner | 423 | Model download/cache | ✅ Keep |
| 5 | monitor | 386 | Worker health + restart | ✅ Keep |
| 6 | resources | 234 | Resource limits | ✅ Keep |
| 7 | shutdown | 179 | Graceful shutdown | ✅ Keep |
| 8 | metrics | 176 | Prometheus metrics | ✅ Keep |
| 9 | restart | 178 | Restart policies | ✅ Keep |

**Total:** 3,187 LOC in libraries + 50 LOC binary (daemon-only) + 650 LOC supporting

**Removed Violations:**
- commands/models.rs: -118 LOC
- commands/workers.rs: -105 LOC
- commands/status.rs: -74 LOC
- cli.rs CLI parsing: ~-50 LOC (simplified to daemon args only)

**Binary Becomes:**
```rust
// main.rs (~50 LOC)
#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "8080")]
    port: u16,
    
    #[arg(short, long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Start daemon (HTTP server)
    daemon::start(args.port, args.config).await
}
```

---

## 📦 CRATE SPECIFICATIONS

### CRATE 1: rbee-hive-registry (644 LOC) ✅ KEEP

**Purpose:** Worker state management (in-memory + optional persistence)  
**Files:** registry/{mod,worker_state,pool_state}.rs

**API:**
```rust
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<WorkerId, WorkerInfo>>>,
}

pub struct WorkerInfo {
    pub worker_id: String,
    pub pid: u32,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub state: WorkerState, // Loading, Idle, Busy, Unhealthy
    pub port: u16,
    pub vram_bytes: u64,
    pub last_heartbeat: Option<Instant>,
}

impl WorkerRegistry {
    pub async fn register(&self, worker: WorkerInfo);
    pub async fn list(&self) -> Vec<WorkerInfo>;
    pub async fn get(&self, worker_id: &str) -> Option<WorkerInfo>;
    pub async fn remove(&self, worker_id: &str);
    pub async fn update_state(&self, worker_id: &str, state: WorkerState);
}
```

**Dependencies:** tokio, serde

**TEAM-130D Note:** Registry accessed via HTTP API, never via CLI

---

### CRATE 2: rbee-hive-http-server (878 LOC) ✅ KEEP

**Purpose:** HTTP API for all hive operations  
**Files:** http/{server,routes,handlers/*}.rs

**API Endpoints:**

**Worker Management:**
- `POST /v1/workers/spawn` - Spawn worker
- `GET /v1/workers/list` - List all workers
- `POST /v1/workers/ready` - Worker ready callback
- `POST /v1/workers/shutdown` - Shutdown worker
- `GET /v1/workers/health` - Worker health status

**Model Management:**
- `POST /v1/models/provision` - Download model
- `GET /v1/models/list` - List cached models
- `GET /v1/models/catalog` - Available models
- `DELETE /v1/models/evict` - Evict model from cache

**Status:**
- `GET /v1/health` - Health check
- `GET /v1/status` - System status
- `GET /v1/metrics` - Prometheus metrics

**Admin:**
- `POST /v1/admin/shutdown` - Graceful shutdown

**Dependencies:** axum, tower, tokio, serde_json, auth-min, audit-logging

**TEAM-130D Verification:**
- ✅ No CLI commands (HTTP only)
- ✅ Called by queen-rbee via HTTP
- ✅ Workers call back via HTTP

---

### CRATE 3: rbee-hive-http-middleware (89 LOC) ✅ KEEP

**Purpose:** HTTP middleware (auth + logging)  
**Files:** http/middleware/{auth,logging}.rs

**Middleware:**
```rust
pub async fn auth_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response> {
    // Timing-safe token comparison
    // Audit logging for auth events
}

pub async fn logging_middleware(
    request: Request,
    next: Next,
) -> Response {
    // Request/response logging
    // Correlation ID propagation
}
```

**Dependencies:** axum, tower, auth-min, audit-logging

---

### CRATE 4: rbee-hive-provisioner (423 LOC) ✅ KEEP

**Purpose:** Model download and cache management  
**Files:** provisioner/{download,cache,verification}.rs

**API:**
```rust
pub async fn provision_model(model_ref: &str, cache_dir: &Path) -> Result<PathBuf> {
    // 1. Parse model reference
    // 2. Check if already cached
    // 3. If not: download from HuggingFace/local
    // 4. Verify checksum
    // 5. Return local path
}

pub async fn evict_model(model_ref: &str) -> Result<()> {
    // Remove from cache
}

pub async fn list_cached_models() -> Result<Vec<ModelInfo>> {
    // List all models in cache
}
```

**Dependencies:** reqwest, tokio, sha2

**TEAM-130D Note:** Called via HTTP `/v1/models/provision` endpoint, not CLI

---

### CRATE 5: rbee-hive-monitor (386 LOC) ✅ KEEP

**Purpose:** Worker health monitoring and auto-restart  
**Files:** monitor/{health_check,restart_policy,spawn}.rs

**API:**
```rust
pub async fn spawn_worker(request: SpawnWorkerRequest) -> Result<WorkerId> {
    // 1. Preflight checks (VRAM, model exists)
    // 2. Select GPU device
    // 3. Spawn worker process
    // 4. Register in registry
    // 5. Wait for ready callback
    // 6. Start health monitoring
}

pub async fn monitor_worker_health(worker_id: &str) {
    loop {
        tokio::time::sleep(Duration::from_secs(10)).await;
        
        // Check worker health (HTTP health endpoint)
        // If unhealthy: restart or mark failed
    }
}

pub async fn shutdown_worker(worker_id: &str) -> Result<()> {
    // 1. HTTP shutdown request
    // 2. Wait for graceful exit (10s)
    // 3. Force kill if timeout
    // 4. Remove from registry
}
```

**Dependencies:** tokio, reqwest, anyhow

**TEAM-130D Note:** Triggered via HTTP API, not CLI

---

### CRATE 6: rbee-hive-resources (234 LOC) ✅ KEEP

**Purpose:** Resource limits and VRAM management  
**Files:** resources/{gpu,vram,limits}.rs

**API:**
```rust
pub fn get_available_vram(device: u32) -> Result<u64>;
pub fn check_vram_sufficient(device: u32, required: u64) -> Result<bool>;
pub fn select_device(backend: &str, required_vram: u64) -> Result<u32>;
```

**Dependencies:** sysinfo (or platform-specific)

---

### CRATE 7: rbee-hive-shutdown (179 LOC) ✅ KEEP

**Purpose:** Graceful shutdown coordination  
**Files:** shutdown/{graceful,signal_handler}.rs

**API:**
```rust
pub async fn graceful_shutdown(registry: Arc<WorkerRegistry>) {
    // 1. Stop accepting new requests
    // 2. Drain queue
    // 3. Shutdown all workers (parallel)
    // 4. Wait for completion (30s timeout)
    // 5. Force kill remaining workers
}
```

**Dependencies:** tokio, signal-hook

**TEAM-130D Note:** Triggered by queen-rbee (SIGTERM) or admin API

---

### CRATE 8: rbee-hive-metrics (176 LOC) ✅ KEEP

**Purpose:** Prometheus metrics collection  
**Files:** metrics/{prometheus,collectors}.rs

**Metrics:**
- `rbee_hive_workers_total` - Total workers
- `rbee_hive_workers_by_state` - Workers by state
- `rbee_hive_vram_used_bytes` - VRAM usage
- `rbee_hive_requests_total` - Request count
- `rbee_hive_request_duration_seconds` - Latency

**Dependencies:** prometheus

---

### CRATE 9: rbee-hive-restart (178 LOC) ✅ KEEP

**Purpose:** Worker restart policies  
**Files:** restart/{policy,backoff}.rs

**API:**
```rust
pub enum RestartPolicy {
    Always,
    OnFailure,
    Never,
}

pub struct BackoffPolicy {
    initial_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
}

pub async fn should_restart(
    policy: &RestartPolicy,
    exit_code: Option<i32>,
    attempt: u32,
) -> bool;
```

**Dependencies:** tokio

---

## 📊 DEPENDENCY GRAPH (CORRECTED)

```
Layer 0 (Standalone):
- registry (644 LOC)
- resources (234 LOC)

Layer 1 (Core):
- provisioner (423 LOC) → uses resources
- restart (178 LOC)

Layer 2 (Management):
- monitor (386 LOC) → uses registry, resources, provisioner, restart
- metrics (176 LOC) → uses registry
- shutdown (179 LOC) → uses registry, monitor

Layer 3 (HTTP):
- http-middleware (89 LOC)
- http-server (878 LOC) → uses ALL Layer 0-2

Binary (50 LOC) → starts http-server
```

**No circular dependencies ✅**  
**No CLI crates ✅**

---

## 🔗 CORRECTED ARCHITECTURE

### Daemon-Only Pattern:

```
queen-rbee (orchestrator)
    ↓ HTTP API calls
rbee-hive (daemon)
    ├─ HTTP server (878 LOC)
    ├─ Worker lifecycle (monitor 386 LOC)
    ├─ Model provisioning (423 LOC)
    └─ Resource management (234 LOC)
```

**NO CLI**, **HTTP API only**

### Example: Worker Spawn Flow

```
queen-rbee:
  POST http://hive-gpu-0:8080/v1/workers/spawn
  {
    "model_ref": "llama-7b",
    "backend": "cuda",
    "device": 0
  }

rbee-hive HTTP handler:
  1. Receive spawn request
  2. Call monitor::spawn_worker()
  3. Return worker_id

monitor::spawn_worker():
  1. Preflight checks (VRAM, model exists)
  2. Call provisioner::provision_model() if needed
  3. Spawn llm-worker-rbee process
  4. Register in registry
  5. Wait for worker ready callback
  6. Start health monitoring task
  7. Return worker_id

llm-worker-rbee:
  1. Load model
  2. Callback: POST http://hive:8080/v1/workers/ready
  3. Wait for inference requests
```

---

## 📋 COMPARISON: TEAM-130C vs TEAM-130D

| Component | 130C (with CLI) | 130D (daemon-only) | Change |
|-----------|-----------------|---------------------|--------|
| **Crates** | 10 (inc. cli) | 9 (no cli) | -1 crate |
| **commands/models.rs** | 118 LOC | REMOVED | -118 LOC |
| **commands/workers.rs** | 105 LOC | REMOVED | -105 LOC |
| **commands/status.rs** | 74 LOC | REMOVED | -74 LOC |
| **cli.rs** | 68 LOC | ~10 LOC (simple args) | -58 LOC |
| **main.rs** | ~100 LOC | ~50 LOC | -50 LOC |
| **Total LOC** | 4,184 | 3,887 | -297 LOC |

**Net Result:** -297 LOC CLI violations removed, all functionality via HTTP

---

## ✅ TEAM-130D CORRECTIONS APPLIED

**Violations Removed:**
1. ✅ Deleted commands/models.rs (118 LOC CLI)
2. ✅ Deleted commands/workers.rs (105 LOC CLI)
3. ✅ Deleted commands/status.rs (74 LOC CLI)
4. ✅ Simplified cli.rs (only port/config args)
5. ✅ Simplified main.rs (~50 LOC daemon-only)

**Architecture Verified:**
- ✅ NO CLI commands (daemon only)
- ✅ ALL functionality via HTTP API
- ✅ Controlled by queen-rbee
- ✅ Workers call back via HTTP

**Why This is Correct:**
- rbee-hive is managed by queen-rbee
- No human should interact with rbee-hive directly
- All operations via HTTP (not CLI)
- Daemon runs persistently (no one-off commands)

---

**Status:** TEAM-130D Complete - CLI Violation Fixed  
**Next:** queen-rbee PART1 (add 15 missing crates)  
**Reference:** `TEAM_130C_ARCHITECTURAL_VIOLATIONS_SUMMARY.md`
