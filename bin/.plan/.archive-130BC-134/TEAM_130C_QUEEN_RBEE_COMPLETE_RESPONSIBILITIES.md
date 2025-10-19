# TEAM-130C: queen-rbee COMPLETE RESPONSIBILITIES

**Date:** 2025-10-19  
**Status:** 🔴 ARCHITECTURAL CRISIS - MASSIVE GAPS IDENTIFIED  
**Source:** Specifications + Component Docs

---

## 🚨 CRITICAL FINDINGS

**queen-rbee is MASSIVELY underspecified in current codebase!**

**What EXISTS (2,015 LOC):**
- ✅ Beehive registry (SQLite)
- ✅ Worker registry (RAM)
- ✅ HTTP server (basic routes)
- ✅ Cascading shutdown via SSH

**What is MISSING (~8,000+ LOC estimated):**
- ❌ Scheduler engine (Rhai programmable - M2 feature but core logic needed)
- ❌ Admission control
- ❌ Queue management (job queue persistence)
- ❌ Load balancing
- ❌ Hive lifecycle START (spawn local + remote hives)
- ❌ Request routing/orchestration logic
- ❌ Model provisioning coordination
- ❌ Health monitoring of hives
- ❌ Eviction policies
- ❌ Retry & backoff logic
- ❌ SSE relay logic

---

## 📋 COMPLETE QUEEN-RBEE RESPONSIBILITIES

### From Specs (00_llama-orch.md SYS-6.1.x):

### 1. **Orchestrator Intelligence (SYS-6.1.1)**

**THE BRAIN** - Makes ALL intelligent decisions

**Core Intelligence Functions:**
- Admission control (accept/reject requests)
- Job queue management
- Worker selection algorithms
- Load balancing across hives
- Eviction policies (worker + model)
- Retry logic with exponential backoff
- Timeout management
- Cancellation handling

**Rhai Scheduler Engine (M2 - but basic scheduler needed now):**
- User-programmable scheduling logic
- 40+ helper functions (workers.least_loaded(), gpu_vram_free(), etc.)
- Platform mode (immutable) vs Home/Lab mode (customizable)
- YAML config support
- Web UI policy builder (future)

---

### 2. **State Management (SYS-6.1.2)**

**Dual Registry System:**

**A. Hive Registry (SQLite - PERSISTENT):**
```sql
CREATE TABLE hives (
    hive_id TEXT PRIMARY KEY,
    hostname TEXT NOT NULL,
    port INTEGER NOT NULL,
    mode TEXT NOT NULL,  -- 'local' or 'network'
    ssh_host TEXT,
    ssh_port INTEGER,
    ssh_user TEXT,
    ssh_key_path TEXT,
    capabilities JSON,  -- GPUs, backends, models
    last_seen TIMESTAMP,
    status TEXT  -- 'online', 'offline', 'error'
);
```

**B. Worker Registry (RAM - EPHEMERAL):**
```rust
struct WorkerRegistry {
    workers: HashMap<WorkerId, WorkerInfo>,
}

struct WorkerInfo {
    worker_id: String,
    hive_id: String,  // Which hive owns this worker
    model_ref: String,
    backend: String,
    device: u32,
    state: WorkerState,  // Loading, Idle, Busy
    url: String,
    slots_total: u32,
    slots_available: u32,
    vram_bytes: Option<u64>,
}
```

**C. Job Queue (SQLite - PERSISTENT):**
```sql
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    tenant_id TEXT,
    model_ref TEXT,
    priority TEXT,  -- 'interactive', 'batch'
    state TEXT,  -- 'queued', 'running', 'complete', 'failed'
    worker_id TEXT,
    submitted_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error TEXT
);
```

---

### 3. **Hive Lifecycle Management**

**A. Local Hives (Same Machine):**
```rust
async fn start_local_hive(config: HiveConfig) -> Result<HiveId> {
    // 1. Spawn rbee-hive daemon as child process
    let child = Command::new("rbee-hive")
        .arg("daemon")  // rbee-hive ONLY daemon mode
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
    
    // 2. Check if rbee-hive binary exists
    // 3. If not, upload binary
    // 4. Start rbee-hive daemon via SSH
    ssh.exec("rbee-hive daemon --port 8080")?;
    
    // 5. Verify health check over network
    // 6. Register in SQLite hive registry
    // 7. Start health monitoring task
}

async fn stop_network_hive(hive_id: &str) {
    // ALREADY EXISTS in current code (shutdown_all_hives function)
    // 1. SSH to remote machine
    // 2. Find rbee-hive PID
    // 3. Send SIGTERM
    // 4. Update SQLite status
}
```

**C. Health Monitoring:**
```rust
async fn monitor_hive_health(hive_id: &str) {
    loop {
        tokio::time::sleep(Duration::from_secs(15)).await;
        
        // 1. GET /health from hive
        // 2. Check response time
        // 3. Update SQLite last_seen
        // 4. If timeout, mark as offline
        // 5. If offline > threshold, trigger failover
    }
}
```

---

### 4. **Request Routing & Orchestration**

**Inference Request Flow:**
```rust
async fn handle_inference_request(req: InferenceRequest) -> Result<SseStream> {
    // 1. ADMISSION CONTROL
    if !admission_check(&req) {
        return reject("quota_exceeded" or "rate_limited");
    }
    
    // 2. QUEUE MANAGEMENT
    let job_id = enqueue_job(&req).await?;
    
    // 3. WORKER SELECTION (Scheduler)
    let worker = select_worker(&req.model_ref).await?;
    
    // 4. If no worker exists, SPAWN WORKER
    if worker.is_none() {
        let hive = select_hive_for_model(&req.model_ref).await?;
        spawn_worker_on_hive(hive, &req.model_ref).await?;
        
        // Wait for worker ready (with timeout)
        worker = wait_for_worker(&req.model_ref, Duration::from_secs(300)).await?;
    }
    
    // 5. ROUTE REQUEST to worker
    let worker_url = worker.url;
    let response = http_client.post(format!("{}/v1/execute", worker_url))
        .json(&req)
        .send().await?;
    
    // 6. SSE RELAY (stream tokens back to client)
    relay_sse_stream(response, job_id).await
}
```

**Worker Selection (Scheduler):**
```rust
async fn select_worker(model_ref: &str) -> Result<Option<WorkerInfo>> {
    // Get all workers with this model
    let candidates = worker_registry.list()
        .filter(|w| w.model_ref == model_ref && w.state == WorkerState::Idle);
    
    if candidates.is_empty() {
        return Ok(None);  // Need to spawn worker
    }
    
    // BASIC SCHEDULER (before Rhai)
    // Use least-loaded worker
    let worker = candidates.min_by_key(|w| w.slots_total - w.slots_available);
    
    // ADVANCED SCHEDULER (M2 - Rhai)
    // let worker = rhai_engine.eval("select_worker(candidates)")?;
    
    Ok(Some(worker))
}
```

---

### 5. **Model Provisioning Coordination**

**Centralized Model Catalog:**
```rust
async fn ensure_model_available(model_ref: &str, hive_id: &str) -> Result<()> {
    // 1. Check queen's model catalog (which hives have which models)
    if hive_has_model(hive_id, model_ref).await? {
        return Ok(());
    }
    
    // 2. Trigger download on hive via HTTP
    let hive_url = get_hive_url(hive_id).await?;
    http_client.post(format!("{}/v1/models/provision", hive_url))
        .json(&ProvisionRequest { model_ref })
        .send().await?;
    
    // 3. Poll for download completion
    wait_for_model_ready(hive_id, model_ref, Duration::from_secs(600)).await?;
    
    // 4. Update queen's model catalog
    update_model_catalog(hive_id, model_ref).await?;
    
    Ok(())
}
```

---

### 6. **Admission Control (SYS-6.1.1)**

```rust
fn admission_check(req: &InferenceRequest) -> Result<()> {
    // 1. Check tenant quota (Platform mode)
    if platform_mode {
        if tenant_over_quota(req.tenant_id) {
            return Err("quota_exceeded");
        }
    }
    
    // 2. Check rate limits
    if rate_limit_exceeded(req.tenant_id) {
        return Err("rate_limited");
    }
    
    // 3. Check system capacity
    let available_workers = worker_registry.count_idle();
    let queue_length = job_queue.len();
    
    if queue_length > MAX_QUEUE_LENGTH && available_workers == 0 {
        return Err("system_at_capacity");
    }
    
    Ok(())
}
```

---

### 7. **Queue Management (SYS-6.1.3 - Persistent State Store)**

```rust
async fn enqueue_job(req: &InferenceRequest) -> Result<JobId> {
    let job = Job {
        job_id: Uuid::new_v4().to_string(),
        tenant_id: req.tenant_id.clone(),
        model_ref: req.model_ref.clone(),
        priority: req.priority.unwrap_or(Priority::Interactive),
        state: JobState::Queued,
        submitted_at: Utc::now(),
        ..Default::default()
    };
    
    // Persist to SQLite
    db.execute("INSERT INTO jobs (...) VALUES (...)", &job).await?;
    
    Ok(job.job_id)
}

async fn dequeue_job() -> Result<Option<Job>> {
    // Priority-based dequeue
    db.query_row(
        "SELECT * FROM jobs WHERE state = 'queued' ORDER BY priority DESC, submitted_at ASC LIMIT 1"
    ).await
}
```

---

### 8. **Load Balancing (SYS-6.1.4 - Queue Optimizer)**

```rust
async fn optimize_queue() {
    loop {
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        // 1. Get queued jobs
        let queued = job_queue.get_queued().await;
        
        // 2. Get available workers
        let workers = worker_registry.list_idle().await;
        
        // 3. Match jobs to workers
        for job in queued {
            if let Some(worker) = find_best_worker(&job, &workers) {
                assign_job_to_worker(job, worker).await;
            }
        }
        
        // 4. Eviction if needed
        if workers.is_empty() && need_eviction() {
            evict_least_recently_used_worker().await;
        }
    }
}
```

---

### 9. **Eviction Policies (SYS-6.1.1)**

```rust
async fn evict_least_recently_used_worker() {
    // Find LRU idle worker
    let lru_worker = worker_registry.list()
        .filter(|w| w.state == WorkerState::Idle)
        .min_by_key(|w| w.last_used_at);
    
    if let Some(worker) = lru_worker {
        // Send shutdown to worker
        http_client.post(format!("{}/v1/admin/shutdown", worker.url)).await?;
        
        // Remove from registry
        worker_registry.remove(&worker.worker_id).await;
    }
}
```

---

### 10. **Retry & Backoff (SYS-6.1.6)**

```rust
async fn retry_with_backoff<F, T>(operation: F) -> Result<T>
where
    F: Fn() -> Future<Output = Result<T>>,
{
    let mut attempt = 0;
    let max_attempts = 3;
    
    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) if attempt < max_attempts => {
                let backoff = Duration::from_secs(2_u64.pow(attempt));
                tokio::time::sleep(backoff).await;
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}
```

---

### 11. **SSE Relay (SYS-6.1.1)**

```rust
async fn relay_sse_stream(
    worker_response: Response,
    job_id: JobId,
) -> Result<SseStream> {
    // 1. Read worker SSE stream
    let worker_stream = worker_response.bytes_stream();
    
    // 2. Add queen metadata (correlation IDs, timing)
    let relay_stream = worker_stream.map(|event| {
        add_queen_metadata(event, &job_id)
    });
    
    // 3. Update job state on completion
    relay_stream.on_complete(|| {
        update_job_state(&job_id, JobState::Complete).await;
    });
    
    Ok(relay_stream)
}
```

---

## 🏗️ PROPOSED QUEEN-RBEE CRATE STRUCTURE

Based on responsibilities, queen-rbee needs **15-20 crates**:

| # | Crate | LOC Est | Purpose |
|---|-------|---------|---------|
| 1 | queen-rbee-registry | 600 | Dual registry (hive + worker) |
| 2 | queen-rbee-hive-lifecycle | 800 | Start/stop/monitor hives (local + SSH) |
| 3 | queen-rbee-admission | 400 | Admission control + rate limiting |
| 4 | queen-rbee-queue | 500 | Job queue management (SQLite) |
| 5 | queen-rbee-scheduler | 1,200 | Worker selection (basic + Rhai M2) |
| 6 | queen-rbee-router | 600 | Request routing logic |
| 7 | queen-rbee-provisioner | 500 | Model provisioning coordination |
| 8 | queen-rbee-eviction | 400 | Eviction policies |
| 9 | queen-rbee-retry | 300 | Retry & backoff logic |
| 10 | queen-rbee-sse-relay | 400 | SSE streaming relay |
| 11 | queen-rbee-http-server | 1,200 | HTTP API endpoints |
| 12 | queen-rbee-remote | 400 | SSH client (hive management) |
| 13 | queen-rbee-monitor | 500 | Health monitoring |
| 14 | queen-rbee-metrics | 300 | Metrics collection |
| 15 | queen-rbee-auth | 200 | Authentication (shared with current) |

**Total:** ~8,300 LOC in libraries + 200 LOC binary

**Current:** 2,015 LOC (only ~24% complete!)

---

## 🔴 ARCHITECTURAL VIOLATIONS IDENTIFIED

### Violation #1: rbee-keeper has SSH ❌

**Current (WRONG):**
- rbee-keeper has ssh-client crate
- Commands like `rbee hive models|workers|status` use SSH
- rbee-keeper talks DIRECTLY to rbee-hive

**CORRECT:**
- rbee-keeper ONLY talks to queen-rbee (HTTP)
- rbee-keeper is CLI frontend for queen-rbee
- queen-rbee has SSH (for managing remote hives)
- rbee-keeper NEVER bypasses queen

**Fix:** Remove ALL SSH from rbee-keeper, add SSH to queen-rbee

---

### Violation #2: rbee-hive has CLI ❌

**Current (WRONG):**
- rbee-hive has 719 LOC CLI crate
- commands/daemon.rs, commands/models.rs, commands/workers.rs
- cli.rs with clap parsing

**CORRECT:**
- rbee-hive is DAEMON ONLY (no CLI)
- Only HTTP API for programmatic control
- queen-rbee orchestrates it via HTTP
- Binary is ~50 LOC (just start daemon)

**Fix:** Remove ALL CLI from rbee-hive, HTTP API only

---

### Violation #3: rbee-keeper does orchestration ❌

**Current (WRONG):**
- rbee-keeper commands directly spawn workers
- rbee-keeper decides which hive to use
- rbee-keeper implements inference logic

**CORRECT:**
- rbee-keeper sends request to queen-rbee
- queen-rbee decides which hive
- queen-rbee orchestrates worker spawn
- rbee-keeper is THIN CLIENT

**Fix:** Move ALL orchestration logic from rbee-keeper to queen-rbee

---

## 📊 COMPARISON: CURRENT vs SPEC

| Responsibility | Current (2,015 LOC) | Spec Required | Gap |
|----------------|---------------------|---------------|-----|
| Hive registry | ✅ Has (200 LOC) | ✅ Required | OK |
| Worker registry | ✅ Has (153 LOC) | ✅ Required | OK |
| HTTP server | ✅ Has (897 LOC) | ✅ Required | OK |
| Shutdown hives | ✅ Has (158 LOC) | ✅ Required | OK |
| **START hives** | ❌ MISSING | ✅ Required | **800 LOC** |
| **Scheduler** | ❌ MISSING | ✅ Required | **1,200 LOC** |
| **Admission** | ❌ MISSING | ✅ Required | **400 LOC** |
| **Queue** | ❌ MISSING | ✅ Required | **500 LOC** |
| **Router** | ❌ MISSING | ✅ Required | **600 LOC** |
| **Provisioner** | ❌ MISSING | ✅ Required | **500 LOC** |
| **Eviction** | ❌ MISSING | ✅ Required | **400 LOC** |
| **Retry** | ❌ MISSING | ✅ Required | **300 LOC** |
| **SSE relay** | ❌ MISSING | ✅ Required | **400 LOC** |
| **Monitor** | ❌ MISSING | ✅ Required | **500 LOC** |
| **Metrics** | ❌ MISSING | ✅ Required | **300 LOC** |

**Completion:** 24% (2,015 / 8,300 LOC)

---

## 🎯 CORRECTED ARCHITECTURE

```
rbee-keeper (CLI - NO SSH!)
    ↓ HTTP ONLY
queen-rbee (Orchestrator Daemon - THE BRAIN)
    ├─ Admission control
    ├─ Job queue (SQLite)
    ├─ Scheduler (basic + Rhai M2)
    ├─ Hive lifecycle (SSH for remote hives)
    │   ├─ Start local hives (child process)
    │   ├─ Start remote hives (SSH)
    │   ├─ Monitor hive health
    │   └─ Shutdown hives (already exists)
    ├─ Request router
    ├─ Model provisioner
    ├─ Eviction manager
    ├─ SSE relay
    └─ Metrics collector
    
rbee-hive (Daemon - NO CLI!)
    ├─ HTTP API ONLY
    ├─ Worker lifecycle (local system)
    ├─ Model provisioning (download/cache)
    └─ GPU management
    
llm-worker-rbee (Worker)
    ├─ HTTP API
    └─ LLM inference
```

---

## 🚨 IMMEDIATE ACTIONS REQUIRED

1. ✅ **Document complete queen-rbee responsibilities** (THIS DOCUMENT)
2. ❌ **Remove SSH from rbee-keeper** (architectural violation)
3. ❌ **Remove CLI from rbee-hive** (architectural violation)
4. ❌ **Add hive lifecycle START to queen-rbee** (~800 LOC)
5. ❌ **Add scheduler to queen-rbee** (~1,200 LOC)
6. ❌ **Add admission control to queen-rbee** (~400 LOC)
7. ❌ **Add queue management to queen-rbee** (~500 LOC)
8. ❌ **Add request router to queen-rbee** (~600 LOC)

**Total missing:** ~8,300 LOC across 15 crates

---

## 📚 REFERENCES

- **Spec:** `bin/.specs/00_llama-orch.md` (SYS-6.1.x - Orchestratord)
- **Component Doc:** `.docs/components/QUEEN_RBEE.md`
- **Scheduler Spec:** `.business/stakeholders/RHAI_PROGRAMMABLE_SCHEDULER.md`
- **Current Code:** `bin/queen-rbee/src/main.rs` (364 lines total)

---

**Status:** 🔴 CRITICAL - queen-rbee is 24% complete, massive architectural violations exist  
**Next:** Update all PART1 documents with correct architecture
