# TEAM-130D: queen-rbee - PART 1: METRICS & CRATES (CORRECTED)

**Binary:** `bin/queen-rbee`  
**Phase:** Phase 2, Day 5-6 (REWRITE)  
**Date:** 2025-10-19  
**Team:** 130D (Complete Responsibilities)

---

## 🎯 EXECUTIVE SUMMARY

**Current:** 2,015 LOC (24% complete - basic structure only)  
**Missing:** 8,300 LOC (76% of orchestrator functionality)  
**Corrected:** 10,315 LOC across 15 crates  
**Risk:** HIGH (core orchestration)  
**Timeline:** 6-8 weeks (3-4 developers)

**Reference:** `TEAM_130C_QUEEN_RBEE_COMPLETE_RESPONSIBILITIES.md`

---

## 🏗️ 15 COMPLETE CRATES

| # | Crate | LOC | Status |
|---|-------|-----|--------|
| 1 | registry | 353 | ✅ Exists |
| 2 | hive-lifecycle | 800 | ❌ Add (START hives) |
| 3 | admission | 400 | ❌ Add |
| 4 | queue | 500 | ❌ Add (SQLite jobs) |
| 5 | scheduler | 1,200 | ❌ Add (worker selection) |
| 6 | router | 600 | ❌ Add (orchestration) |
| 7 | provisioner | 500 | ❌ Add (model coord) |
| 8 | eviction | 400 | ❌ Add |
| 9 | retry | 300 | ❌ Add |
| 10 | sse-relay | 400 | ❌ Add |
| 11 | http-server | 1,200 | ⚠️ Expand (+300) |
| 12 | remote | 400 | ⚠️ Expand (+324) |
| 13 | monitor | 500 | ❌ Add (hive health) |
| 14 | metrics | 300 | ❌ Add |
| 15 | auth | 200 | ✅ Exists (partial) |

**Total:** 8,553 LOC libraries + 200 LOC binary + 1,562 LOC supporting = 10,315 LOC

**Breakdown:**
- Exists: 2,015 LOC
- Add: 6,600 LOC (crates 2-10, 13-14)
- Expand: 624 LOC (crates 11-12)
- Supporting: 1,076 LOC

---

## 📦 CRITICAL MISSING CRATES

### 1. hive-lifecycle (800 LOC) - START HIVES

**Purpose:** Start/stop local + SSH remote hives

```rust
// Local hive (same machine)
pub async fn start_local_hive(config: LocalHiveConfig) -> Result<HiveId> {
    // 1. Check if running (health check)
    // 2. Find rbee-hive binary
    // 3. Spawn child process
    // 4. Wait for ready
    // 5. Register in hive registry
}

// Network hive (SSH remote)
pub async fn start_network_hive(config: NetworkHiveConfig) -> Result<HiveId> {
    // 1. SSH to remote machine
    // 2. Check rbee-hive binary exists
    // 3. Start via SSH (nohup rbee-hive &)
    // 4. Wait for health check over network
    // 5. Register in hive registry
    // 6. Start health monitoring
}
```

**CRITICAL:** Queen can only STOP hives (exists), cannot START them (missing)

---

### 2. scheduler (1,200 LOC) - WORKER SELECTION

**Purpose:** Select worker for inference (basic + Rhai M2)

```rust
pub async fn select_worker(&self, model_ref: &str) -> Result<Option<WorkerInfo>> {
    // Get idle workers with model
    let candidates = worker_registry.list()
        .filter(|w| w.model_ref == model_ref && w.state == Idle);
    
    // Basic: least-loaded
    candidates.min_by_key(|w| w.current_load)
}

pub async fn select_hive_for_model(&self, model_ref: &str) -> Result<HiveId> {
    // Select hive with most free VRAM
    hive_registry.list()
        .filter(|h| h.status == "online")
        .max_by_key(|h| h.free_vram_bytes)
}
```

**CRITICAL:** Core orchestration intelligence (50% of queen's brain)

---

### 3. queue (500 LOC) - JOB PERSISTENCE

**Purpose:** Job queue (SQLite)

```sql
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    tenant_id TEXT,
    model_ref TEXT,
    priority TEXT, -- interactive, batch
    state TEXT, -- queued, running, complete, failed
    worker_id TEXT,
    submitted_at INTEGER,
    completed_at INTEGER
);
```

---

### 4. router (600 LOC) - REQUEST ROUTING

**Purpose:** Complete inference orchestration

```rust
pub async fn route_inference(&self, req: InferenceRequest) -> Result<SseStream> {
    // 1. Select worker (or spawn if needed)
    let worker = self.scheduler.select_worker(&req.model).await?;
    
    let worker = if let Some(w) = worker {
        w
    } else {
        // No worker - spawn one
        let hive_id = self.scheduler.select_hive_for_model(&req.model).await?;
        self.hive_lifecycle.ensure_hive_running(&hive_id).await?;
        self.spawn_worker_on_hive(&hive_id, &req.model).await?
    };
    
    // 2. Route to worker
    let response = reqwest::post(format!("{}/v1/execute", worker.url))
        .json(&req)
        .send().await?;
    
    // 3. Relay SSE
    Ok(self.sse_relay.relay(response).await?)
}
```

**CRITICAL:** Core request flow (30% of queen's brain)

---

### 5. admission (400 LOC) - QUOTA & RATE LIMITING

```rust
pub async fn check_admission(&self, req: &InferenceRequest) -> Result<()> {
    // 1. Check quota
    if over_quota(&req.tenant_id) {
        return Err("quota_exceeded");
    }
    
    // 2. Check rate limit
    if rate_limited(&req.tenant_id) {
        return Err("rate_limited");
    }
    
    // 3. Check capacity
    if queue_full() && no_idle_workers() {
        return Err("system_at_capacity");
    }
    
    Ok(())
}
```

---

### 6. provisioner (500 LOC) - MODEL COORDINATION

```rust
pub async fn ensure_model_available(&self, model_ref: &str, hive_id: &str) -> Result<()> {
    if hive_has_model(hive_id, model_ref).await? {
        return Ok(());
    }
    
    // Trigger download on hive
    reqwest::post(format!("http://hive-{}/v1/models/provision", hive_id))
        .json(&json!({ "model_ref": model_ref }))
        .send().await?;
    
    // Wait for completion
    wait_for_model_ready(hive_id, model_ref).await?;
    
    Ok(())
}
```

---

### 7-10. Supporting Crates

- **eviction (400 LOC):** LRU worker eviction
- **retry (300 LOC):** Exponential backoff
- **sse-relay (400 LOC):** SSE stream relay worker→queen→keeper
- **monitor (500 LOC):** Hive health monitoring

---

### 11. http-server (897→1,200 LOC) EXPAND

**Add NEW Endpoints:**
- `POST /v2/models/download` (via provisioner)
- `GET /v2/models/list` (via provisioner)
- `POST /v2/workers/spawn` (via router)
- `GET /v2/logs` (via hive HTTP)

---

### 12. remote (76→400 LOC) EXPAND SSH

**Current:** Only shutdown
**Add:** Full SSH client, connection pooling, command execution

---

## 📊 COMPARISON: 130C vs 130D

| Metric | 130C (Incomplete) | 130D (Complete) |
|--------|-------------------|-----------------|
| LOC | 2,015 | 10,315 |
| Crates | 4 | 15 |
| Completion | 24% | 100% |
| Missing | 8 crates | 0 |

**Key Additions:**
- hive-lifecycle: START hives (800 LOC)
- scheduler: Worker selection (1,200 LOC)
- queue: Job persistence (500 LOC)
- router: Orchestration (600 LOC)
- 7 more crates (3,600 LOC)

---

**Status:** TEAM-130D Complete - All Responsibilities Documented  
**Next:** llm-worker PART1 (minor corrections)
