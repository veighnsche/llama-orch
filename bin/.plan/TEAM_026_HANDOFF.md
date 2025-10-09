# TEAM-026 Handoff: Implement test-001 MVP

**Date:** 2025-10-09T23:05:00+02:00  
**From:** TEAM-025  
**To:** TEAM-026  
**Status:** Ready for implementation  
**Priority:** CRITICAL - MVP implementation

---

## Executive Summary

**What TEAM-025 Completed:**
1. ✅ Re-analyzed MVP requirements from test-001-mvp.md
2. ✅ Corrected all architecture documents (TEAM-024 had wrong assumptions)
3. ✅ Applied rebranding (rbees-* → rbee-* naming)
4. ✅ Clarified that rbee-hive is an HTTP daemon (not SSH-controlled CLI)
5. ✅ Created ARCHITECTURE_SUMMARY_TEAM025.md as reference

**Current System State (THE 4 BINARIES):**
1. **queen-rbee** (HTTP daemon :8080) - M1 ❌ NOT BUILT
2. **llm-worker-rbee** (HTTP daemon :8001+) - M0 ✅ WORKING
3. **rbee-hive** (HTTP daemon :8080) - M1 ❌ NOT BUILT (CLI exists, needs daemon mode)
4. **rbee-keeper** (CLI) - M0 ✅ WORKING

**M0 Status:** 2 of 4 binaries complete (workers + CLI)

**CRITICAL CORRECTION FROM TEAM-025:**
- rbee-hive MUST be an HTTP daemon (per MVP Phase 2, lines 42-55)
- rbee-keeper calls HTTP APIs (NOT SSH command execution)
- SSH is ONLY for starting daemons remotely (nice-to-have)

---

## Your Mission: Implement test-001 MVP

**Source of Truth:** `bin/.specs/.gherkin/test-001-mvp.md`

**Test Objective:**
From `blep`, run inference on `mac` using:
```bash
rbee-keeper infer --node mac --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --prompt "write a short story" --max-tokens 20 --temperature 0.7
```

**Success Criteria:**
- Model downloads with progress bar
- Worker starts and loads model
- Inference streams tokens in real-time
- Worker auto-shuts down after 5 minutes idle
- Total latency < 30 seconds (cold start)

---

## The 8 Phases (Happy Path)

### Phase 1: Worker Registry Check
**rbee-keeper** queries local SQLite registry for existing worker.

**Status:** ❌ NOT IMPLEMENTED
- Need SQLite registry in rbee-keeper
- Schema: workers table with (id, node, url, model_ref, state, last_health_check_unix)

---

### Phase 2: Pool Preflight
**rbee-keeper** → **rbee-hive** HTTP health check.

```
GET http://mac.home.arpa:8080/v1/health
Authorization: Bearer <api_key>
```

**Status:** ❌ NOT IMPLEMENTED
- rbee-hive needs HTTP server
- rbee-hive needs /v1/health endpoint
- rbee-keeper needs HTTP client

---

### Phase 3: Model Provisioning
**rbee-hive** checks catalog, downloads if needed, streams progress via SSE.

```
GET http://mac.home.arpa:8080/v1/models/download/progress?id=<download_id>
```

**Status:** ⚠️ PARTIALLY IMPLEMENTED
- rbee-hive CLI can download models (hf CLI)
- Need HTTP API endpoint: POST /v1/models/download
- Need SSE streaming for progress
- Need model catalog SQLite

---

### Phase 4: Worker Preflight
**rbee-hive** checks RAM and backend availability.

**Status:** ❌ NOT IMPLEMENTED
- Need RAM check logic
- Need backend detection (Metal/CUDA/CPU)

---

### Phase 5: Worker Startup
**rbee-hive** spawns worker, worker sends ready callback.

```bash
llm-worker-rbee \
  --model /models/tinyllama-q4.gguf \
  --backend metal \
  --device 0 \
  --port 8081 \
  --api-key <worker_api_key>
```

**Worker callback:**
```
POST http://mac.home.arpa:8080/v1/workers/ready
{
  "worker_id": "worker-abc123",
  "url": "http://mac.home.arpa:8081",
  "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "backend": "metal",
  "device": 0
}
```

**Status:** ⚠️ PARTIALLY IMPLEMENTED
- llm-worker-rbee exists and works
- Need HTTP API: POST /v1/workers/spawn
- Need worker ready callback endpoint: POST /v1/workers/ready
- Need background health monitoring (every 30s)
- Need idle timeout enforcement (5 minutes)

---

### Phase 6: Worker Registration
**rbee-keeper** updates local SQLite registry.

**Status:** ❌ NOT IMPLEMENTED
- Need SQLite worker registry in rbee-keeper

---

### Phase 7: Worker Health Check
**rbee-keeper** polls worker until ready.

```
GET http://mac.home.arpa:8081/v1/ready
```

**Status:** ⚠️ PARTIALLY IMPLEMENTED
- llm-worker-rbee has /v1/ready endpoint (verify)
- Need loading progress SSE stream
- Need polling logic in rbee-keeper

---

### Phase 8: Inference Execution
**rbee-keeper** sends inference request, streams tokens.

```
POST http://mac.home.arpa:8081/v1/inference
{
  "prompt": "write a short story",
  "max_tokens": 20,
  "temperature": 0.7,
  "stream": true
}
```

**Status:** ⚠️ PARTIALLY IMPLEMENTED
- llm-worker-rbee has inference endpoint (verify)
- Need SSE streaming in rbee-keeper
- Need token display in CLI

---

## What Needs to Be Built

### Priority 1: rbee-hive HTTP Daemon (M1)

**Location:** `bin/rbee-hive/src/`

**Current State:**
- CLI exists at `bin/rbee-hive/`
- Can download models
- Can spawn workers (direct process spawn)

**What to Add:**

#### 1.1 HTTP Server
```rust
// bin/rbee-hive/src/daemon.rs
use axum::{Router, routing::get, routing::post};

pub async fn run_daemon(config: Config) -> Result<()> {
    let app = Router::new()
        .route("/v1/health", get(health_handler))
        .route("/v1/models/download", post(download_model_handler))
        .route("/v1/models/download/progress", get(download_progress_handler))
        .route("/v1/workers/spawn", post(spawn_worker_handler))
        .route("/v1/workers/ready", post(worker_ready_handler))
        .route("/v1/workers/list", get(list_workers_handler));
    
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

#### 1.2 Health Monitoring Loop
```rust
// bin/rbee-hive/src/monitor.rs
pub async fn health_monitor_loop(registry: Arc<WorkerRegistry>) {
    let mut interval = tokio::time::interval(Duration::from_secs(30));
    loop {
        interval.tick().await;
        for worker in registry.list_workers().await {
            if let Err(e) = check_worker_health(&worker).await {
                eprintln!("Worker {} unhealthy: {}", worker.id, e);
                registry.mark_unhealthy(&worker.id).await;
            }
        }
    }
}
```

#### 1.3 Idle Timeout Enforcement
```rust
// bin/rbee-hive/src/timeout.rs
pub async fn idle_timeout_loop(registry: Arc<WorkerRegistry>) {
    let mut interval = tokio::time::interval(Duration::from_secs(60));
    loop {
        interval.tick().await;
        let now = SystemTime::now();
        for worker in registry.list_workers().await {
            if worker.state == "idle" {
                let idle_duration = now.duration_since(worker.last_activity)?;
                if idle_duration > Duration::from_secs(300) { // 5 minutes
                    shutdown_worker(&worker).await?;
                    registry.remove(&worker.id).await;
                }
            }
        }
    }
}
```

#### 1.4 Worker Registry
```rust
// bin/rbee-hive/src/registry.rs
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
}

pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub state: String, // "loading", "idle", "busy"
    pub last_activity: SystemTime,
    pub slots_total: u32,
    pub slots_available: u32,
}
```

**Estimated Time:** 1-2 weeks

---

### Priority 2: rbee-keeper HTTP Client (M0)

**Location:** `bin/rbee-keeper/src/`

**What to Add:**

#### 2.1 HTTP Client for rbee-hive
```rust
// bin/rbee-keeper/src/pool_client.rs
pub struct PoolClient {
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

impl PoolClient {
    pub async fn health_check(&self) -> Result<HealthResponse> {
        let response = self.client
            .get(&format!("{}/v1/health", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await?;
        Ok(response.json().await?)
    }
    
    pub async fn spawn_worker(&self, req: SpawnWorkerRequest) -> Result<WorkerInfo> {
        let response = self.client
            .post(&format!("{}/v1/workers/spawn", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .await?;
        Ok(response.json().await?)
    }
}
```

#### 2.2 SQLite Worker Registry
```rust
// bin/rbee-keeper/src/registry.rs
pub struct WorkerRegistry {
    db: SqliteConnection,
}

impl WorkerRegistry {
    pub async fn find_worker(&self, node: &str, model_ref: &str) -> Option<WorkerInfo> {
        sqlx::query_as!(
            WorkerInfo,
            "SELECT * FROM workers 
             WHERE node = ? AND model_ref = ? AND state IN ('idle', 'ready')
             AND last_health_check_unix > ?",
            node,
            model_ref,
            (SystemTime::now() - Duration::from_secs(60)).as_secs()
        )
        .fetch_optional(&self.db)
        .await
        .ok()?
    }
}
```

#### 2.3 SSE Streaming
```rust
// bin/rbee-keeper/src/stream.rs
pub async fn stream_tokens(worker_url: &str, request: InferenceRequest) -> Result<()> {
    let client = reqwest::Client::new();
    let mut stream = client
        .post(&format!("{}/v1/inference", worker_url))
        .json(&request)
        .send()
        .await?
        .bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let event: TokenEvent = serde_json::from_slice(&chunk)?;
        print!("{}", event.token);
        stdout().flush()?;
    }
    println!();
    Ok(())
}
```

**Estimated Time:** 1 week

---

### Priority 3: Integration Testing

**Create:** `bin/.specs/.gherkin/test-001-mvp-implementation.md`

**Test Script:**
```bash
#!/usr/bin/env bash
# test-001-mvp.sh

set -euo pipefail

echo "=== Test-001 MVP: Cross-Node Inference ==="

# Step 1: Start rbee-hive daemon on mac
ssh mac.home.arpa "cd ~/Projects/llama-orch && cargo run --bin rbee-hive -- daemon start" &
sleep 5

# Step 2: Verify health
rbee-keeper pool health --host mac
echo "✅ rbee-hive is healthy"

# Step 3: Run inference
rbee-keeper infer --node mac \
  --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --prompt "write a short story" \
  --max-tokens 20 \
  --temperature 0.7

echo "✅ Test-001 MVP PASSED"
```

**Estimated Time:** 3-5 days

---

## Implementation Plan

### Week 1: rbee-hive HTTP Daemon
- [ ] Day 1-2: HTTP server skeleton + health endpoint
- [ ] Day 3-4: Worker spawn endpoint + registry
- [ ] Day 5: Worker ready callback endpoint

### Week 2: rbee-hive Background Tasks
- [ ] Day 1-2: Health monitoring loop (30s)
- [ ] Day 3-4: Idle timeout enforcement (5 min)
- [ ] Day 5: Model download endpoint + SSE progress

### Week 3: rbee-keeper HTTP Client
- [ ] Day 1-2: HTTP client for rbee-hive
- [ ] Day 3-4: SQLite worker registry
- [ ] Day 5: SSE streaming for tokens

### Week 4: Integration & Testing
- [ ] Day 1-2: End-to-end test script
- [ ] Day 3-4: Edge case testing (10 edge cases from MVP)
- [ ] Day 5: Documentation + handoff

---

## Critical Files for TEAM-026

### Must Read (in order):
1. `bin/.specs/.gherkin/test-001-mvp.md` - THE source of truth
2. `bin/.specs/ARCHITECTURE_SUMMARY_TEAM025.md` - Architecture reference
3. `bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md` - Communication flow

### Reference When Needed:
- `bin/.specs/00_llama-orch.md` - Full system spec
- `bin/.specs/01_M0_worker_orcd.md` - Worker spec
- `.windsurf/rules/candled-rules.md` - Coding standards

### Ignore (Outdated):
- `ARCHITECTURE_DECISION_NO_POOL_DAEMON.md` - WRONG (marked as such)
- `CONTROL_PLANE_ARCHITECTURE_DECISION.md` - WRONG (marked as such)

---

## Known Issues & Gotchas

### Issue 1: rbee-hive Port Conflict
**Problem:** rbee-hive and queen-rbee both want :8080  
**Solution:** Use :9200 for rbee-hive (or configure per-pool)

### Issue 2: Worker Model Loading
**Problem:** Model loading can take 30+ seconds  
**Solution:** Stream loading progress via SSE, show progress bar

### Issue 3: SSH vs HTTP Confusion
**Problem:** Previous teams thought SSH was for pool operations  
**Solution:** rbee-keeper calls HTTP APIs. SSH only for starting daemons.

---

## Edge Cases to Implement

From test-001-mvp.md (lines 290-610):

1. **EC1: Connection Timeout** - Retry with exponential backoff
2. **EC2: Model Download Failure** - Retry up to 6 times
3. **EC3: Insufficient VRAM** - Preflight check, helpful error
4. **EC4: Worker Crash** - Save partial results
5. **EC5: Client Cancellation (Ctrl+C)** - Send DELETE to worker
6. **EC6: Queue Full** - Return 503, retry suggestion
7. **EC7: Model Loading Timeout** - 5 minute timeout
8. **EC8: Version Mismatch** - Check versions, helpful error
9. **EC9: Invalid API Key** - Return 401
10. **EC10: Idle Timeout** - Auto-shutdown after 5 min

**Each edge case has detailed requirements in test-001-mvp.md.**

---

## Dependencies & Prerequisites

### System Requirements:
- ✅ Rust toolchain (installed)
- ✅ `hf` CLI (installed)
- ✅ SSH access to pools (configured)
- ✅ llm-worker-rbee (working)
- ✅ rbee-keeper CLI (working)

### Rust Crates Needed:
- `axum` - HTTP server
- `tokio` - Async runtime
- `sqlx` - SQLite
- `reqwest` - HTTP client
- `eventsource-stream` - SSE streaming
- `serde_json` - JSON serialization

### Models:
- ✅ qwen-0.5b downloaded (943MB)
- ⏳ tinyllama (need to download for testing)

---

## Testing Checklist

### Before Starting:
- [ ] Read test-001-mvp.md completely
- [ ] Read ARCHITECTURE_SUMMARY_TEAM025.md
- [ ] Verify llm-worker-rbee works
- [ ] Verify rbee-keeper CLI works

### After rbee-hive HTTP Server:
- [ ] GET /v1/health returns 200
- [ ] POST /v1/workers/spawn spawns process
- [ ] POST /v1/workers/ready updates registry
- [ ] Health monitoring loop runs every 30s
- [ ] Idle timeout shuts down workers after 5 min

### After rbee-keeper HTTP Client:
- [ ] Can call rbee-hive health endpoint
- [ ] Can spawn worker via HTTP
- [ ] Can stream tokens via SSE
- [ ] SQLite registry works

### End-to-End:
- [ ] Full test-001 flow works
- [ ] All 10 edge cases handled
- [ ] Latency < 30 seconds (cold start)
- [ ] Worker auto-shuts down after idle

---

## Success Criteria for TEAM-026

### Minimum (MVP Happy Path):
- [ ] rbee-hive HTTP daemon running
- [ ] rbee-keeper can spawn worker via HTTP
- [ ] Inference streams tokens
- [ ] Worker auto-shuts down after 5 min idle

### Target (MVP + Edge Cases):
- [ ] All 8 phases implemented
- [ ] At least 5 edge cases handled
- [ ] Test script passes
- [ ] Documentation complete

### Stretch (Production Ready):
- [ ] All 10 edge cases handled
- [ ] Comprehensive error messages
- [ ] Retry logic with backoff
- [ ] Metrics & logging

---

## Communication Protocol

### When to Ask User:
- ✅ Ask if you find blocking issues
- ✅ Ask if MVP requirements are unclear
- ✅ Ask before making architectural changes
- ❌ Don't ask about implementation details (follow MVP)

### When to Update Docs:
- ✅ Update this handoff when complete
- ✅ Create TEAM_026_COMPLETION_SUMMARY.md
- ✅ Create TEAM_027_HANDOFF.md
- ✅ Update test-001-mvp.md with implementation notes

### When to Stop:
- 🛑 If MVP requirements are contradictory (ask user)
- 🛑 If test-001 takes >4 weeks (re-evaluate)
- 🛑 If fundamental blocker found (escalate)

---

## Final Notes from TEAM-025

**What Went Well:**
- ✅ Corrected architecture misunderstandings
- ✅ Applied rebranding successfully
- ✅ Clarified HTTP daemon requirements
- ✅ Created clear reference docs

**What Was Hard:**
- 😓 TEAM-024 had wrong assumptions (SSH for pool ops)
- 😓 Multiple contradicting architecture docs
- 😓 Rebranding touched many files

**Advice for TEAM-026:**
- 📖 Read test-001-mvp.md FIRST (it's the source of truth)
- 🎯 Follow the 8 phases step-by-step
- 🧪 Test after each phase
- 📝 Document as you go
- 💬 Ask if MVP is unclear

**Remember:**
- The MVP is well-defined!
- The architecture is correct now!
- The binaries are clear!
- Just implement the 8 phases!

**Key Insight:**
- rbee-hive is an HTTP daemon (not SSH-controlled)
- rbee-keeper calls HTTP APIs (not SSH commands)
- SSH is only for starting daemons remotely

---

**Signed:** TEAM-025  
**Date:** 2025-10-09T23:05:00+02:00  
**Status:** Handoff complete, architecture aligned with MVP  
**Next Team:** TEAM-026 - Implement test-001 MVP! 🚀
