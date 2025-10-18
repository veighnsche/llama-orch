# HANDOFF TO TEAM-043 (COMPLETE)

**HISTORICAL NOTE (TEAM-054):** This document describes the architecture BEFORE TEAM-037/TEAM-038
introduced queen-rbee orchestration. At that time, rbee-hive used port 8080 and rbee-keeper
connected directly to rbee-hive. This architecture is NO LONGER VALID.

**Current architecture:** queen-rbee (8080) → rbee-hive (9200) → workers (8001+)
**See:** `bin/.specs/.gherkin/test-001.md` for current normative spec.

**From:** TEAM-042  
**To:** TEAM-043  
**Date:** 2025-10-10  
**Status:** 🟢 COMPLETE SURVEY - READY FOR IMPLEMENTATION

---

## Executive Summary

After **comprehensive investigation** of both `test-harness/bdd/src/steps/` and `bin/`, here's the complete picture:

### 🎯 BDD-First Principle

**CRITICAL: BDD tests are the specification. The bin/ implementation must conform to what the BDD tests expect.**

When there's a mismatch:
1. **BDD is correct** - It defines the contract
2. **bin/ must change** - Implementation follows the spec
3. **Don't skip tests** - Fix the implementation instead

### ✅ What Exists
1. **BDD step definitions** - 330+ steps defining the complete specification
2. **rbee-hive** - Pool manager (needs adjustments to match BDD)
3. **llm-worker-rbee** - Worker (needs adjustments to match BDD)
4. **rbee-keeper infer** - Inference orchestration (needs adjustments)
5. **model-catalog** - SQLite model tracking (matches BDD)

### ❌ What's Missing in bin/
1. **queen-rbee** - BDD expects it, not implemented
2. **rbee-keeper setup** - BDD expects it, not implemented
3. **Worker `/v1/ready` endpoint** - BDD expects it, uses `/v1/loading/progress` instead
4. **Proper architecture** - BDD expects rbee-keeper → queen-rbee → rbee-hive, bin/ has rbee-keeper → rbee-hive

### 🎯 Your Mission
**Implement the BDD step definitions AND fix bin/ to match what the tests expect.**

---

## Complete Step Definition Inventory

### Files with Step Definitions (17 files, 330+ steps)

1. **`background.rs`** (6 steps) - Topology setup
2. **`beehive_registry.rs`** (19 steps) - Registry operations (TEAM-042 mocked)
3. **`cli_commands.rs`** (24 steps) - CLI command execution
4. **`edge_cases.rs`** (37 steps) - Error scenarios
5. **`error_responses.rs`** (6 steps) - Error handling
6. **`gguf.rs`** (20 steps) - GGUF validation
7. **`happy_path.rs`** (42 steps) - Main flow (TEAM-042 mocked)
8. **`inference_execution.rs`** (13 steps) - Inference execution
9. **`lifecycle.rs`** (64 steps) - Worker lifecycle
10. **`model_provisioning.rs`** (24 steps) - Model download/catalog
11. **`pool_preflight.rs`** (16 steps) - Health checks
12. **`registry.rs`** (18 steps) - Worker registry
13. **`worker_health.rs`** (14 steps) - Worker health monitoring
14. **`worker_preflight.rs`** (20 steps) - Worker preflight checks
15. **`worker_registration.rs`** (3 steps) - Worker registration
16. **`worker_startup.rs`** (13 steps) - Worker startup
17. **`world.rs`** - World state (not steps)

**Total: ~330 step definitions**

---

## What's Actually Implemented in `bin/`

### 1. rbee-hive - COMPLETE ✅

**HTTP Server:**
- Port: 8080 (configurable)
- Framework: Axum
- State: In-memory worker registry + SQLite model catalog

**Endpoints:**
```
GET  /v1/health                     → HealthResponse { status, version, api_version }
POST /v1/workers/spawn              → SpawnWorkerResponse { worker_id, url, state }
POST /v1/workers/ready              → WorkerReadyResponse { message }
GET  /v1/workers/list               → ListWorkersResponse { workers: Vec<WorkerInfo> }
POST /v1/models/download            → DownloadModelResponse { download_id, local_path }
GET  /v1/models/download/progress   → SSE stream (download progress)
```

**Worker Spawning Flow:**
1. Check model catalog (SQLite at `~/.rbee/models.db`)
2. Download model if not found (HuggingFace)
3. Register model in catalog
4. Spawn worker: `llm-worker-rbee --worker-id X --model PATH --port 8081 --callback-url http://localhost:8080/v1/workers/ready`
5. Register worker in in-memory registry (state: loading)
6. Return worker URL

**Worker Registry:**
```rust
pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub state: WorkerState,  // Loading, Idle, Busy
    pub last_activity: SystemTime,
    pub slots_total: u32,
    pub slots_available: u32,
}
```

**Model Catalog (SQLite):**
```sql
CREATE TABLE models (
    reference TEXT NOT NULL,
    provider TEXT NOT NULL,
    local_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    downloaded_at INTEGER NOT NULL,
    PRIMARY KEY (reference, provider)
)
```

### 2. llm-worker-rbee - COMPLETE ✅

**CLI Args:**
```bash
llm-worker-rbee \
  --worker-id worker-abc123 \
  --model /models/tinyllama.gguf \
  --port 8081 \
  --callback-url http://localhost:8080/v1/workers/ready
```

**Startup Flow:**
1. Load model (GGUF format)
2. Send ready callback to pool manager:
   ```json
   POST {callback_url}
   {
     "worker_id": "worker-abc123",
     "vram_bytes": 669000000,
     "uri": "http://localhost:8081"
   }
   ```
3. Start HTTP server
4. Run forever (until killed)

**Endpoints:**
```
GET  /health                  → { status: "ok" }
POST /v1/inference            → SSE stream (token streaming)
GET  /v1/loading/progress     → SSE stream (loading progress)
```

**Inference SSE Events:**
```json
data: {"type":"token","t":"Once","i":0}
data: {"type":"token","t":" upon","i":1}
...
data: {"type":"end","tokens_out":20,"decode_time_ms":150}
data: [DONE]
```

**Loading Progress SSE Events:**
```json
data: {"stage":"loading_to_vram","layers_loaded":24,"layers_total":32,"vram_mb":4096}
data: {"stage":"ready"}
data: [DONE]
```

### 3. rbee-keeper - PARTIAL ✅

**Implemented Commands:**
- ✅ `infer` - Full 8-phase inference flow
- ✅ `pool` - Pool management (models, worker, git, status)
- ✅ `install` - Binary installation
- ❌ `setup` - NOT IMPLEMENTED

**Infer Command Flow:**
```rust
// Phase 1: Skipped (ephemeral mode)
// Phase 2: Pool preflight
let health = pool_client.health_check().await?;  // GET /v1/health

// Phase 3-5: Spawn worker
let worker = pool_client.spawn_worker(request).await?;  // POST /v1/workers/spawn

// Phase 7: Wait for ready
loop {
    let response = client.get(&format!("{}/v1/ready", worker.url)).send().await?;
    if response.json::<ReadyResponse>().await?.ready { break; }
}

// Phase 8: Execute inference
let response = client.post(&format!("{}/v1/inference", worker.url))
    .json(&request).send().await?;
// Stream SSE tokens to stdout
```

**Pool Client:**
```rust
pub struct PoolClient {
    base_url: String,  // e.g., "http://localhost:8080"
    api_key: String,
    client: reqwest::Client,
}

impl PoolClient {
    pub async fn health_check(&self) -> Result<HealthResponse>;
    pub async fn spawn_worker(&self, request: SpawnWorkerRequest) -> Result<SpawnWorkerResponse>;
}
```

### 4. model-catalog - COMPLETE ✅

**Shared Crate:**
```rust
pub struct ModelCatalog {
    db_path: String,  // e.g., "~/.rbee/models.db"
}

impl ModelCatalog {
    pub async fn init(&self) -> Result<()>;
    pub async fn find_model(&self, reference: &str, provider: &str) -> Result<Option<ModelInfo>>;
    pub async fn register_model(&self, model: &ModelInfo) -> Result<()>;
    pub async fn remove_model(&self, reference: &str, provider: &str) -> Result<()>;
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>>;
}
```

---

## Critical Gaps in bin/ Implementation

### 🚨 BDD-First Principle: Fix bin/ to Match BDD

**The BDD tests define the contract. When bin/ doesn't match, bin/ is wrong.**

### 1. Worker Ready Endpoint Missing ❌

**BDD Specification (CORRECT):**
```
GET /v1/ready → { ready: true, state: "idle" }
```

**Current bin/ Implementation (WRONG):**
```
GET /v1/loading/progress → SSE stream with loading events
```

**Required Fix:**
Add `/v1/ready` endpoint to `bin/llm-worker-rbee/src/http/routes.rs`:
```rust
.route("/v1/ready", get(ready::handle_ready))
```

### 2. queen-rbee Not Implemented ❌

**BDD Specification (CORRECT):**
```
rbee-keeper → queen-rbee → rbee-hive → worker
```

**Current bin/ Implementation (WRONG):**
```
rbee-keeper → rbee-hive → worker (queen-rbee skipped)
```

**Required Fix:**
Implement `bin/queen-rbee/` with:
- HTTP server on port 8080
- rbee-hive registry (SQLite at `~/.rbee/beehives.db`)
- SSH connection management
- Registry API endpoints (`/v2/registry/beehives/*`)

### 3. Setup Commands Missing ❌

**BDD Specification (CORRECT):**
```
rbee-keeper setup add-node --name mac ...
rbee-keeper setup list-nodes
rbee-keeper setup remove-node --name mac
rbee-keeper setup install --node mac
```

**Current bin/ Implementation (WRONG):**
```
❌ No setup subcommand in CLI
```

**Required Fix:**
Add setup commands to `bin/rbee-keeper/src/cli.rs` and implement handlers.

---

## Implementation Strategy for TEAM-043

### 🎯 BDD-First Approach

**Remember: The BDD tests are the specification. Implement to make tests pass.**

### Phase 1: Implement Step Definitions (Priority 1) 🎯

**Start by implementing the step definitions to execute real code:**

1. **Start rbee-hive:**
```rust
#[given(expr = "rbee-hive is running at {string}")]
pub async fn given_rbee_hive_running(world: &mut World, url: String) {
    let mut child = tokio::process::Command::new("./target/debug/rbee-hive")
        .arg("daemon")
        .arg("--addr")
        .arg("0.0.0.0:8080")
        .spawn()
        .expect("Failed to start rbee-hive");
    
    // Wait for ready
    for _ in 0..30 {
        if reqwest::get("http://localhost:8080/v1/health").await.is_ok() {
            world.rbee_hive_process = Some(child);
            return;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    panic!("rbee-hive failed to start");
}
```

2. **Execute Commands:**
```rust
#[when(expr = "I run:")]
pub async fn when_i_run_command(world: &mut World, step: &cucumber::gherkin::Step) {
    let command = step.docstring.as_ref().unwrap().trim();
    let parts: Vec<&str> = command.split_whitespace().collect();
    
    let output = tokio::process::Command::new("./target/debug/rbee-keeper")
        .args(&parts[1..])
        .output()
        .await
        .expect("Failed to execute");
    
    world.last_stdout = String::from_utf8_lossy(&output.stdout).to_string();
    world.last_exit_code = output.status.code();
}
```

3. **Verify HTTP Responses:**
```rust
#[then(expr = "the response status is {int}")]
pub async fn then_response_status(world: &mut World, status: u16) {
    let client = reqwest::Client::new();
    let response = client.get("http://localhost:8080/v1/health").send().await.unwrap();
    assert_eq!(response.status().as_u16(), status);
}
```

4. **Verify Worker Registry:**
```rust
#[then(expr = "rbee-hive registers the worker in the in-memory registry")]
pub async fn then_register_worker(world: &mut World) {
    let client = reqwest::Client::new();
    let response = client.get("http://localhost:8080/v1/workers/list").send().await.unwrap();
    let workers: serde_json::Value = response.json().await.unwrap();
    assert!(!workers["workers"].as_array().unwrap().is_empty());
}
```

5. **Verify Model Catalog:**
```rust
#[then(expr = "rbee-hive registers the model in SQLite catalog with local_path {string}")]
pub async fn then_register_model_in_catalog(world: &mut World, local_path: String) {
    let db_path = shellexpand::tilde("~/.rbee/models.db");
    let conn = rusqlite::Connection::open(db_path.as_ref()).unwrap();
    let mut stmt = conn.prepare("SELECT 1 FROM models WHERE local_path = ?").unwrap();
    assert!(stmt.exists([&local_path]).unwrap());
}
```

### Phase 2: Fix bin/ Implementation (Priority 2) 🔧

**When tests fail because bin/ doesn't match BDD, fix bin/:**

1. **Add `/v1/ready` endpoint to worker:**
```rust
// In bin/llm-worker-rbee/src/http/routes.rs
.route("/v1/ready", get(ready::handle_ready))

// In bin/llm-worker-rbee/src/http/ready.rs
pub async fn handle_ready<B: InferenceBackend>(
    State(backend): State<Arc<Mutex<B>>>,
) -> Json<ReadyResponse> {
    let backend = backend.lock().await;
    Json(ReadyResponse {
        ready: backend.is_ready(),
        state: if backend.is_ready() { "idle" } else { "loading" },
    })
}
```

2. **Implement queen-rbee:**
See earlier section "Required Fix" for complete implementation guide.

3. **Implement setup commands:**
```rust
// In bin/rbee-keeper/src/cli.rs
pub enum Commands {
    // ... existing ...
    Setup {
        #[command(subcommand)]
        action: SetupAction,
    },
}
```

### Phase 3: Iterate Until All Tests Pass ✅

**The goal: All BDD scenarios pass with real binaries.**

1. Run tests: `cargo run --bin bdd-runner -- --tags @setup`
2. Test fails → Identify what's missing in bin/
3. Implement missing functionality in bin/
4. Re-run tests
5. Repeat until green

---

## Test Execution Guide

### Smoke Test (Verify MVP Works)

```bash
# Terminal 1: Start rbee-hive
cd bin/rbee-hive
cargo run -- daemon --addr 0.0.0.0:8080

# Terminal 2: Run inference
cd bin/rbee-keeper
cargo run -- infer \
  --node localhost \
  --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --prompt "hello world" \
  --max-tokens 20

# Expected: Tokens stream to stdout
```

### Run BDD Tests

```bash
# Run all tests
cd test-harness/bdd
cargo run --bin bdd-runner

# Run specific tag
cargo run --bin bdd-runner -- --tags @happy

# Run specific scenario
cargo run --bin bdd-runner -- --name "Pool preflight check"
```

### Debug Failed Steps

```bash
# Enable debug logging
RUST_LOG=debug cargo run --bin bdd-runner -- --tags @happy

# Check what's actually running
ps aux | grep rbee
lsof -i :8080
lsof -i :8081
```

---

## Summary

### 🎯 BDD-First Principle (CRITICAL)

**The BDD tests ARE the specification. bin/ must conform to BDD, not the other way around.**

### What You Have
- ✅ 330+ step definitions defining the complete specification
- ⚠️ Partial rbee-hive implementation (needs adjustments)
- ⚠️ Partial llm-worker-rbee implementation (needs `/v1/ready` endpoint)
- ⚠️ Partial rbee-keeper implementation (needs setup commands)
- ❌ queen-rbee not implemented (BDD requires it)

### What You Need to Do
1. **Implement step definitions** (replace stubs with real execution)
2. **Fix bin/ to match BDD** (add missing endpoints, commands, binaries)
3. **Run tests iteratively** until all pass
4. **Don't skip tests** - implement what's missing instead

### What You Should NOT Do
- ❌ Skip BDD scenarios because bin/ doesn't match
- ❌ Change BDD tests to match bin/
- ❌ Accept "good enough" - make all tests pass

### Success Criteria
- [ ] All 330+ step definitions implemented (no stubs)
- [ ] All BDD scenarios pass with real binaries
- [ ] bin/ fully implements what BDD specifies
- [ ] queen-rbee implemented and working
- [ ] rbee-keeper setup commands implemented
- [ ] Worker `/v1/ready` endpoint implemented
- [ ] No mocks - everything uses real execution

---

**Good luck, TEAM-043! Remember: BDD is the spec, bin/ must follow it. Make the tests pass!** 🚀
