# HANDOFF TO TEAM-043 (FINAL)

**HISTORICAL NOTE (TEAM-054):** This document describes the architecture BEFORE TEAM-037/TEAM-038
introduced queen-rbee orchestration. At that time, rbee-hive used port 8080 and rbee-keeper
connected directly to rbee-hive. This architecture is NO LONGER VALID.

**Current architecture:** queen-rbee (8080) → rbee-hive (9200) → workers (8001+)
**See:** `bin/.specs/.gherkin/test-001.md` for current normative spec.: Wire BDD Tests to Real Binaries (FINAL - AFTER DEEP SURVEY)

**From:** TEAM-042  
**To:** TEAM-043  
**Date:** 2025-10-10  
**Status:** 🟢 DEEP SURVEY COMPLETE - ACCURATE IMPLEMENTATION GUIDE

---

## Executive Summary

After deep investigation of `bin/`, here's the **accurate** state:

### ✅ What's FULLY Working (Can Use Immediately)
1. **rbee-hive** - Complete pool manager with HTTP server, worker spawning, model provisioning
2. **llm-worker-rbee** - Complete inference worker with SSE streaming, ready callbacks
3. **rbee-keeper infer** - Complete 8-phase inference flow (connects to rbee-hive)
4. **model-catalog** - Complete SQLite model tracking

### ❌ What's Missing (Must Implement)
1. **queen-rbee** - Just a scaffold, needs full implementation
2. **rbee-keeper setup** - No setup commands exist

### 🎯 **CRITICAL INSIGHT**
The current architecture is **Mode 1: Ephemeral** (rbee-keeper → rbee-hive directly).
**queen-rbee is NOT used in current MVP!** It's a future M1+ feature.

---

## Current Architecture (What Actually Works)

### Mode 1: Ephemeral (MVP - Currently Implemented)

```
Developer runs: rbee-keeper infer --node localhost --model tinyllama --prompt "hello"

Flow:
1. rbee-keeper connects DIRECTLY to rbee-hive at localhost:8080
2. rbee-hive spawns llm-worker-rbee process
3. Worker loads model, starts HTTP server on port 8081
4. Worker sends ready callback to rbee-hive: POST /v1/workers/ready
5. rbee-hive registers worker in in-memory registry
6. rbee-hive returns worker URL to rbee-keeper
7. rbee-keeper sends inference request DIRECTLY to worker: POST /v1/inference
8. Worker streams tokens via SSE back to rbee-keeper
9. User gets result
```

**Key Points:**
- ✅ **rbee-hive** is the pool manager (fully implemented)
- ✅ **llm-worker-rbee** is the worker (fully implemented)
- ✅ **rbee-keeper infer** orchestrates the flow (fully implemented)
- ❌ **queen-rbee** is NOT used in current MVP (future M1+ feature)

---

## What's Actually Implemented in `bin/`

### 1. `bin/rbee-hive/` - FULLY IMPLEMENTED ✅

**HTTP Server** (`src/commands/daemon.rs`):
- ✅ Starts on port 8080 (configurable)
- ✅ In-memory worker registry (ephemeral)
- ✅ SQLite model catalog at `~/.rbee/models.db`
- ✅ Health monitor loop (30s interval)
- ✅ Idle timeout loop (5min threshold)
- ✅ Graceful shutdown with worker cleanup

**HTTP Endpoints** (`src/http/`):
- ✅ `GET /v1/health` - Health check
- ✅ `POST /v1/workers/spawn` - Spawn worker (with model provisioning!)
- ✅ `POST /v1/workers/ready` - Worker ready callback
- ✅ `GET /v1/workers/list` - List all workers
- ✅ `POST /v1/models/download` - Download model
- ✅ `GET /v1/models/download/progress` - SSE progress stream

**Worker Spawning** (`src/http/workers.rs`):
- ✅ Checks model catalog (SQLite)
- ✅ Downloads model if not found (HuggingFace)
- ✅ Registers model in catalog
- ✅ Spawns worker process: `llm-worker-rbee --worker-id X --model PATH --port 8081 --callback-url http://localhost:8080/v1/workers/ready`
- ✅ Registers worker in in-memory registry (state: loading)
- ✅ Returns worker URL to caller

**Worker Registry** (`src/registry.rs`):
- ✅ In-memory `HashMap<String, WorkerInfo>`
- ✅ Thread-safe with `Arc<RwLock>`
- ✅ Methods: register, update_state, get, list, remove, find_idle_worker
- ✅ Comprehensive unit tests

**Model Provisioner** (`src/provisioner/`):
- ✅ Downloads models from HuggingFace
- ✅ Tracks download progress
- ✅ Integrates with model catalog

**Shutdown** (`src/commands/daemon.rs`):
- ✅ Catches SIGTERM
- ✅ Sends `POST /v1/shutdown` to all workers
- ✅ Clears registry
- ✅ Exits gracefully

### 2. `bin/llm-worker-rbee/` - FULLY IMPLEMENTED ✅

**Main Binary** (`src/main.rs`):
- ✅ CLI args: `--worker-id`, `--model`, `--port`, `--callback-url`
- ✅ Loads model (GGUF format)
- ✅ Sends ready callback to pool manager
- ✅ Starts HTTP server
- ✅ Runs forever (until killed)

**Ready Callback** (`src/common/startup.rs`):
- ✅ Sends `POST` to callback URL with:
  ```json
  {
    "worker_id": "worker-abc123",
    "vram_bytes": 669000000,
    "uri": "http://localhost:8081"
  }
  ```

**HTTP Endpoints** (`src/http/`):
- ✅ `GET /health` - Health check
- ✅ `POST /v1/inference` - Execute inference (SSE streaming)
- ✅ `GET /v1/loading/progress` - Model loading progress (SSE)

**Inference** (`src/http/execute.rs`):
- ✅ Validates request
- ✅ Executes inference with Candle backend
- ✅ Streams tokens via SSE
- ✅ Sends `[DONE]` marker at end (OpenAI compatible)
- ✅ Narration events for progress

**Shutdown**:
- ❌ No `/v1/shutdown` endpoint implemented yet
- ✅ Worker exits when killed by pool manager

### 3. `bin/rbee-keeper/` - PARTIALLY IMPLEMENTED ⚠️

**Infer Command** (`src/commands/infer.rs`) - ✅ FULLY IMPLEMENTED:
- ✅ Connects to rbee-hive at `http://{node}.home.arpa:8080`
- ✅ Health check: `GET /v1/health`
- ✅ Spawn worker: `POST /v1/workers/spawn`
- ✅ Wait for ready: polls `GET /v1/ready` on worker
- ✅ Execute inference: `POST /v1/inference` on worker
- ✅ Stream tokens via SSE
- ✅ Display results to stdout
- ✅ Error handling with retries

**Setup Commands** - ❌ NOT IMPLEMENTED:
- ❌ No `setup` subcommand exists
- ❌ No registry management
- ❌ No node configuration

**CLI Structure** (`src/cli.rs`):
```rust
pub enum Commands {
    Infer { /* IMPLEMENTED */ },
    Pool { /* IMPLEMENTED */ },
    Install { /* IMPLEMENTED */ },
    // Setup { /* MISSING */ },
}
```

### 4. `bin/queen-rbee/` - SCAFFOLD ONLY ❌

**Current State**:
- ❌ Just CLI arg parsing
- ❌ No HTTP server
- ❌ No registry module
- ❌ No orchestration logic
- ❌ **NOT USED IN CURRENT MVP**

**Future M1+ Feature**:
- Will manage multiple rbee-hive instances
- Will use SSH to control remote hives
- Will maintain rbee-hive registry (SQLite at `~/.rbee/beehives.db`)
- Will cascade shutdown to all hives

### 5. `bin/shared-crates/model-catalog/` - FULLY IMPLEMENTED ✅

**Model Catalog** (`src/lib.rs`):
- ✅ SQLite at `~/.rbee/models.db`
- ✅ Table: `models` (reference, provider, local_path, size_bytes, downloaded_at)
- ✅ Methods: init, find_model, register_model, remove_model, list_models
- ✅ Comprehensive unit tests
- ✅ Used by rbee-hive for model tracking

---

## What the BDD Tests Should Actually Test

### Current MVP (Mode 1: Ephemeral)

**The tests should verify:**
1. ✅ `rbee-hive` starts and listens on port 8080
2. ✅ `rbee-keeper infer` connects to rbee-hive
3. ✅ rbee-hive spawns worker process
4. ✅ Worker loads model and sends ready callback
5. ✅ rbee-hive registers worker in registry
6. ✅ rbee-keeper sends inference request to worker
7. ✅ Worker streams tokens via SSE
8. ✅ User gets result

**The tests should NOT test:**
- ❌ queen-rbee (not used in MVP)
- ❌ rbee-hive registry (doesn't exist yet, future M1+)
- ❌ SSH connections (not used in MVP)
- ❌ Multi-node orchestration (future M1+)

---

## How to Wire BDD Tests (Correct Approach)

### Step 1: Start rbee-hive

```rust
#[given(expr = "rbee-hive is running at {string}")]
pub async fn given_rbee_hive_running(world: &mut World, url: String) {
    // Start rbee-hive as background process
    let mut child = tokio::process::Command::new("./target/debug/rbee-hive")
        .arg("daemon")
        .arg("--addr")
        .arg("0.0.0.0:8080")
        .spawn()
        .expect("Failed to start rbee-hive");
    
    // Wait for HTTP server to be ready
    for _ in 0..30 {
        if reqwest::get("http://localhost:8080/v1/health").await.is_ok() {
            world.rbee_hive_process = Some(child);
            tracing::info!("✅ rbee-hive started and ready");
            return;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
    
    panic!("rbee-hive failed to start");
}
```

### Step 2: Execute rbee-keeper Commands

```rust
#[when(expr = "I run:")]
pub async fn when_i_run_command(world: &mut World, step: &cucumber::gherkin::Step) {
    let command = step.docstring.as_ref().unwrap().trim();
    
    // Parse command: "rbee-keeper infer --node localhost --model ..."
    let parts: Vec<&str> = command.split_whitespace().collect();
    
    // Execute real command
    let output = tokio::process::Command::new("./target/debug/rbee-keeper")
        .args(&parts[1..]) // Skip "rbee-keeper"
        .output()
        .await
        .expect("Failed to execute command");
    
    world.last_stdout = String::from_utf8_lossy(&output.stdout).to_string();
    world.last_stderr = String::from_utf8_lossy(&output.stderr).to_string();
    world.last_exit_code = output.status.code();
    
    tracing::info!("✅ Command executed: exit_code={:?}", world.last_exit_code);
}
```

### Step 3: Verify Real Behavior

```rust
#[then(expr = "rbee-hive spawns worker process {string} on port {int}")]
pub async fn then_spawn_worker(world: &mut World, binary: String, port: u16) {
    // Query real rbee-hive registry
    let client = reqwest::Client::new();
    let response = client
        .get("http://localhost:8080/v1/workers/list")
        .send()
        .await
        .expect("Failed to query workers");
    
    let workers: serde_json::Value = response.json().await.expect("Failed to parse response");
    let workers_list = workers["workers"].as_array().expect("No workers array");
    
    // Verify worker exists
    assert!(!workers_list.is_empty(), "No workers spawned");
    
    let worker = &workers_list[0];
    assert!(worker["url"].as_str().unwrap().contains(&port.to_string()));
    
    tracing::info!("✅ Verified worker spawned on port {}", port);
}
```

### Step 4: Verify Model Catalog

```rust
#[then(expr = "rbee-hive registers the model in SQLite catalog with local_path {string}")]
pub async fn then_register_model_in_catalog(world: &mut World, local_path: String) {
    // Query real SQLite database
    let db_path = shellexpand::tilde("~/.rbee/models.db");
    let conn = rusqlite::Connection::open(db_path.as_ref())
        .expect("Failed to open model catalog");
    
    let mut stmt = conn.prepare("SELECT local_path FROM models WHERE local_path = ?")
        .expect("Failed to prepare query");
    
    let exists: bool = stmt.exists([&local_path]).expect("Query failed");
    
    assert!(exists, "Model not found in catalog");
    tracing::info!("✅ Verified model in catalog: {}", local_path);
}
```

---

## What TEAM-043 Should Actually Do

### Priority 1: Wire BDD Tests to Current MVP ✅

**Focus on what EXISTS:**
1. Start rbee-hive daemon
2. Execute rbee-keeper infer commands
3. Verify worker spawning
4. Verify model catalog operations
5. Verify inference execution
6. Verify SSE token streaming

**Skip what DOESN'T EXIST:**
- ❌ queen-rbee (not in MVP)
- ❌ rbee-hive registry (future M1+)
- ❌ Setup commands (not implemented)
- ❌ SSH connections (not in MVP)

### Priority 2: Implement Missing Worker Shutdown Endpoint (Optional)

If you want complete lifecycle testing:

**Add to `bin/llm-worker-rbee/src/http/routes.rs`:**
```rust
.route("/v1/shutdown", post(shutdown::handle_shutdown))
```

**Create `bin/llm-worker-rbee/src/http/shutdown.rs`:**
```rust
pub async fn handle_shutdown() -> StatusCode {
    tracing::info!("Shutdown requested, exiting gracefully");
    std::process::exit(0);
}
```

### Priority 3: Implement queen-rbee + Setup Commands (Future M1+)

**Only if you have time** and want to implement the full M1+ architecture:
- Implement queen-rbee registry module
- Implement rbee-keeper setup commands
- Wire BDD tests for multi-node orchestration

But this is **NOT required** for MVP testing!

---

## Test Execution Strategy

### Phase 1: Smoke Test (Verify Current MVP Works)

```bash
# Terminal 1: Start rbee-hive
cd bin/rbee-hive
cargo run -- daemon --addr 0.0.0.0:8080

# Terminal 2: Run inference
cd bin/rbee-keeper
cargo run -- infer --node localhost --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --prompt "hello" --max-tokens 10

# Expected: Tokens stream to stdout, inference completes
```

### Phase 2: Wire BDD Tests

```bash
# Run BDD tests (they will start/stop rbee-hive automatically)
cd test-harness/bdd
cargo run --bin bdd-runner -- --tags @happy
```

### Phase 3: Iterate Until All Pass

Fix step definitions one by one until all scenarios pass.

---

## Summary

### What I Found (Accurate)
- ✅ **rbee-hive**: Fully implemented pool manager
- ✅ **llm-worker-rbee**: Fully implemented worker
- ✅ **rbee-keeper infer**: Fully implemented inference flow
- ✅ **model-catalog**: Fully implemented SQLite tracking
- ❌ **queen-rbee**: Scaffold only (not used in MVP)
- ❌ **rbee-keeper setup**: Not implemented (future feature)

### What You Should Do
1. **Wire BDD tests to current MVP** (rbee-hive + rbee-keeper infer)
2. **Skip queen-rbee** (not in MVP)
3. **Skip setup commands** (not implemented)
4. **Focus on what works** (ephemeral mode)

### My Mocks Are Still Useful
- They show expected behavior
- They show data flow
- Use them as hints for verification logic

---

**This is the ACCURATE handoff based on deep code survey. Good luck, TEAM-043!**
