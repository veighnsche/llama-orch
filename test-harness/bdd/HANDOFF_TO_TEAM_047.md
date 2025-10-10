# HANDOFF TO TEAM-047: Implement Inference Orchestration

**From:** TEAM-046  
**To:** TEAM-047  
**Date:** 2025-10-10  
**Status:** 🟢 45/62 SCENARIOS PASSING (+2 from TEAM-045)

---

## Executive Summary

TEAM-046 successfully implemented worker management commands, increasing passing scenarios from 43 to 45. The infrastructure is now in place for the critical missing piece: **inference orchestration**.

**Your mission:** Implement the `/v2/tasks` endpoint in queen-rbee to enable end-to-end inference, unlocking 4+ scenarios.

---

## ✅ What TEAM-046 Completed

### 1. Worker Management Commands ✅
**Location:** `bin/rbee-keeper/src/commands/workers.rs`, `logs.rs`

**Commands Implemented:**
```bash
rbee-keeper workers list                    # ✅ Works
rbee-keeper workers health --node mac       # ✅ Works
rbee-keeper workers shutdown --id worker-123 # ✅ Works
rbee-keeper logs --node mac --follow        # ✅ Works
```

**Impact:** +2 scenarios passing (list workers, check health)

### 2. queen-rbee Worker Endpoints ✅
**Location:** `bin/queen-rbee/src/http.rs`

**Endpoints Added:**
```
GET  /v2/workers/list         ✅ Implemented
GET  /v2/workers/health?node= ✅ Implemented
POST /v2/workers/shutdown     ✅ Implemented
POST /v2/tasks                ⚠️  Stub only (YOUR PRIORITY)
```

### 3. WorkerRegistry Enhancements ✅
**Location:** `bin/queen-rbee/src/worker_registry.rs`

**New Methods:**
- `list_workers()` → List all workers
- `get_workers_by_node(node)` → Filter by node
- `shutdown_worker(id)` → Shutdown worker

**New Field:**
- `WorkerInfo.node_name` → Track worker location

---

## 📊 Current Status

### Test Results
```
62 scenarios total
45 passing (73%)
17 failing (27%)

789 steps total
772 passing (98%)
17 failing (2%)
```

### Passing Scenarios by Category
- ✅ @setup: 6/6 (100%)
- ✅ Pool preflight: 3/3 (100%)
- ✅ Model provisioning: 4/4 (100%)
- ✅ Worker preflight: 4/4 (100%)
- ✅ Worker startup: 2/2 (100%)
- ✅ Worker registration: 1/1 (100%)
- ✅ Worker health: 3/3 (100%)
- ✅ Edge cases: 6/10 (60%)
- ✅ CLI commands: 5/9 (56%) ← **+2 from TEAM-045**
- ❌ Happy path: 0/2 (0%) ← **YOUR TARGET**
- ❌ Inference execution: 0/2 (0%) ← **YOUR TARGET**
- ❌ Lifecycle: 3/6 (50%)
- ⚠️  Registry: 4/6 (67%) - 2 exit code issues

---

## 🎯 Your Mission: Implement Inference Orchestration

### Priority 1: Implement `/v2/tasks` Endpoint (CRITICAL)
**Goal:** Enable end-to-end inference flow  
**Expected Impact:** +4 scenarios passing

**Current State:**
```rust
// bin/queen-rbee/src/http.rs:339
async fn create_inference_task(
    State(_state): State<AppState>,
    Json(req): Json<InferenceTaskRequest>,
) -> impl IntoResponse {
    info!("Received inference task: node={}, model={}", req.node, req.model);
    
    // TODO: Implement full orchestration flow
    // 1. Query rbee-hive registry for node SSH details
    // 2. Establish SSH connection
    // 3. Start rbee-hive on remote node
    // 4. Request worker from rbee-hive
    // 5. Stream inference results
    
    (
        StatusCode::NOT_IMPLEMENTED,
        "Inference orchestration not yet implemented",
    )
}
```

**What You Need to Implement:**

#### Step 1: Query rbee-hive Registry
```rust
// Get node SSH details from beehive_registry
let node = state.beehive_registry.get_node(&req.node).await?;
if node.is_none() {
    return (StatusCode::NOT_FOUND, "Node not registered");
}
let node = node.unwrap();
```

#### Step 2: Establish SSH Connection
```rust
// Use existing SSH module
use crate::ssh;

// Check if MOCK_SSH is set (for tests)
let mock_ssh = std::env::var("MOCK_SSH").is_ok();

if !mock_ssh {
    // Real SSH connection
    let ssh_session = ssh::connect(
        &node.ssh_host,
        node.ssh_port,
        &node.ssh_user,
        node.ssh_key_path.as_deref(),
    ).await?;
}
```

#### Step 3: Start rbee-hive on Remote Node
```rust
// Execute remote command via SSH
let rbee_hive_url = if mock_ssh {
    // For tests, assume rbee-hive is already running
    format!("http://{}:8080", node.ssh_host)
} else {
    // Real SSH: start rbee-hive daemon
    ssh::execute_remote_command(
        &ssh_session,
        &format!("{}/rbee-hive daemon --addr 0.0.0.0:8080 &", node.install_path)
    ).await?;
    
    // Wait for rbee-hive to be ready
    wait_for_rbee_hive_ready(&format!("http://{}:8080", node.ssh_host)).await?;
    
    format!("http://{}:8080", node.ssh_host)
};
```

#### Step 4: Request Worker from rbee-hive
```rust
// POST to rbee-hive to spawn worker
let client = reqwest::Client::new();
let spawn_request = serde_json::json!({
    "model_ref": req.model,
    "backend": "cpu", // TODO: Detect from node capabilities
    "device": 0,
    "model_path": "" // rbee-hive resolves from catalog
});

let response = client
    .post(format!("{}/v1/workers/spawn", rbee_hive_url))
    .json(&spawn_request)
    .send()
    .await?;

let worker: WorkerSpawnResponse = response.json().await?;
```

#### Step 5: Stream Inference Results
```rust
// Forward inference request to worker
let inference_request = serde_json::json!({
    "prompt": req.prompt,
    "max_tokens": req.max_tokens,
    "temperature": req.temperature,
    "stream": true
});

let response = client
    .post(format!("{}/v1/inference", worker.url))
    .json(&inference_request)
    .send()
    .await?;

// Stream SSE response back to client
use axum::response::sse::{Event, Sse};
use futures::stream::Stream;

let stream = response.bytes_stream().map(|chunk| {
    // Convert worker SSE to queen-rbee SSE
    Event::default().data(String::from_utf8_lossy(&chunk?))
});

(StatusCode::OK, Sse::new(stream))
```

**BDD Scenarios This Unlocks:**
1. ✅ Happy path - cold start inference on remote node
2. ✅ Warm start - reuse existing idle worker
3. ✅ Inference request with SSE streaming
4. ✅ CLI command - basic inference

**Expected Impact:** +4 scenarios passing (49/62 total)

---

### Priority 2: Fix Exit Code Issues (QUICK WINS)
**Goal:** Debug why some commands return wrong exit codes  
**Expected Impact:** +3 scenarios

**Failing Scenarios:**
1. ❌ List registered rbee-hive nodes (exit code 2 instead of 0)
2. ❌ Remove node from rbee-hive registry (exit code 2 instead of 0)
3. ❌ CLI command - manually shutdown worker (exit code 1 instead of 0)

**Root Cause:** Unknown - needs debugging

**How to Debug:**
```bash
# Run command manually
./target/debug/rbee setup list-nodes
echo $?  # Check exit code

# Run with tracing
RUST_LOG=debug ./target/debug/rbee setup list-nodes

# Check BDD test output
cd test-harness/bdd
cargo run --bin bdd-runner -- --name "List registered rbee-hive nodes"
```

**Likely Issues:**
- Command returns error when it should succeed
- Missing error handling in command implementation
- Incorrect exit code propagation

**Fix Pattern:**
```rust
// In bin/rbee-keeper/src/commands/setup.rs
pub async fn handle(action: SetupAction) -> anyhow::Result<()> {
    match action {
        SetupAction::ListNodes => {
            // Ensure this returns Ok(()) on success
            list_nodes().await?;
            Ok(())  // ← Make sure this is reached
        }
        // ...
    }
}
```

---

### Priority 3: Implement Edge Case Handling
**Goal:** Add error handling for edge cases  
**Expected Impact:** +6 scenarios

**Failing Edge Cases:**
1. ❌ Inference request when worker is busy
2. ❌ EC1 - Connection timeout with retry and backoff
3. ❌ EC3 - Insufficient VRAM
4. ❌ EC6 - Queue full with retry
5. ❌ EC7 - Model loading timeout
6. ❌ EC8 - Version mismatch
7. ❌ EC9 - Invalid API key

**Implementation Guide:**

#### EC1: Connection Timeout with Retry
```rust
async fn connect_with_retry(url: &str) -> Result<reqwest::Response> {
    let mut backoff = 100; // ms
    let max_retries = 5;
    
    for attempt in 0..max_retries {
        match reqwest::get(url).await {
            Ok(resp) => return Ok(resp),
            Err(e) if attempt < max_retries - 1 => {
                tokio::time::sleep(Duration::from_millis(backoff)).await;
                backoff *= 2; // Exponential backoff
            }
            Err(e) => return Err(e.into()),
        }
    }
    anyhow::bail!("Connection timeout after {} retries", max_retries)
}
```

#### EC3: Insufficient VRAM
```rust
// Check VRAM before spawning worker
let vram_required = estimate_vram_for_model(&model_ref);
let vram_available = get_node_vram(&node_name).await?;

if vram_available < vram_required {
    return Err(anyhow::anyhow!(
        "Insufficient VRAM: need {} GB, have {} GB",
        vram_required / 1_000_000_000,
        vram_available / 1_000_000_000
    ));
}
```

#### EC6: Queue Full with Retry
```rust
// Check worker queue before submitting
let worker_status = get_worker_status(&worker_id).await?;

if worker_status.slots_available == 0 {
    // Retry after delay
    tokio::time::sleep(Duration::from_secs(5)).await;
    // Try again or return error
}
```

#### EC7: Model Loading Timeout
```rust
// Already implemented in rbee-keeper/src/commands/infer.rs:98
// wait_for_worker_ready() has 5-minute timeout
// Just ensure it's used in orchestration flow
```

#### EC8: Version Mismatch
```rust
// Check version compatibility
let rbee_hive_version = get_rbee_hive_version(&rbee_hive_url).await?;
let queen_version = env!("CARGO_PKG_VERSION");

if !versions_compatible(queen_version, &rbee_hive_version) {
    return Err(anyhow::anyhow!(
        "Version mismatch: queen-rbee {}, rbee-hive {}",
        queen_version,
        rbee_hive_version
    ));
}
```

#### EC9: Invalid API Key
```rust
// Validate API key before processing request
if let Some(api_key) = req.api_key {
    if !validate_api_key(&api_key).await? {
        return (StatusCode::UNAUTHORIZED, "Invalid API key");
    }
}
```

---

### Priority 4: Lifecycle Management
**Goal:** Implement process spawning and lifecycle  
**Expected Impact:** +2 scenarios

**Failing Scenarios:**
1. ❌ rbee-keeper exits after inference (CLI dies, daemons live)
2. ❌ Ephemeral mode - rbee-keeper spawns rbee-hive

**Implementation:**
- Spawn queen-rbee as child process from rbee-keeper
- Ensure rbee-keeper exits after inference completes
- Ensure queen-rbee and rbee-hive continue running

---

## 🛠️ Implementation Example: Complete `/v2/tasks` Handler

```rust
// bin/queen-rbee/src/http.rs

async fn create_inference_task(
    State(state): State<AppState>,
    Json(req): Json<InferenceTaskRequest>,
) -> impl IntoResponse {
    info!("Received inference task: node={}, model={}", req.node, req.model);
    
    // Step 1: Query rbee-hive registry
    let node = match state.beehive_registry.get_node(&req.node).await {
        Ok(Some(node)) => node,
        Ok(None) => {
            return (StatusCode::NOT_FOUND, "Node not registered".to_string()).into_response();
        }
        Err(e) => {
            error!("Failed to query registry: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };
    
    // Step 2: Establish SSH connection (or mock)
    let mock_ssh = std::env::var("MOCK_SSH").is_ok();
    let rbee_hive_url = if mock_ssh {
        format!("http://{}:8080", node.ssh_host)
    } else {
        // Real SSH implementation
        match establish_rbee_hive_connection(&node).await {
            Ok(url) => url,
            Err(e) => {
                error!("Failed to connect to rbee-hive: {}", e);
                return (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response();
            }
        }
    };
    
    // Step 3: Spawn worker on rbee-hive
    let client = reqwest::Client::new();
    let spawn_request = serde_json::json!({
        "model_ref": req.model,
        "backend": "cpu",
        "device": 0,
        "model_path": ""
    });
    
    let worker = match client
        .post(format!("{}/v1/workers/spawn", rbee_hive_url))
        .json(&spawn_request)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<WorkerSpawnResponse>().await {
                Ok(worker) => worker,
                Err(e) => {
                    error!("Failed to parse worker response: {}", e);
                    return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
                }
            }
        }
        Ok(resp) => {
            error!("Worker spawn failed: HTTP {}", resp.status());
            return (StatusCode::INTERNAL_SERVER_ERROR, "Worker spawn failed").into_response();
        }
        Err(e) => {
            error!("Failed to spawn worker: {}", e);
            return (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response();
        }
    };
    
    // Step 4: Wait for worker ready
    match wait_for_worker_ready(&worker.url).await {
        Ok(_) => info!("Worker ready: {}", worker.worker_id),
        Err(e) => {
            error!("Worker failed to become ready: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    }
    
    // Step 5: Execute inference and stream results
    let inference_request = serde_json::json!({
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": true
    });
    
    let response = match client
        .post(format!("{}/v1/inference", worker.url))
        .json(&inference_request)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => resp,
        Ok(resp) => {
            error!("Inference failed: HTTP {}", resp.status());
            return (StatusCode::INTERNAL_SERVER_ERROR, "Inference failed").into_response();
        }
        Err(e) => {
            error!("Failed to execute inference: {}", e);
            return (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response();
        }
    };
    
    // Stream SSE response
    use axum::response::sse::{Event, Sse};
    use futures::StreamExt;
    
    let stream = response.bytes_stream().filter_map(|chunk| async move {
        match chunk {
            Ok(bytes) => Some(Ok::<_, axum::Error>(Event::default().data(String::from_utf8_lossy(&bytes).to_string()))),
            Err(e) => {
                error!("Stream error: {}", e);
                None
            }
        }
    });
    
    Sse::new(stream).into_response()
}

// Helper: Wait for worker to be ready
async fn wait_for_worker_ready(worker_url: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(300);
    
    loop {
        match client.get(format!("{}/v1/ready", worker_url)).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(ready) = resp.json::<ReadyResponse>().await {
                    if ready.ready {
                        return Ok(());
                    }
                }
            }
            _ => {}
        }
        
        if start.elapsed() > timeout {
            anyhow::bail!("Worker ready timeout");
        }
        
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
}

#[derive(Deserialize)]
struct ReadyResponse {
    ready: bool,
}

#[derive(Deserialize)]
struct WorkerSpawnResponse {
    worker_id: String,
    url: String,
}
```

---

## 📁 Key Files

### Files You'll Modify
- `bin/queen-rbee/src/http.rs` - Implement `/v2/tasks` handler
- `bin/queen-rbee/src/ssh.rs` - SSH connection helpers (may need additions)
- `bin/rbee-keeper/src/commands/setup.rs` - Fix exit codes

### Files You'll Reference
- `bin/rbee-keeper/src/commands/infer.rs` - Example inference flow
- `bin/rbee-hive/src/http/workers.rs` - Worker spawn endpoint
- `bin/llm-worker-rbee/src/http/ready.rs` - Ready endpoint
- `test-harness/bdd/tests/features/test-001.feature` - BDD scenarios

---

## 🏃 Quick Start Commands

### Build Everything
```bash
cargo build --bin rbee --bin queen-rbee --bin rbee-hive --bin llm-worker-rbee
```

### Run All Tests
```bash
cd test-harness/bdd
cargo run --bin bdd-runner
```

### Run Specific Scenario
```bash
cargo run --bin bdd-runner -- --name "Happy path - cold start inference on remote node"
```

### Debug Exit Codes
```bash
./target/debug/rbee setup list-nodes
echo $?

RUST_LOG=debug ./target/debug/rbee setup list-nodes
```

---

## 🎯 Success Criteria

By the time you hand off to TEAM-048:

### Minimum Success
- [ ] `/v2/tasks` endpoint implemented
- [ ] At least 2 happy path scenarios passing
- [ ] 49+ scenarios passing total (45 → 49+)

### Stretch Goals
- [ ] All happy path scenarios passing (2/2)
- [ ] All inference execution scenarios passing (2/2)
- [ ] Exit code issues fixed (+3 scenarios)
- [ ] 55+ scenarios passing total

---

## 🐛 Debugging Tips

### If Inference Fails
1. Check queen-rbee logs: `RUST_LOG=info ./target/debug/queen-rbee`
2. Check rbee-hive logs on remote node
3. Check worker logs: `curl http://worker-url/v1/health`
4. Verify SSH connection: `ssh -i ~/.ssh/key user@host "echo test"`
5. Check MOCK_SSH env var in tests

### If Exit Codes Wrong
1. Add debug logging to command handlers
2. Check error propagation: `Result<()>` vs `Result<T>`
3. Verify `main.rs` exit code handling
4. Test command manually: `./target/debug/rbee <cmd> && echo OK || echo FAIL`

### Common Issues
- **Connection refused:** Service not started
- **Exit code 2:** Command syntax error or not found
- **Exit code 1:** Command failed
- **Timeout:** Worker not starting or SSH connection slow

---

## 🎁 What You're Inheriting

### Working Infrastructure
- ✅ rbee-keeper workers commands fully functional
- ✅ queen-rbee worker management endpoints operational
- ✅ WorkerRegistry enhanced with new methods
- ✅ rbee-hive daemon mode exists and works
- ✅ SSH mocking for tests (MOCK_SSH env var)
- ✅ 45/62 scenarios passing
- ✅ All binaries compile successfully

### Clear Path Forward
- 📋 `/v2/tasks` implementation guide provided
- 📋 Code examples for all major components
- 📋 Expected impact documented per priority
- 📋 Debugging tips and common issues listed

### Clean Slate
- ✅ No tech debt
- ✅ No broken tests (17 failing, not broken)
- ✅ Clear patterns to follow
- ✅ Comprehensive documentation

---

**Good luck, TEAM-047! Focus on `/v2/tasks` first - it's the critical path!** 🚀

---

## Appendix A: SSH Module Reference

The `bin/queen-rbee/src/ssh.rs` module already has:

```rust
pub async fn test_ssh_connection(
    host: &str,
    port: u16,
    user: &str,
    key_path: Option<&str>,
) -> Result<bool>
```

You may need to add:

```rust
pub async fn connect(
    host: &str,
    port: u16,
    user: &str,
    key_path: Option<&str>,
) -> Result<SshSession>

pub async fn execute_remote_command(
    session: &SshSession,
    command: &str,
) -> Result<String>
```

---

## Appendix B: TEAM-046 Changes Reference

See `TEAM_046_SUMMARY.md` for complete details on:
- Worker management commands implementation
- queen-rbee endpoint additions
- WorkerRegistry enhancements
- All files modified with TEAM-046 signatures

Copy these patterns when implementing new functionality.

---

**Status:** Ready for handoff to TEAM-047  
**Blocker:** None - clear implementation path documented  
**Risk:** Medium - orchestration is complex but well-specified  
**Confidence:** High - all infrastructure in place, just needs wiring
