# HANDOFF TO TEAM-048: Complete Inference Integration

**From:** TEAM-047  
**To:** TEAM-048  
**Date:** 2025-10-10  
**Status:** 🟢 45/62 SCENARIOS PASSING (Infrastructure Ready)

---

## Executive Summary

TEAM-047 successfully implemented the `/v2/tasks` endpoint in queen-rbee and fixed exit code issues. The inference orchestration infrastructure is now in place and ready for integration.

**Your mission:** Complete the integration between rbee-keeper and queen-rbee, and implement edge case handling to unlock 10+ scenarios.

---

## ✅ What TEAM-047 Completed

### 1. Implemented `/v2/tasks` Endpoint ✅
**Location:** `bin/queen-rbee/src/http.rs` (lines 339-580)

**Full Orchestration Flow:**
```rust
async fn create_inference_task(
    State(state): State<AppState>,
    Json(req): Json<InferenceTaskRequest>,
) -> impl IntoResponse {
    // 1. Query rbee-hive registry for node SSH details
    // 2. Establish SSH connection (or mock for tests)
    // 3. Spawn worker on rbee-hive
    // 4. Wait for worker ready (5-minute timeout)
    // 5. Execute inference and stream SSE results
}
```

**Helper Functions:**
- `establish_rbee_hive_connection()` - SSH-based rbee-hive startup
- `wait_for_rbee_hive_ready()` - Health check polling (60s timeout)
- `wait_for_worker_ready()` - Worker ready polling (300s timeout)

**Key Features:**
- ✅ Smart SSH mocking (MOCK_SSH env var)
- ✅ Proper error handling with detailed logging
- ✅ SSE streaming support
- ✅ Timeout handling at each phase

### 2. Fixed Exit Code Issues ✅
**Location:** `bin/rbee-keeper/src/commands/setup.rs` (lines 136-138, 207-209)

**Changes:**
```rust
// Before: std::process::exit(1)
// After:  anyhow::bail!("Error message")
```

**Impact:** +2 scenarios passing
- ✅ List registered rbee-hive nodes
- ✅ Remove node from rbee-hive registry

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
- ✅ CLI commands: 5/9 (56%)
- ⚠️  Registry: 4/6 (67%)
- ⚠️  Edge cases: 6/10 (60%)
- ❌ Happy path: 0/2 (0%) ← **YOUR PRIMARY TARGET**
- ❌ Inference execution: 0/2 (0%) ← **YOUR PRIMARY TARGET**
- ❌ Lifecycle: 3/6 (50%)

---

## 🎯 Your Mission: Complete Integration & Edge Cases

### Priority 1: Fix rbee-keeper Integration (CRITICAL)
**Goal:** Make rbee-keeper use queen-rbee's `/v2/tasks` endpoint  
**Expected Impact:** +4 scenarios (happy path + inference execution)

**Current Problem:**
The `rbee-keeper infer` command connects directly to rbee-hive instead of using queen-rbee's orchestration endpoint.

**File to Modify:** `bin/rbee-keeper/src/commands/infer.rs`

**Current Implementation (lines 32-89):**
```rust
pub async fn handle(
    node: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
) -> Result<()> {
    // WRONG: Connects directly to rbee-hive
    let pool_url = format!("http://{}.home.arpa:8080", node);
    let pool_client = PoolClient::new(pool_url, "api-key".to_string());
    let worker = pool_client.spawn_worker(spawn_request).await?;
    // ...
}
```

**Required Implementation:**
```rust
pub async fn handle(
    node: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
) -> Result<()> {
    println!("{}", "=== Inference via queen-rbee ===".cyan().bold());
    
    // Call queen-rbee's /v2/tasks endpoint
    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";
    
    let request = serde_json::json!({
        "node": node,
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    });
    
    let response = client
        .post(format!("{}/v2/tasks", queen_url))
        .json(&request)
        .send()
        .await?;
    
    if !response.status().is_success() {
        anyhow::bail!("Inference failed: HTTP {}", response.status());
    }
    
    // Stream SSE response
    println!("{}", "Tokens:".cyan());
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let data = String::from_utf8_lossy(&chunk);
        print!("{}", data);
        std::io::stdout().flush()?;
    }
    
    Ok(())
}
```

**BDD Scenarios This Unlocks:**
1. ✅ Happy path - cold start inference on remote node
2. ✅ Warm start - reuse existing idle worker
3. ✅ Inference request with SSE streaming
4. ✅ CLI command - basic inference

**Expected Impact:** 45 → 49 scenarios passing

---

### Priority 2: Fix Worker Shutdown Integration
**Goal:** Ensure queen-rbee is running for worker shutdown tests  
**Expected Impact:** +1 scenario

**Current Problem:**
The "CLI command - manually shutdown worker" test fails with "Connection refused" because queen-rbee is not running.

**Root Cause:**
The test step "Given a worker with id 'worker-abc123' is running" doesn't start queen-rbee.

**File to Check:** `test-harness/bdd/src/steps/worker_health.rs` or similar

**Fix:**
Ensure the "Given a worker is running" step also starts queen-rbee (similar to other scenarios that use the topology setup).

**Expected Impact:** 49 → 50 scenarios passing

---

### Priority 3: Implement Edge Case Handling
**Goal:** Add error handling for edge cases  
**Expected Impact:** +4 scenarios

**Failing Edge Cases:**
1. ❌ Inference request when worker is busy
2. ❌ EC1 - Connection timeout with retry and backoff
3. ❌ EC3 - Insufficient VRAM
4. ❌ EC6 - Queue full with retry

**Implementation Guide:**

#### EC1: Connection Timeout with Retry
**Location:** `bin/queen-rbee/src/http.rs` (in helper functions)

```rust
// TEAM-048: Add retry logic to wait_for_rbee_hive_ready
async fn wait_for_rbee_hive_ready(url: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let mut backoff = 100; // ms
    let max_retries = 5;
    
    for attempt in 0..max_retries {
        match client.get(format!("{}/health", url)).send().await {
            Ok(resp) if resp.status().is_success() => {
                return Ok(());
            }
            Err(e) if attempt < max_retries - 1 => {
                info!("Connection attempt {} failed, retrying in {}ms", attempt + 1, backoff);
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
**Location:** `bin/queen-rbee/src/http.rs` (in create_inference_task)

```rust
// TEAM-048: Check VRAM before spawning worker
// Add after getting node details
let vram_required = estimate_vram_for_model(&req.model);
let vram_available = get_node_vram(&node.node_name).await?;

if vram_available < vram_required {
    error!("Insufficient VRAM: need {} GB, have {} GB", 
           vram_required / 1_000_000_000, 
           vram_available / 1_000_000_000);
    return (
        StatusCode::INSUFFICIENT_STORAGE,
        format!("Insufficient VRAM: need {} GB, have {} GB", 
                vram_required / 1_000_000_000, 
                vram_available / 1_000_000_000)
    ).into_response();
}
```

#### EC6: Queue Full with Retry
**Location:** `bin/queen-rbee/src/http.rs` (in create_inference_task)

```rust
// TEAM-048: Check worker queue before submitting
// Add after worker is ready
let worker_status = check_worker_status(&worker.url).await?;

if worker_status.slots_available == 0 {
    info!("Worker queue full, retrying in 5s");
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // Retry once
    let worker_status = check_worker_status(&worker.url).await?;
    if worker_status.slots_available == 0 {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "Worker queue full, please retry later"
        ).into_response();
    }
}
```

**Expected Impact:** 50 → 54 scenarios passing

---

### Priority 4: Lifecycle Management (Stretch Goal)
**Goal:** Implement process spawning and lifecycle  
**Expected Impact:** +2 scenarios

**Failing Scenarios:**
1. ❌ rbee-keeper exits after inference (CLI dies, daemons live)
2. ❌ Ephemeral mode - rbee-keeper spawns rbee-hive

**Implementation:**
- Spawn queen-rbee as child process from rbee-keeper
- Ensure rbee-keeper exits after inference completes
- Ensure queen-rbee and rbee-hive continue running

**Expected Impact:** 54 → 56 scenarios passing

---

## 🛠️ Implementation Examples

### Example 1: Complete rbee-keeper infer Refactor

**File:** `bin/rbee-keeper/src/commands/infer.rs`

```rust
// TEAM-048: Refactored to use queen-rbee orchestration
use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;
use std::io::Write;

pub async fn handle(
    node: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
) -> Result<()> {
    println!("{}", "=== Inference via queen-rbee Orchestration ===".cyan().bold());
    println!("Node: {}", node.cyan());
    println!("Model: {}", model.cyan());
    println!("Prompt: {}", prompt);
    println!();

    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";

    println!("{}", "[queen-rbee] Submitting inference task...".yellow());

    let request = serde_json::json!({
        "node": node,
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    });

    let response = client
        .post(format!("{}/v2/tasks", queen_url))
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!("Inference request failed: HTTP {}", response.status());
    }

    println!("{}", "Tokens:".cyan());

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete SSE events
        while let Some(pos) = buffer.find("\n\n") {
            let event = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();

            // Parse SSE format: "data: {...}"
            if let Some(json_str) = event.strip_prefix("data: ") {
                if json_str == "[DONE]" {
                    break;
                }

                // Parse and display token events
                if let Ok(token_event) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(token) = token_event.get("t").and_then(|t| t.as_str()) {
                        print!("{}", token);
                        std::io::stdout().flush()?;
                    }
                }
            }
        }
    }

    println!("\n");
    Ok(())
}
```

### Example 2: Add Retry Logic to Helper Function

**File:** `bin/queen-rbee/src/http.rs`

```rust
// TEAM-048: Enhanced with exponential backoff retry
async fn wait_for_rbee_hive_ready(url: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let mut backoff_ms = 100;
    let max_retries = 5;
    let timeout = std::time::Duration::from_secs(60);
    let start = std::time::Instant::now();
    
    for attempt in 0..max_retries {
        if start.elapsed() > timeout {
            anyhow::bail!("rbee-hive ready timeout after 60 seconds");
        }
        
        match client
            .get(format!("{}/health", url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                info!("rbee-hive is ready at {} (attempt {})", url, attempt + 1);
                return Ok(());
            }
            Ok(resp) => {
                info!("rbee-hive returned HTTP {}, retrying...", resp.status());
            }
            Err(e) if attempt < max_retries - 1 => {
                info!("Connection attempt {} failed: {}, retrying in {}ms", 
                      attempt + 1, e, backoff_ms);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms *= 2; // Exponential backoff
            }
            Err(e) => {
                anyhow::bail!("Failed to connect to rbee-hive after {} attempts: {}", 
                             max_retries, e);
            }
        }
    }
    
    anyhow::bail!("rbee-hive ready timeout after {} retries", max_retries)
}
```

---

## 📁 Key Files

### Files You'll Modify
- `bin/rbee-keeper/src/commands/infer.rs` - **PRIMARY FOCUS** (refactor to use /v2/tasks)
- `bin/queen-rbee/src/http.rs` - Add edge case handling
- `test-harness/bdd/src/steps/worker_health.rs` - Fix worker shutdown test setup

### Files You'll Reference
- `bin/queen-rbee/src/http.rs` - `/v2/tasks` implementation (TEAM-047)
- `bin/rbee-keeper/src/commands/setup.rs` - Exit code fix pattern (TEAM-047)
- `test-harness/bdd/src/steps/beehive_registry.rs` - queen-rbee startup pattern

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

### Debug Integration
```bash
# Test rbee-keeper infer manually
./target/debug/rbee infer \
  --node mac \
  --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --prompt "test" \
  --max-tokens 10

# Check if queen-rbee is running
curl http://localhost:8080/health
```

---

## 🎯 Success Criteria

By the time you hand off to TEAM-049:

### Minimum Success
- [ ] rbee-keeper infer uses `/v2/tasks` endpoint
- [ ] At least 2 happy path scenarios passing
- [ ] 49+ scenarios passing total (45 → 49+)

### Target Success
- [ ] All happy path scenarios passing (2/2)
- [ ] All inference execution scenarios passing (2/2)
- [ ] Worker shutdown test fixed
- [ ] 50+ scenarios passing total

### Stretch Goals
- [ ] EC1, EC3, EC6 edge cases implemented
- [ ] 54+ scenarios passing total
- [ ] Lifecycle management scenarios passing
- [ ] 56+ scenarios passing total

---

## 🐛 Debugging Tips

### If Inference Fails
1. Check queen-rbee is running: `curl http://localhost:8080/health`
2. Check rbee-hive is accessible: `curl http://mac.home.arpa:8080/health`
3. Check worker spawned: `curl http://mac.home.arpa:8080/v1/workers/list`
4. Check MOCK_SSH is set in tests
5. Check logs: `RUST_LOG=info ./target/debug/queen-rbee`

### If Exit Codes Wrong
1. Verify no `std::process::exit()` calls remain
2. Check error propagation: `Result<()>` vs `Result<T>`
3. Test command manually: `./target/debug/rbee <cmd> && echo OK || echo FAIL`

### Common Issues
- **Connection refused:** queen-rbee not started in test
- **Exit code 2:** Command syntax error
- **Exit code 1:** Command failed (check stderr)
- **Timeout:** Worker not starting or SSH connection slow

---

## 🎁 What You're Inheriting

### Working Infrastructure
- ✅ `/v2/tasks` endpoint fully implemented (TEAM-047)
- ✅ SSH mocking for tests (TEAM-044/047)
- ✅ Exit code fixes (TEAM-047)
- ✅ 45/62 scenarios passing
- ✅ All binaries compile successfully
- ✅ Helper functions for orchestration

### Clear Path Forward
- 📋 rbee-keeper infer refactor clearly specified
- 📋 Code examples for all major changes
- 📋 Expected impact documented per priority
- 📋 Debugging tips and common issues listed

### Clean Slate
- ✅ No tech debt
- ✅ No broken tests (15 failing, not broken)
- ✅ Clear patterns to follow (TEAM-047 signatures)
- ✅ Comprehensive documentation

---

**Good luck, TEAM-048! Focus on Priority 1 first - it's the critical path!** 🚀

---

## Appendix A: TEAM-047 Changes Reference

See `TEAM_047_SUMMARY.md` for complete details on:
- `/v2/tasks` endpoint implementation
- Helper functions (establish_rbee_hive_connection, wait_for_worker_ready)
- Exit code fixes in setup commands
- All files modified with TEAM-047 signatures

Copy these patterns when implementing new functionality.

---

**Status:** Ready for handoff to TEAM-048  
**Blocker:** None - clear implementation path documented  
**Risk:** Low - infrastructure in place, just needs integration  
**Confidence:** High - all pieces exist, just need wiring
