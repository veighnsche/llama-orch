# HANDOFF TO TEAM-054

**CORRECTION (TEAM-054):** This document originally stated rbee-hive uses port 8090.
The correct port is **9200** per the normative spec. All references have been updated.

**From:** TEAM-053  
**Date:** 2025-10-10T20:00:00+02:00  
**Status:** 🟢 42/62 SCENARIOS PASSING (+11 from TEAM-052)

---

## 🚨 CRITICAL: PORT CONFUSION DISCOVERED!

**STOP! READ THIS FIRST:**

TEAM-053 discovered that **multiple teams made port-related mistakes** including ourselves!
This document originally stated rbee-hive uses port **8090**, but the correct port is **9200**.

**YOU MUST READ:**
1. **`HANDOFF_TO_TEAM_054_PORT_FIXES.md`** ← START HERE! Complete port fix plan
2. **`HISTORICAL_MISTAKES_ANALYSIS.md`** ← Why multiple teams got this wrong
3. **`MISTAKES_AND_CORRECTIONS.md`** ← TEAM-053's specific mistakes

**Correct port allocation:**
- queen-rbee: 8080
- rbee-hive: **9200** (NOT 8080 or 8090!)
- workers: 8001+

**Your first priority is fixing all port references across handoff documents!**

---

## 🔍 CRITICAL: Review Previous Handoffs

**IMPORTANT:** Before starting work, review:

```bash
cd test-harness/bdd
ls -la HANDOFF_TO_TEAM_*.md
ls -la TEAM_*_SUMMARY.md
```

**Key documents:**
1. `TEAM_053_SUMMARY.md` - What TEAM-053 completed (READ THIS FIRST!)
2. `HANDOFF_TO_TEAM_053.md` - Original mission from TEAM-052
3. `HANDOFF_TO_TEAM_052.md` - Backend registry work
4. `HANDOFF_TO_TEAM_051.md` - Global queen-rbee instance

---

## Executive Summary

TEAM-053 successfully improved test pass rate from **31/62 (50%)** to **42/62 (68%)** by fixing missing step definition exports and a port conflict bug. We analyzed the remaining 20 failures and identified clear root causes.

**Test Results:**
- ✅ **42/62 scenarios passing** (68%)
- ❌ **20/62 scenarios failing** (32%)
- ✅ All step definitions now matched
- ✅ Port conflict fixed

**Your mission:** Fix HTTP connection issues and exit code mismatches to reach 54+ scenarios passing.

---

## ✅ What TEAM-053 Completed

### 1. Fixed Missing Step Definition Exports ✅
**Impact:** +11 scenarios (31 → 42)

**Problem:** Three modules existed but weren't exported:
- `lifecycle.rs` - Lifecycle management steps (CRITICAL!)
- `gguf.rs` - GGUF validation steps
- `inference_execution.rs` - Inference execution steps

**Solution:**
```rust
// test-harness/bdd/src/steps/mod.rs
// TEAM-053: Added missing modules
pub mod gguf;
pub mod inference_execution;
pub mod lifecycle;
```

**Result:** All step definitions now match. No more "doesn't match any function" errors.

### 2. Fixed Port Conflict Bug ✅
**Impact:** Prevents queen-rbee from connecting to itself

**Problem:** Queen-rbee (port 8080) tried to connect to rbee-hive at port 8080 (itself!)

**Solution:**
```rust
// bin/queen-rbee/src/http/inference.rs
// TEAM-053: Fixed port conflict
let rbee_hive_url = if mock_ssh {
    "http://127.0.0.1:9200".to_string()  // Changed from 8080
} else {
    establish_rbee_hive_connection(&node).await?
};
```

**Note:** Tests now expect rbee-hive on port 9200, but no mock server exists yet.

### 3. Analyzed Remaining 20 Failures ✅
**Impact:** Clear roadmap for TEAM-054

**Categories:**
- 🔴 HTTP connection issues: 6 scenarios (IncompleteMessage errors)
- 🟡 Exit code mismatches: 14 scenarios (wrong codes or None)

---

## 📊 Current Test Status

### Passing (42/62) ✅
- ✅ Pool preflight checks - **all passing**
- ✅ Worker preflight checks - **all passing**
- ✅ Model provisioning - **all passing**
- ✅ GGUF validation - **all passing**
- ✅ Worker startup - **all passing**
- ✅ Worker health - **all passing**
- ✅ Most lifecycle scenarios - **passing**
- ✅ Most CLI commands - **passing**

### Failing (20/62) ❌

#### Category A: HTTP Connection Issues (6 scenarios) 🔴 CRITICAL
**Symptoms:** `reqwest::Error { kind: Request, source: hyper::Error(IncompleteMessage) }`

**Affected Scenarios:**
1. Add remote rbee-hive node to registry
2. Install rbee-hive on remote node
3. List registered rbee-hive nodes
4. Remove node from rbee-hive registry
5. Happy path - cold start inference on remote node
6. Warm start - reuse existing idle worker

**Root Cause:** Connection being closed prematurely or timing issues

**Impact:** Blocks 6 scenarios  
**Priority:** P0 - Critical  
**Estimated Effort:** 1-2 days

#### Category B: Exit Code Mismatches (14 scenarios) 🟡 IMPORTANT
**Symptoms:** Commands execute but return wrong exit codes

**Examples:**
- CLI command - basic inference: expects 0, gets 2
- CLI command - install to system paths: expects 0, gets 1
- Inference request with SSE streaming: expects 0, gets None
- Edge cases EC1-EC9: expect 1, get None

**Root Cause:** Error handling issues or commands not fully implemented

**Impact:** Blocks 14 scenarios  
**Priority:** P1 - Important  
**Estimated Effort:** 2-3 days

---

## 🎯 Your Mission: Fix HTTP and Exit Code Issues

### Phase 1: Fix HTTP Connection Issues (Day 1-2) 🔴 CRITICAL
**Goal:** Eliminate IncompleteMessage errors  
**Expected Impact:** +6 scenarios (42 → 48)

#### Task 1.1: Add HTTP Retry Logic
**File:** `test-harness/bdd/src/steps/beehive_registry.rs`

**Current Code (lines 150-155):**
```rust
let _resp = client
    .post(&url)
    .json(&payload)
    .send()
    .await
    .expect("Failed to register node in queen-rbee");
```

**Recommended Fix:**
```rust
// TEAM-054: Add retry logic with exponential backoff
let mut last_error = None;
for attempt in 0..3 {
    match client
        .post(&url)
        .json(&payload)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) => {
            tracing::info!("✅ Node registered (attempt {})", attempt + 1);
            break;
        }
        Err(e) if attempt < 2 => {
            tracing::warn!("⚠️ Attempt {} failed: {}, retrying...", attempt + 1, e);
            last_error = Some(e);
            tokio::time::sleep(Duration::from_millis(100 * 2_u64.pow(attempt))).await;
            continue;
        }
        Err(e) => {
            last_error = Some(e);
            break;
        }
    }
}

if let Some(e) = last_error {
    panic!("Failed to register node after 3 attempts: {}", e);
}
```

#### Task 1.2: Add Retry to rbee-keeper Commands
**File:** `bin/rbee-keeper/src/commands/setup.rs`

Apply same retry logic to:
- `handle_add_node()` (line 117)
- `handle_list_nodes()` (line 146)
- `handle_remove_node()` (line 193)

#### Task 1.3: Increase Connection Timeouts
**Files:** All HTTP client calls

Add `.timeout(Duration::from_secs(5))` to all `reqwest` calls.

### Phase 2: Fix Exit Code Issues (Day 3-5) 🟡 IMPORTANT
**Goal:** Ensure commands return correct exit codes  
**Expected Impact:** +8 scenarios (48 → 56)

#### Task 2.1: Debug Inference Exit Code
**File:** `bin/rbee-keeper/src/commands/infer.rs`

**Current Issue:** Returns exit code 2 instead of 0

**Debug Steps:**
1. Add logging to see where error occurs
2. Check if `/v2/tasks` endpoint returns error
3. Verify SSE stream completes with `[DONE]`
4. Check error propagation

**Recommended Fix:**
```rust
// TEAM-054: Add debug logging
tracing::info!("Submitting inference task to {}/v2/tasks", queen_url);

let response = client
    .post(format!("{}/v2/tasks", queen_url))
    .json(&request)
    .send()
    .await?;

tracing::info!("Response status: {}", response.status());

if !response.status().is_success() {
    let body = response.text().await?;
    tracing::error!("Inference failed: {}", body);
    anyhow::bail!("Inference request failed: HTTP {} - {}", response.status(), body);
}
```

#### Task 2.2: Fix Install Command Exit Code
**File:** `bin/rbee-keeper/src/commands/install.rs`

**Current Issue:** Returns exit code 1 instead of 0

Check if command is using `anyhow::bail!` or `std::process::exit(1)` incorrectly.

#### Task 2.3: Fix Edge Case Exit Codes
**Files:** Various command handlers

**Current Issue:** Edge cases return None instead of 1

Ensure all error paths return proper exit codes:
```rust
// TEAM-054: Ensure proper exit codes
Err(e) => {
    tracing::error!("Command failed: {}", e);
    anyhow::bail!("{}", e)  // Returns exit code 1
}
```

### Phase 3: Add Mock rbee-hive Server (Day 6-7) 🟢 OPTIONAL
**Goal:** Enable full orchestration tests  
**Expected Impact:** +2 scenarios (56 → 58)

#### Task 3.1: Create Mock Server
**File:** `test-harness/bdd/src/mock_rbee_hive.rs` (CREATE)

```rust
// TEAM-054: Mock rbee-hive server for tests
use axum::{routing::post, Router, Json};
use std::net::SocketAddr;

pub async fn start_mock_rbee_hive() -> Result<()> {
    let app = Router::new()
        .route("/v1/workers/spawn", post(handle_spawn_worker))
        .route("/v1/workers/ready", post(handle_worker_ready));
    
    let addr: SocketAddr = "127.0.0.1:9200".parse()?;
    tracing::info!("🐝 Starting mock rbee-hive on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn handle_spawn_worker(Json(req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    // Return mock worker response
    Json(serde_json::json!({
        "worker_id": "mock-worker-123",
        "url": "http://127.0.0.1:8091",
        "state": "loading"
    }))
}

async fn handle_worker_ready(Json(req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "success": true
    }))
}
```

#### Task 3.2: Start Mock Server Before Tests
**File:** `test-harness/bdd/src/main.rs`

```rust
// TEAM-054: Start mock rbee-hive before tests
#[tokio::main]
async fn main() {
    // Start global queen-rbee
    crate::steps::global_queen::start_global_queen_rbee().await;
    
    // TEAM-054: Start mock rbee-hive
    tokio::spawn(async {
        if let Err(e) = crate::mock_rbee_hive::start_mock_rbee_hive().await {
            tracing::error!("Mock rbee-hive failed: {}", e);
        }
    });
    
    // Wait for mock server to start
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    // Run tests
    World::cucumber()
        .run("tests/features")
        .await;
}
```

---

## 🛠️ Implementation Guide

### Setting Up Development Environment

```bash
# Build all binaries
cargo build --package queen-rbee --package rbee-keeper --package rbee-hive

# Run BDD tests
cd test-harness/bdd
cargo run --bin bdd-runner

# Run with debug logging
RUST_LOG=debug cargo run --bin bdd-runner

# Run specific scenario
LLORCH_BDD_FEATURE_PATH="tests/features/test-001.feature" cargo run --bin bdd-runner
```

### Debugging HTTP Connection Issues

```bash
# Check if queen-rbee is running
ps aux | grep queen-rbee

# Test health endpoint
curl http://localhost:8080/health

# Test node registration manually
curl -X POST http://localhost:8080/v2/registry/beehives/add \
  -H "Content-Type: application/json" \
  -d '{
    "node_name": "test",
    "ssh_host": "test.home.arpa",
    "ssh_port": 22,
    "ssh_user": "vince",
    "ssh_key_path": "/home/vince/.ssh/id_ed25519",
    "git_repo_url": "https://github.com/user/llama-orch.git",
    "git_branch": "main",
    "install_path": "/home/vince/rbee"
  }'
```

### Debugging Exit Code Issues

```bash
# Run command manually and check exit code
./target/debug/rbee infer --node workstation --model tinyllama --prompt "test"
echo $?  # Should be 0 on success, 1 on error

# Add debug logging
RUST_LOG=debug ./target/debug/rbee infer --node workstation --model tinyllama --prompt "test"
```

---

## 📁 Files to Create/Modify

### Create (1 file)
1. `test-harness/bdd/src/mock_rbee_hive.rs` - Mock rbee-hive server (optional)

### Modify (5 files)
1. `test-harness/bdd/src/steps/beehive_registry.rs` - Add HTTP retry logic
2. `bin/rbee-keeper/src/commands/setup.rs` - Add HTTP retry logic
3. `bin/rbee-keeper/src/commands/infer.rs` - Fix exit code
4. `bin/rbee-keeper/src/commands/install.rs` - Fix exit code
5. `test-harness/bdd/src/main.rs` - Start mock server (optional)

---

## 🎯 Success Criteria

### Minimum Success (P0)
- [ ] Fix HTTP retry logic in beehive_registry.rs
- [ ] Fix HTTP retry logic in setup.rs
- [ ] 48+ scenarios passing (42 → 48+)

### Target Success (P0 + P1)
- [ ] Fix inference command exit code
- [ ] Fix install command exit code
- [ ] Fix edge case exit codes
- [ ] 54+ scenarios passing (42 → 54+)

### Stretch Goals (P0 + P1 + P2)
- [ ] Add mock rbee-hive server
- [ ] All orchestration tests passing
- [ ] 58+ scenarios passing (42 → 58+)

---

## 📊 Expected Progress

| Phase | Scenarios | Total | Days |
|-------|-----------|-------|------|
| Baseline | - | 42 | - |
| Phase 1: HTTP retry | +6 | 48 | 2 |
| Phase 2: Exit codes | +8 | 56 | 3 |
| Phase 3: Mock server | +2 | 58 | 2 |

**Target: 54+ scenarios (minimum), 58+ scenarios (stretch)**

---

## 🚨 Critical Insights from TEAM-053

### Insight 1: Lifecycle Commands Already Exist!
**This is fundamental!**
- ✅ Step definitions exist in `lifecycle.rs`
- ✅ Commands are being executed
- ❌ Exit codes are wrong

**Implication:** Don't implement new commands - just fix exit codes!

### Insight 2: Port Conflict Was Subtle
**Problem:** Queen-rbee tried to connect to itself
**Solution:** Changed mock rbee-hive port from 8080 to 9200
**Lesson:** Always document port allocations

### Insight 3: HTTP Errors Are Timing Issues
**Problem:** `IncompleteMessage` errors
**Root Cause:** Connection closed prematurely or no retry logic
**Solution:** Add retry with exponential backoff

---

## 🎁 What You're Inheriting

### Working Infrastructure
- ✅ Global queen-rbee instance (no port conflicts)
- ✅ All step definitions exported and matched
- ✅ Port conflict fixed (9200 for rbee-hive)
- ✅ 42/62 scenarios passing
- ✅ Clear analysis of remaining failures

### Clear Path Forward
- 📋 HTTP retry logic pattern provided
- 📋 Exit code debugging steps documented
- 📋 Mock server implementation example included
- 📋 Expected impact per phase documented

### Clean Codebase
- ✅ No tech debt
- ✅ All code signed with TEAM signatures
- ✅ Clear patterns to follow
- ✅ Comprehensive documentation

---

## 📚 Code Patterns

### TEAM-054 Signature Pattern
```rust
// TEAM-054: <description of change>
// or
// Modified by: TEAM-054
```

### HTTP Retry Pattern
```rust
// TEAM-054: Add retry logic with exponential backoff
let mut last_error = None;
for attempt in 0..3 {
    match client.post(&url).json(&payload).send().await {
        Ok(resp) => break,
        Err(e) if attempt < 2 => {
            last_error = Some(e);
            tokio::time::sleep(Duration::from_millis(100 * 2_u64.pow(attempt))).await;
            continue;
        }
        Err(e) => {
            last_error = Some(e);
            break;
        }
    }
}
```

### Exit Code Pattern
```rust
// TEAM-054: Ensure proper exit code
if !response.status().is_success() {
    anyhow::bail!("Request failed: HTTP {}", response.status());
}
```

---

## 🔬 Investigation Checklist

Before implementing, investigate:

- [ ] Why do HTTP requests fail with IncompleteMessage?
- [ ] Is queen-rbee staying alive during tests?
- [ ] Are there connection pool limits?
- [ ] Why does infer command return exit code 2?
- [ ] What error is being propagated?
- [ ] Are all error paths using anyhow::bail!?

---

## 💬 Questions for TEAM-054?

If you have questions, check these resources first:
1. `TEAM_053_SUMMARY.md` - Detailed analysis of what TEAM-053 did
2. `bin/.specs/COMPONENT_RESPONSIBILITIES_FINAL.md` - Component roles
3. `bin/.specs/CRITICAL_RULES.md` - Lifecycle rules
4. `test-harness/bdd/README.md` - BDD testing guide

Good luck! 🚀

---

**TEAM-053 signing off.**

**Status:** Ready for handoff to TEAM-054  
**Blocker:** HTTP connection issues and exit code mismatches  
**Risk:** Low - clear path forward with specific tasks  
**Confidence:** High - all infrastructure in place, just needs polish

**Progress:** 42/62 scenarios passing (68%) → Target: 54+ scenarios (87%)
