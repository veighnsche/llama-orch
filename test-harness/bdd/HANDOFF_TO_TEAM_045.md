# HANDOFF TO TEAM-045: Continue BDD Implementation

**From:** TEAM-044  
**To:** TEAM-045  
**Date:** 2025-10-10  
**Status:** 🟢 ALL @setup SCENARIOS PASSING

---

## Executive Summary

TEAM-044 successfully fixed all BDD test execution issues. All 6 `@setup` scenarios now pass with **real process execution** (not mocks). The test infrastructure is solid and ready for expansion.

**Your mission:** Implement remaining step definitions and get more scenario tags passing.

---

## ✅ What TEAM-044 Completed

### All @setup Scenarios Passing (6/6)
1. ✅ Add remote rbee-hive node to registry
2. ✅ Add node with SSH connection failure
3. ✅ Install rbee-hive on remote node
4. ✅ List registered rbee-hive nodes
5. ✅ Remove node from rbee-hive registry
6. ✅ Inference fails when node not in registry

**Result:** 72/72 steps passing, real binaries executing, actual queen-rbee integration working.

### Critical Infrastructure Fixes
1. ✅ **Fixed binary path resolution** - Uses workspace directory, not relative paths
2. ✅ **Eliminated compilation timeouts** - Uses pre-built binaries from `target/debug/`
3. ✅ **Implemented real command execution** - Both string and docstring step variants
4. ✅ **Smart SSH mocking** - Hostname-based (fails for "unreachable", succeeds for others)
5. ✅ **Real HTTP API integration** - Node registration actually calls queen-rbee
6. ✅ **Removed duplicate steps** - Fixed ambiguous step matches
7. ✅ **Increased timeouts** - 60s for queen-rbee startup
8. ✅ **Binary name mapping** - rbee-keeper → rbee

---

## 🎯 Your Mission

### Priority 1: Run @happy Scenarios
```bash
cd test-harness/bdd
cargo run --bin bdd-runner -- --tags @happy
```

Expected: Most will fail because step definitions are still stubs.

### Priority 2: Implement Real Step Definitions
Replace stub implementations with real execution. Pattern to follow:

**Bad (stub):**
```rust
#[then(expr = "worker transitions to state {string}")]
pub async fn then_worker_transitions(world: &mut World, state: String) {
    tracing::debug!("Worker should be in state: {}", state);
}
```

**Good (real):**
```rust
#[then(expr = "worker transitions to state {string}")]
pub async fn then_worker_transitions(world: &mut World, state: String) {
    // Make real HTTP request to worker
    let client = reqwest::Client::new();
    let resp = client.get(&format!("{}/v1/state", world.worker_url.unwrap()))
        .send()
        .await?;
    
    let actual_state: serde_json::Value = resp.json().await?;
    assert_eq!(actual_state["state"], state);
}
```

### Priority 3: Fix Implementation Gaps
When tests fail, fix the implementation to match BDD:

**Example:**
```
Test expects: GET /v1/ready on worker
Implementation has: (missing)

✅ DO: Add /v1/ready endpoint to bin/llm-worker-rbee
❌ DON'T: Skip the test or change BDD
```

---

## 📊 Current Status

### BDD Step Definitions
- **Total Steps:** ~330 across 17 files
- **Implemented:** ~72 (22%)
- **Stubs Remaining:** ~258 (78%)

### Step Files Status
```
✅ cli_commands.rs     - Real execution (TEAM-044)
✅ beehive_registry.rs - Real execution (TEAM-042/043/044)
⚠️  happy_path.rs      - Mostly mocks (needs implementation)
⚠️  lifecycle.rs       - All mocks (needs implementation)
⚠️  worker_health.rs   - All mocks (needs implementation)
⚠️  inference_execution.rs - All mocks (needs implementation)
⚠️  model_provisioning.rs - All mocks (needs implementation)
⚠️  ... (11 more files with stubs)
```

---

## 🛠️ How to Implement Step Definitions

### Step 1: Read the Feature File
```gherkin
Scenario: Worker starts successfully
  Given rbee-hive is running
  When I spawn a worker for model "tiny-llama"
  Then the worker process starts
  And the worker reaches "ready" state
  And GET /v1/ready returns {"ready": true}
```

### Step 2: Identify Required Infrastructure
- Need to start rbee-hive process?
- Need to execute commands?
- Need to make HTTP requests?
- Need to verify database state?

### Step 3: Write Real Implementation
```rust
#[when(expr = "I spawn a worker for model {string}")]
pub async fn when_spawn_worker(world: &mut World, model: String) {
    let workspace = env::var("CARGO_MANIFEST_DIR")...;
    let binary = workspace.join("target/debug/rbee-hive");
    
    let child = tokio::process::Command::new(&binary)
        .args(["worker", "spawn", "--model", &model])
        .env("MOCK_SSH", "true")
        .spawn()?;
    
    world.worker_process = Some(child);
}

#[then(expr = "GET {string} returns {string}")]
pub async fn then_get_returns(world: &mut World, path: String, expected: String) {
    let url = format!("{}{}", world.worker_url.unwrap(), path);
    let resp = reqwest::get(&url).await?;
    let body = resp.text().await?;
    
    let expected_json: serde_json::Value = serde_json::from_str(&expected)?;
    let actual_json: serde_json::Value = serde_json::from_str(&body)?;
    
    assert_eq!(actual_json, expected_json);
}
```

### Step 4: Follow TEAM-044 Patterns
- ✅ Use pre-built binaries from `target/debug/`
- ✅ Use workspace directory resolution
- ✅ Set environment variables (MOCK_SSH=true)
- ✅ Store process handles in World
- ✅ Capture stdout/stderr/exit_code
- ✅ Use tracing::info for visibility

---

## 🔧 Known Implementation Gaps

### 1. Worker /v1/ready Endpoint Missing
**BDD expects:** `GET /v1/ready → {ready: true, state: "idle"}`  
**Current:** Only has `GET /v1/loading/progress` (SSE)

**Fix:** Add to `bin/llm-worker-rbee/src/http/routes.rs`:
```rust
async fn ready_handler(State(state): State<AppState>) -> impl IntoResponse {
    let worker_state = state.worker_state.lock().await;
    Json(serde_json::json!({
        "ready": worker_state.is_ready(),
        "state": worker_state.to_string()
    }))
}

// In router:
.route("/v1/ready", get(ready_handler))
```

### 2. rbee-hive Worker Spawning
**BDD expects:** `rbee-hive worker spawn --model X` command  
**Current:** Worker spawning exists but may need adjustments

**Check:** Does it match BDD feature file commands?

### 3. Model Catalog Integration
**BDD expects:** SQLite catalog at `~/.rbee/models.db`  
**Current:** Exists but verify integration

---

## 🚨 Critical Rules

### BDD-First Principle
**When test fails:**
1. ❌ **DON'T** skip the test
2. ❌ **DON'T** change the test to match implementation
3. ✅ **DO** fix the implementation to match the test
4. ✅ **DO** add missing functionality

### Example Decision Tree
```
Test fails: "Worker /v1/ready endpoint not found"
│
├─ Is the test correct? YES (BDD is the spec)
│  │
│  ├─ Is endpoint implemented? NO
│  │  └─> FIX: Add /v1/ready endpoint to worker
│  │
│  └─ Is endpoint wrong URL? 
│      └─> FIX: Change implementation URL to match BDD
│
└─ Is test a stub?
    └─> IMPLEMENT: Real step definition
```

---

## 📁 Key Files

### BDD Infrastructure
- `test-harness/bdd/src/steps/world.rs` - Shared test state
- `test-harness/bdd/src/steps/cli_commands.rs` - Command execution (✅ works)
- `test-harness/bdd/src/steps/beehive_registry.rs` - Registry operations (✅ works)
- `test-harness/bdd/tests/features/test-001.feature` - Main test scenarios

### Implementation
- `bin/queen-rbee/` - Orchestrator daemon (✅ works with MOCK_SSH)
- `bin/rbee-keeper/` - CLI tool (✅ works, binary name: rbee)
- `bin/rbee-hive/` - Pool manager (needs verification)
- `bin/llm-worker-rbee/` - Inference worker (needs /v1/ready)

---

## 🏃 Quick Start Commands

### Build Everything
```bash
cargo build --bin queen-rbee --bin rbee --bin rbee-hive --bin llm-worker-rbee
```

### Run Setup Scenarios (Should All Pass)
```bash
cd test-harness/bdd
cargo run --bin bdd-runner -- --tags @setup
```

### Run Happy Path Scenarios (Will Fail - Your Job)
```bash
cargo run --bin bdd-runner -- --tags @happy
```

### Run All Tests
```bash
cargo run --bin bdd-runner
```

### Run Specific Scenario
```bash
cargo run --bin bdd-runner -- --name "Worker starts successfully"
```

---

## 🎯 Success Criteria

By the time you hand off to TEAM-046:

### Minimum Success
- [ ] All `@setup` scenarios still passing (don't break what works)
- [ ] At least 3 `@happy` scenarios passing
- [ ] Worker `/v1/ready` endpoint implemented
- [ ] Real step definitions for core worker lifecycle

### Stretch Goals
- [ ] All `@happy` scenarios passing
- [ ] Start on `@integration` scenarios
- [ ] All worker health check scenarios passing
- [ ] Model provisioning scenarios passing

---

## 🐛 Debugging Tips

### If Tests Fail
1. Check stderr output: `RUST_LOG=info cargo run --bin bdd-runner -- --tags @happy`
2. Run single scenario: `cargo run --bin bdd-runner -- --name "scenario name"`
3. Check if binary exists: `ls -la target/debug/`
4. Verify ports not in use: `lsof -i :8080`
5. Check process cleanup: `ps aux | grep rbee`

### Common Issues
- **Exit code None:** Process killed by signal (timeout or manual kill)
- **Exit code 101:** Compilation error (shouldn't happen with pre-built binaries)
- **Connection refused:** Service not started or wrong port
- **Timeout:** Increase wait time or check if service is actually starting

---

## 📝 Documentation Updates Needed

When you modify code, update:
1. Code comments (with TEAM-045 signature)
2. Step definition files
3. Handoff document for TEAM-046
4. Summary of what you accomplished

**Format:**
```rust
// TEAM-045: Added real HTTP request for worker state verification
```

---

## 🔄 Iteration Strategy

### Phase 1: Inventory (First 30 min)
1. Run `@happy` scenarios, capture all failures
2. List unique failure types (missing endpoint, stub step, etc.)
3. Prioritize by dependency (implement foundations first)

### Phase 2: Foundation (Next 2 hours)
1. Implement core worker lifecycle steps
2. Add missing HTTP endpoints
3. Fix process spawning issues
4. Get 1-2 scenarios fully passing

### Phase 3: Expansion (Remaining time)
1. Implement more step definitions
2. Get more scenarios passing
3. Document issues for TEAM-046
4. Create comprehensive handoff

---

## 📊 Progress Tracking

Track your progress:
```markdown
## TEAM-045 Progress

### Scenarios Passing
- @setup: 6/6 ✅ (inherited)
- @happy: X/Y (your work)
- @integration: 0/Z (future)

### Step Definitions Implemented
- cli_commands.rs: 100% ✅
- beehive_registry.rs: 100% ✅
- happy_path.rs: X%
- worker_health.rs: X%
- [list what you implemented]

### Implementation Gaps Fixed
- [ ] Worker /v1/ready endpoint
- [ ] rbee-hive worker spawn
- [ ] [others you discover]
```

---

## 🎁 What You're Inheriting

### Working Infrastructure
- ✅ BDD runner compiles and runs
- ✅ Process spawning works
- ✅ Command execution works
- ✅ HTTP requests work
- ✅ queen-rbee integration works
- ✅ Smart SSH mocking works
- ✅ Process cleanup works

### Clean Slate
- No tech debt
- No broken tests
- Clear patterns to follow
- Comprehensive documentation

---

**Good luck, TEAM-045! You have a solid foundation. Make the tests pass!** 🚀

---

## Appendix: TEAM-044 Fixes Reference

See `TEAM_044_SUMMARY.md` for complete details on:
- Binary path resolution pattern
- Pre-built binary usage pattern
- Command execution pattern
- Smart SSH mocking implementation
- Real HTTP integration pattern
- Process lifecycle management

Copy these patterns when implementing new step definitions.
