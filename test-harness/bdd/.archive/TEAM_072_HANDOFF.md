# TEAM-072 HANDOFF - TESTING & VALIDATION PHASE! 🐝

**From:** TEAM-071  
**To:** TEAM-072  
**Date:** 2025-10-11  
**Status:** All 123 known functions complete - Now test everything!

---

## Your Mission - NICE!

**All 123 functions are implemented! Your mission is to TEST everything we've built.**

TEAM-071 completed the final 36 known functions (360% of minimum). Now we need to verify that all implementations work correctly with:

1. **SSH Testing** - Test against `workstation.apra.home`
2. **Local Inference Testing** - Run inference tests locally
3. **Integration Testing** - Verify all BDD scenarios pass
4. **Audit & Fix** - Find and fix any broken implementations

---

## What TEAM-071 Completed - NICE!

### Priorities 15-17 (100% Complete)
- ✅ Priority 15: GGUF (20/20)
- ✅ Priority 16: Pool Preflight (15/15)
- ✅ Priority 17: Background (1/1)

**Total: 36 functions with real API calls**

---

## Testing Priorities - NICE!

### Priority 1: SSH Testing Against workstation.apra.home 🎯

**Target:** `workstation.apra.home`  
**Purpose:** Verify SSH connectivity, remote command execution, and beehive registry operations

#### Test Scenarios

1. **SSH Connection Test**
   ```bash
   # Test SSH connectivity
   ssh vince@workstation.apra.home "echo 'SSH connection OK'"
   ```

2. **Beehive Registry Operations**
   - Test node registration via SSH
   - Test remote rbee-hive installation
   - Test SSH key authentication
   - Test remote command execution

3. **BDD Features to Run**
   ```bash
   cd test-harness/bdd
   
   # Test beehive registry features
   LLORCH_BDD_FEATURE_PATH=tests/features/beehive_registry.feature \
     cargo test --bin bdd-runner
   
   # Test SSH-related scenarios
   LLORCH_BDD_FEATURE_PATH=tests/features/pool_preflight.feature \
     cargo test --bin bdd-runner
   ```

#### Configuration Required

Update `tests/features/` to use real SSH target:

```gherkin
Given node "workstation" is at "workstation.apra.home"
And SSH user is "vince"
And SSH key is at "~/.ssh/id_rsa"
When rbee-keeper connects to "workstation.apra.home"
Then SSH connection succeeds
```

### Priority 2: Local Inference Testing 🎯

**Target:** Local machine  
**Purpose:** Verify inference execution, model loading, and worker lifecycle

#### Test Scenarios

1. **Model Loading Test**
   ```bash
   # Test GGUF model loading
   LLORCH_BDD_FEATURE_PATH=tests/features/gguf.feature \
     cargo test --bin bdd-runner
   ```

2. **Worker Lifecycle Test**
   ```bash
   # Test worker startup and registration
   LLORCH_BDD_FEATURE_PATH=tests/features/worker_startup.feature \
     cargo test --bin bdd-runner
   ```

3. **Inference Execution Test**
   ```bash
   # Test local inference
   LLORCH_BDD_FEATURE_PATH=tests/features/inference_execution.feature \
     cargo test --bin bdd-runner
   ```

4. **Happy Path Test**
   ```bash
   # Test complete workflow
   LLORCH_BDD_FEATURE_PATH=tests/features/happy_path.feature \
     cargo test --bin bdd-runner
   ```

#### Prerequisites

- ✅ Local model available (e.g., `~/models/tinyllama-1.1b-q4_0.gguf`)
- ✅ rbee-hive binary built (`cargo build --bin rbee-hive`)
- ✅ llm-worker-rbee binary built (`cargo build --bin llm-worker-rbee`)
- ✅ queen-rbee running locally (port 8000)

### Priority 3: Integration Testing 🎯

**Purpose:** Run ALL BDD features and verify they pass

```bash
cd test-harness/bdd

# Run all tests
cargo test --bin bdd-runner -- --nocapture

# Or run specific feature sets
for feature in tests/features/*.feature; do
  echo "Testing: $feature"
  LLORCH_BDD_FEATURE_PATH="$feature" cargo test --bin bdd-runner
done
```

### Priority 4: Audit & Fix Broken Implementations

**Purpose:** Find functions that don't work correctly and fix them

#### Audit Process

1. **Run tests and capture failures**
   ```bash
   cargo test --bin bdd-runner 2>&1 | tee test_results.log
   ```

2. **Identify failing scenarios**
   ```bash
   grep -A 5 "FAILED" test_results.log
   ```

3. **Fix implementations**
   - Update functions to use correct APIs
   - Add missing error handling
   - Fix borrow checker issues

4. **Re-test until green**
   ```bash
   cargo test --bin bdd-runner
   ```

---

## Test Environment Setup - NICE!

### SSH Configuration for workstation.apra.home

1. **Verify SSH Access**
   ```bash
   # Test basic SSH connection
   ssh vince@workstation.apra.home "hostname"
   
   # Should output: workstation
   ```

2. **Set Up SSH Config** (Optional but recommended)
   ```bash
   # Add to ~/.ssh/config
   cat >> ~/.ssh/config <<EOF
   
   Host workstation
       HostName workstation.apra.home
       User vince
       IdentityFile ~/.ssh/id_rsa
       ServerAliveInterval 60
   EOF
   
   # Test with short name
   ssh workstation "hostname"
   ```

3. **Environment Variables for Tests**
   ```bash
   # Set these before running SSH tests
   export LLORCH_SSH_TEST_HOST="workstation.apra.home"
   export LLORCH_SSH_TEST_USER="vince"
   export LLORCH_SSH_TEST_KEY="~/.ssh/id_rsa"
   ```

### Local Inference Setup

1. **Download Test Model**
   ```bash
   # Create models directory
   mkdir -p ~/models
   
   # Download a small test model (TinyLlama 1.1B Q4_0)
   cd ~/models
   wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
   ```

2. **Build Required Binaries**
   ```bash
   cd /home/vince/Projects/llama-orch
   
   # Build all required binaries
   cargo build --release --bin queen-rbee
   cargo build --release --bin rbee-hive
   cargo build --release --bin llm-worker-rbee
   ```

3. **Start queen-rbee**
   ```bash
   # In a separate terminal
   ./target/release/queen-rbee --port 8000
   ```

4. **Set Environment Variables**
   ```bash
   export LLORCH_MODELS_DIR="$HOME/models"
   export LLORCH_QUEEN_URL="http://127.0.0.1:8000"
   ```

---

## Testing Workflow - NICE!

### Day 1: SSH Testing

**Goal:** Verify all SSH-related functions work with workstation.apra.home

```bash
# 1. Verify SSH connectivity
ssh vince@workstation.apra.home "echo 'Connection OK'"

# 2. Run beehive registry tests
cd test-harness/bdd
LLORCH_SSH_TEST_HOST="workstation.apra.home" \
LLORCH_SSH_TEST_USER="vince" \
LLORCH_BDD_FEATURE_PATH=tests/features/beehive_registry.feature \
  cargo test --bin bdd-runner -- --nocapture

# 3. Document failures
# Create TEAM_072_SSH_TEST_RESULTS.md with:
# - Which scenarios passed
# - Which scenarios failed
# - Error messages
# - Functions that need fixing
```

### Day 2: Local Inference Testing

**Goal:** Verify all inference functions work locally

```bash
# 1. Start queen-rbee
./target/release/queen-rbee --port 8000 &

# 2. Run GGUF tests
cd test-harness/bdd
LLORCH_MODELS_DIR="$HOME/models" \
LLORCH_BDD_FEATURE_PATH=tests/features/gguf.feature \
  cargo test --bin bdd-runner -- --nocapture

# 3. Run worker lifecycle tests
LLORCH_BDD_FEATURE_PATH=tests/features/worker_startup.feature \
  cargo test --bin bdd-runner -- --nocapture

# 4. Run inference tests
LLORCH_BDD_FEATURE_PATH=tests/features/inference_execution.feature \
  cargo test --bin bdd-runner -- --nocapture

# 5. Document results
# Create TEAM_072_INFERENCE_TEST_RESULTS.md
```

### Day 3: Integration & Fix

**Goal:** Run all tests, identify failures, fix implementations

```bash
# 1. Run ALL tests
cd test-harness/bdd
cargo test --bin bdd-runner -- --nocapture 2>&1 | tee full_test_results.log

# 2. Analyze failures
grep -B 5 -A 10 "FAILED\|panicked" full_test_results.log > failures.txt

# 3. Fix broken functions
# For each failure:
# - Identify the function
# - Review the implementation
# - Fix the issue
# - Re-test

# 4. Re-run until green
cargo test --bin bdd-runner
```

---

## Expected Issues & Solutions - NICE!

### Common Issues You'll Encounter

1. **SSH Authentication Failures**
   - **Issue:** SSH key not found or permissions wrong
   - **Fix:** `chmod 600 ~/.ssh/id_rsa`

2. **Model Not Found**
   - **Issue:** GGUF file path incorrect
   - **Fix:** Update `LLORCH_MODELS_DIR` to correct path

3. **Port Already in Use**
   - **Issue:** queen-rbee port 8000 already taken
   - **Fix:** `killall queen-rbee` or use different port

4. **Worker Registration Timeout**
   - **Issue:** Worker takes too long to start
   - **Fix:** Increase timeout in test or use smaller model

5. **Borrow Checker Errors**
   - **Issue:** Functions hold borrows too long
   - **Fix:** Use scoped borrows like TEAM-071 did

### Debugging Tips

```bash
# Enable verbose logging
export RUST_LOG=debug

# Run single scenario
LLORCH_BDD_FEATURE_PATH=tests/features/gguf.feature \
  cargo test --bin bdd-runner -- --nocapture test_name

# Check process status
ps aux | grep -E "queen-rbee|rbee-hive|llm-worker"

# Check port usage
lsof -i :8000
```

---

## Available APIs - NICE!

### WorkerRegistry (`rbee_hive::registry`)
```rust
let registry = world.hive_registry();

// Full CRUD operations
let workers = registry.list().await;
let worker = registry.get(&id).await;
registry.register(worker_info).await;
registry.update_state(&id, state).await;
registry.remove(&id).await;
let idle = registry.get_idle_workers().await;
```

### HTTP Client
```rust
let client = crate::steps::world::create_http_client();

// GET request
let response = client.get(&url).send().await?;
let status = response.status().as_u16();
let body = response.text().await?;

// POST request
let response = client.post(&url)
    .json(&payload)
    .send().await?;
```

### File System Operations
```rust
// Read file
let bytes = std::fs::read(&path)?;
let metadata = std::fs::metadata(&path)?;

// Write file
let mut file = std::fs::File::create(&path)?;
file.write_all(b"content")?;

// Directory operations
std::fs::create_dir_all(&dir)?;
let entries = std::fs::read_dir(&dir)?;
```

### World State
```rust
// Model catalog
world.model_catalog.insert(ref, entry);

// Topology
world.topology.insert(node, info);

// Error tracking
world.last_error = Some(ErrorResponse { ... });
world.last_exit_code = Some(1);

// HTTP state
world.last_http_status = Some(200);
world.last_http_response = Some(body);
```

---

## Critical Rules - NICE!

### ⚠️ BDD Rules (MANDATORY)
1. ✅ **Implement at least 10 functions** - No exceptions
2. ✅ **Each function MUST call real API** - No `tracing::debug!()` only
3. ❌ **NEVER mark functions as TODO** - Implement or leave for next team
4. ❌ **NEVER delete checklist items** - Update status only
5. ✅ **Handoff must be 2 pages or less** - Be concise
6. ✅ **Include code examples** - Show the pattern

### ⚠️ Dev-Bee Rules (MANDATORY)
1. ✅ **Add team signature** - "TEAM-072: [Description] NICE!"
2. ❌ **Don't remove other teams' signatures** - Preserve history
3. ✅ **Update existing files** - Don't create multiple .md files
4. ✅ **Follow priorities** - Start with highest impact

---

## Verification Commands - NICE!

### Check Compilation
```bash
cd test-harness/bdd
cargo check --bin bdd-runner
```

Should output: `Finished \`dev\` profile [unoptimized + debuginfo] target(s)`

### Count Your Functions
```bash
grep -r "TEAM-072:" src/steps/ | wc -l
```

Should be at least 10!

### Identify Logging-Only Functions
```bash
# Find functions with only tracing calls
grep -A 10 "pub async fn" src/steps/happy_path.rs | \
  grep -B 5 "tracing::debug\|tracing::info" | \
  grep "pub async fn"
```

---

## Success Checklist - NICE!

Before creating your handoff, verify:

- [ ] Implemented at least 10 functions
- [ ] Each function calls real API (not just tracing::debug!)
- [ ] All functions have "TEAM-072: ... NICE!" signature
- [ ] `cargo check --bin bdd-runner` passes (0 errors)
- [ ] Created `TEAM_072_COMPLETION.md` (2 pages max)
- [ ] No TODO markers added to code
- [ ] Honest completion ratios shown

---

## Example Implementation - NICE!

Here's a complete example for happy_path.rs:

```rust
// TEAM-072: Verify health check response NICE!
#[then(expr = "queen-rbee returns health check OK")]
pub async fn then_health_check_response(world: &mut World) {
    // Verify HTTP status is 200
    assert_eq!(world.last_http_status, Some(200), 
        "Expected health check to return 200 OK");
    
    // Verify response body contains expected fields
    if let Some(ref body) = world.last_http_response {
        assert!(body.contains("status") || body.contains("ok"),
            "Expected health check response to contain status");
        tracing::info!("✅ Health check response verified NICE!");
    } else {
        panic!("No HTTP response available");
    }
}
```

---

## Success Criteria - NICE!

Before creating your completion report, verify:

- [ ] SSH tests run successfully against workstation.apra.home
- [ ] Local inference tests pass with real model
- [ ] All BDD features run without panics
- [ ] Documented all test results (pass/fail)
- [ ] Fixed at least 10 broken implementations
- [ ] Created `TEAM_072_TEST_RESULTS.md` with:
  - SSH test results
  - Inference test results
  - Integration test results
  - List of fixes made
  - Remaining issues (if any)

---

## Summary - NICE!

**Current Progress:**
- TEAM-068: 43 functions
- TEAM-069: 21 functions
- TEAM-070: 23 functions
- TEAM-071: 36 functions
- **Total: 123 functions (100% implemented)**

**Your Goal:**
- Test SSH against workstation.apra.home
- Test local inference with real models
- Run full integration test suite
- Fix broken implementations
- Document all results

**Recommended Workflow:**
1. Day 1: SSH testing (workstation.apra.home)
2. Day 2: Local inference testing
3. Day 3: Integration testing & fixes

---

## Deliverables Expected

1. **`TEAM_072_SSH_TEST_RESULTS.md`**
   - SSH connectivity results
   - Beehive registry test results
   - Functions that work vs. need fixing

2. **`TEAM_072_INFERENCE_TEST_RESULTS.md`**
   - GGUF model loading results
   - Worker lifecycle results
   - Inference execution results

3. **`TEAM_072_COMPLETION.md`**
   - Summary of all testing
   - Functions fixed
   - Test pass rate
   - Remaining issues

4. **Code fixes** (if needed)
   - Update broken implementations
   - Add missing error handling
   - Fix borrow checker issues

---

**TEAM-071 says: All functions implemented! Now test everything! NICE! 🐝**

**Good luck, TEAM-072! Make sure SSH works with workstation.apra.home and inference runs locally!**
