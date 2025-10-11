# TEAM-073 QUICK START GUIDE 🐝

**Date:** 2025-10-11  
**Status:** Ready to start testing!

---

## TL;DR - Start Here!

```bash
# 1. Run the tests (they won't hang anymore!)
cd test-harness/bdd
cargo run --bin bdd-runner 2>&1 | tee test_results.log

# 2. Analyze results
grep "⏱️" test_results.log        # See scenario timings
grep "❌ SCENARIO TIMEOUT" test_results.log  # See timeouts
grep "FAILED\|panicked" test_results.log     # See failures

# 3. Fix at least 10 functions
# See TEAM_073_HANDOFF.md for details

# 4. Document everything
# Create TEAM_073_TEST_RESULTS.md
# Create TEAM_073_COMPLETION.md
```

---

## What Changed - NICE!

### TEAM-072's Critical Fix

**Problem:** Tests were hanging forever  
**Solution:** Added 60-second per-scenario timeout

**Now:**
- Tests timeout after 60s (no more infinite hangs!)
- Clear error messages when timeout occurs
- Automatic process cleanup
- Timing logged for every scenario

---

## Your Mission - NICE!

### Required Work

1. **Run Tests** - Execute full test suite
2. **Document Results** - Create `TEAM_073_TEST_RESULTS.md`
3. **Fix 10+ Functions** - Replace logging-only implementations
4. **Create Completion Report** - `TEAM_073_COMPLETION.md`

### Success Criteria

- [ ] Ran full test suite at least once
- [ ] Documented all test results
- [ ] Fixed at least 10 functions with real API calls
- [ ] All functions have "TEAM-073: ... NICE!" signature
- [ ] `cargo check --bin bdd-runner` passes (0 errors)
- [ ] Created both required markdown files

---

## Known Issues to Fix - NICE!

### High-Priority TODO Items

**File:** `happy_path.rs`
1. Line 122: `then_pool_preflight_check` - Make real HTTP request
2. Line 162: `then_download_progress_stream` - Connect to SSE
3. Line 411: `then_stream_loading_progress` - Connect to worker SSE
4. Line 463: `then_stream_tokens` - Connect to inference SSE

**File:** `model_provisioning.rs`
5. Line 358: `then_if_retries_fail_return_error` - Verify error

**File:** `registry.rs`
6-10. Multiple functions with only `tracing::debug!()` calls

### Statistics

- **Total Functions:** ~383 functions in `src/steps/`
- **Real Implementations:** ~123 (32%)
- **Logging-Only:** ~260 (68%)
- **TODO Markers:** 8 explicit TODOs

---

## Quick Commands - NICE!

### Run Tests

```bash
# Full suite (will timeout properly now!)
cargo run --bin bdd-runner

# Specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/test-001.feature \
  cargo run --bin bdd-runner

# With verbose logging
RUST_LOG=debug cargo run --bin bdd-runner
```

### Check Your Work

```bash
# Count your functions
grep -r "TEAM-073:" src/steps/ | wc -l

# Check compilation
cargo check --bin bdd-runner

# Find remaining TODOs
grep -r "TODO" src/steps/
```

---

## Example Fix - NICE!

### Before (Logging-Only)

```rust
#[then(expr = "queen-rbee performs pool preflight check at {string}")]
pub async fn then_pool_preflight_check(world: &mut World, url: String) {
    // TODO: Make HTTP request to rbee-hive health endpoint
    // For now, store expected response for test assertions
    world.last_http_response = Some(serde_json::json!({
        "status": "alive",
    }));
    tracing::info!("✅ Mock preflight check at: {}", url);
}
```

### After (Real Implementation)

```rust
// TEAM-073: Implement real HTTP health check NICE!
#[then(expr = "queen-rbee performs pool preflight check at {string}")]
pub async fn then_pool_preflight_check(world: &mut World, url: String) {
    let client = crate::steps::world::create_http_client();
    let health_url = format!("{}/health", url);
    
    match client.get(&health_url).send().await {
        Ok(response) => {
            let status = response.status().as_u16();
            let body = response.json::<serde_json::Value>().await
                .unwrap_or(serde_json::json!({}));
            
            world.last_http_status = Some(status);
            world.last_http_response = Some(body);
            tracing::info!("✅ Pool preflight check completed: {} NICE!", status);
        }
        Err(e) => {
            world.last_error = Some(crate::steps::world::ErrorResponse {
                code: "PREFLIGHT_FAILED".to_string(),
                message: format!("Preflight check failed: {}", e),
                details: None,
            });
            tracing::warn!("⚠️  Preflight check failed: {}", e);
        }
    }
}
```

---

## Expected Test Behavior - NICE!

### Before TEAM-072 Fix

```
🎬 Starting scenario: Add remote rbee-hive node to registry
[HANGS FOREVER - NO OUTPUT]
[Must manually kill with Ctrl+C]
```

### After TEAM-072 Fix

```
🎬 Starting scenario: Add remote rbee-hive node to registry
[... scenario runs ...]
⏱️  Scenario 'Add remote rbee-hive node to registry' completed in 2.3s

OR if it hangs:

🎬 Starting scenario: Add remote rbee-hive node to registry
[... waits 60 seconds ...]
❌ SCENARIO TIMEOUT: 'Add remote rbee-hive node to registry' exceeded 60 seconds!
❌ SCENARIO TIMEOUT DETECTED - KILLING PROCESSES
🧹 Cleaning up all test processes...
```

---

## Critical Rules - NICE!

### BDD Rules (MANDATORY)
1. ✅ Implement at least 10 functions
2. ✅ Each function MUST call real API
3. ❌ NEVER mark functions as TODO
4. ✅ Document test results

### Dev-Bee Rules (MANDATORY)
1. ✅ Add "TEAM-073: ... NICE!" signature
2. ❌ Don't remove other teams' signatures
3. ✅ Update existing files, don't create many new ones

### Timeout Awareness (NEW!)
- Tests timeout after 60s per scenario
- If timeout occurs, check logs for hung step
- Fix implementation to not wait forever
- Use `create_http_client()` for all HTTP requests (has timeouts)

---

## Workflow - NICE!

### Step 1: Run Tests (15 minutes)

```bash
cd test-harness/bdd
cargo run --bin bdd-runner 2>&1 | tee test_results.log
```

### Step 2: Analyze Results (15 minutes)

```bash
# Count scenarios
grep "🎬 Starting scenario" test_results.log | wc -l

# Find timeouts
grep "❌ SCENARIO TIMEOUT" test_results.log

# Find failures
grep "FAILED\|panicked" test_results.log

# Check timings
grep "⏱️" test_results.log
```

### Step 3: Pick Functions to Fix (10 minutes)

Priority order:
1. Functions with TODO markers (8 known)
2. Functions that caused test failures
3. Functions with only `tracing::debug!()`

### Step 4: Implement Fixes (2-3 hours)

For each function:
1. Read the step definition
2. Identify what API to call
3. Implement with error handling
4. Add "TEAM-073: ... NICE!" signature
5. Test locally

### Step 5: Re-run Tests (15 minutes)

```bash
cargo run --bin bdd-runner 2>&1 | tee test_results_after.log
```

### Step 6: Document (30 minutes)

Create:
- `TEAM_073_TEST_RESULTS.md` - Test results and analysis
- `TEAM_073_COMPLETION.md` - Functions fixed and lessons learned

---

## Available APIs - NICE!

### HTTP Client (with timeouts!)
```rust
let client = crate::steps::world::create_http_client();
let response = client.get(&url).send().await?;
```

### WorkerRegistry
```rust
let registry = world.hive_registry();
let workers = registry.list().await;
```

### File System
```rust
let bytes = std::fs::read(&path)?;
let metadata = std::fs::metadata(&path)?;
```

---

## Help & Resources - NICE!

### Read These First
1. `TEAM_073_HANDOFF.md` - Full mission details
2. `TEAM_072_COMPLETION.md` - Timeout fix details
3. `TEAM_071_COMPLETION.md` - Recent implementations

### Example Implementations
- `src/steps/gguf.rs` - TEAM-071 file operations
- `src/steps/pool_preflight.rs` - TEAM-071 HTTP checks
- `src/steps/worker_health.rs` - TEAM-070 examples

### Verification Commands
```bash
# Check compilation
cargo check --bin bdd-runner

# Count your functions
grep -r "TEAM-073:" src/steps/ | wc -l

# Run specific test
LLORCH_BDD_FEATURE_PATH=tests/features/test-001.feature \
  cargo run --bin bdd-runner
```

---

**TEAM-072 says: Tests won't hang anymore! Go test everything! NICE! 🐝**

**Time estimate: 4-5 hours total work**
