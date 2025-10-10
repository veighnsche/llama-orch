# Mocks That Need Replacing with Real Implementation

**Created by:** TEAM-059  
**Date:** 2025-10-10  
**Purpose:** Catalog of all mock/stub implementations that should be replaced with real components

---

## Priority Classification

**🔴 P0 - Critical for 62/62 passing** - Must be replaced  
**🟡 P1 - Important for real testing** - Should be replaced  
**🟢 P2 - Nice to have** - Can remain mock for now

---

## 🔴 P0: Critical Mocks (Block Test Passing)

### 1. Edge Case Exit Codes ❌ FAKE

**File:** `src/steps/edge_cases.rs`

**Current (WRONG):**
```rust
#[when(expr = "attempt connection to unreachable node")]
pub async fn when_attempt_connection(world: &mut World) {
    world.last_exit_code = Some(1);  // FAKE!
}
```

**What's needed (RIGHT):**
```rust
#[when(expr = "attempt connection to unreachable node")]
pub async fn when_attempt_connection(world: &mut World) {
    // TEAM-060: Execute REAL SSH command
    let result = tokio::process::Command::new("ssh")
        .arg("-o").arg("ConnectTimeout=1")
        .arg("unreachable.invalid")
        .arg("echo test")
        .output()
        .await
        .expect("Failed to execute ssh");
    
    world.last_exit_code = result.status.code();  // REAL!
}
```

**Impact:** 5-10 failing scenarios depend on real exit codes

**Edge cases to fix:**
- EC1: Connection timeout
- EC2: Download failure
- EC3: VRAM check failure
- EC6: Queue full
- EC7: Model loading timeout
- EC8: Version mismatch
- EC9: Invalid API key

---

### 2. CLI Command Execution ❌ FAKE

**File:** `src/steps/cli_commands.rs`

**Current:** Uses `tracing::debug!()` for most steps

**Examples:**
```rust
#[then(expr = "the command executes the full inference flow")]
pub async fn then_execute_full_flow(world: &mut World) {
    tracing::debug!("Should execute full inference flow");  // FAKE!
}

#[then(expr = "tokens are streamed to stdout")]
pub async fn then_tokens_streamed_stdout(world: &mut World) {
    tracing::debug!("Tokens should be streamed to stdout");  // FAKE!
}
```

**What's needed:**
- Actually execute commands and verify output
- Check stdout/stderr for expected content
- Verify exit codes

**Impact:** 3-5 failing scenarios

---

### 3. Worker Ready Callback ❌ PARTIALLY FAKE

**File:** `src/steps/happy_path.rs:245`

**Current:**
```rust
#[then(expr = "the worker sends ready callback to {string}")]
pub async fn then_worker_ready_callback(world: &mut World, url: String) {
    // Mock: ready callback sent
    world.workers.insert("worker-abc123".to_string(), ...);  // FAKE!
}
```

**What's needed:**
- Wait for REAL callback from spawned mock-worker
- Verify callback was received by queen-rbee
- Check worker appears in registry

**Impact:** 2-3 failing scenarios

---

## 🟡 P1: Important Mocks (Improve Test Quality)

### 4. SSH Query/Installation ❌ FAKE

**File:** `src/steps/beehive_registry.rs`

**Examples:**
```rust
#[then(expr = "queen-rbee executes SSH installation commands:")]
pub async fn then_ssh_installation(world: &mut World, step: &cucumber::gherkin::Step) {
    // Mock: simulate SSH installation
    tracing::info!("✅ Mock SSH installation executed");  // FAKE!
    world.last_exit_code = Some(0);
}
```

**What's needed:**
- Use real SSH mock server or actual SSH commands
- Verify commands are executed
- Check real exit codes

---

### 5. HTTP Queries to Worker Registry ❌ FAKE

**File:** `src/steps/happy_path.rs:53`

**Current:**
```rust
#[then(expr = "queen-rbee queries rbee-hive worker registry at {string}")]
pub async fn then_query_worker_registry(world: &mut World, url: String) {
    // Mock: simulate HTTP request
    world.last_http_response = Some(serde_json::json!({"workers": []}).to_string());
}
```

**What's needed:**
- Make REAL HTTP GET to mock rbee-hive
- Get actual worker list from spawned workers
- Verify response format

---

### 6. Model Download Progress ❌ FAKE

**File:** `src/steps/happy_path.rs:99-122`

**Current:**
```rust
#[then(expr = "rbee-hive downloads the model from Hugging Face")]
pub async fn then_download_from_hf(world: &mut World) {
    // Mock: initiate download
    tracing::info!("✅ Mock download initiated");  // FAKE!
}

#[then(expr = "rbee-keeper displays a progress bar...")]
pub async fn then_display_progress_bar(world: &mut World) {
    // Mock: display progress bar
    tracing::info!("✅ Mock progress bar: [████████----] 40%");  // FAKE!
}
```

**What's needed:**
- Simulate download with actual file I/O
- Generate real SSE progress events
- Verify progress bar rendering

---

### 7. Worker Preflight Checks ❌ FAKE

**File:** `src/steps/happy_path.rs:153-175`

**Current:**
```rust
#[then(expr = "RAM check passes with {int} MB available")]
pub async fn then_ram_check_passes(world: &mut World, ram_mb: usize) {
    // Mock: RAM check passes
    tracing::info!("✅ RAM check passed: {} MB", ram_mb);  // FAKE!
}

#[then(expr = "CUDA backend check passes")]
pub async fn then_cuda_check_passes(world: &mut World) {
    // Mock: backend check passes
    tracing::info!("✅ CUDA backend check passed");  // FAKE!
}
```

**What's needed:**
- Check actual system RAM (or mock it realistically)
- Verify CUDA/Metal availability
- Return real check results

---

### 8. Worker HTTP Server Startup ❌ FAKE

**File:** `src/steps/happy_path.rs:238`

**Current:**
```rust
#[then(expr = "the worker HTTP server starts on port {int}")]
pub async fn then_worker_http_starts(world: &mut World, port: u16) {
    // Mock: HTTP server started
    tracing::info!("✅ Worker HTTP server started on port {}", port);  // FAKE!
}
```

**What's needed:**
- Verify mock-worker actually started
- Check port is listening
- Test health endpoint responds

**Note:** TEAM-059 built mock-worker, but this step doesn't verify it started

---

## 🟢 P2: Nice to Have (Can Stay Mock)

### 9. Model Catalog Operations

**File:** `src/steps/happy_path.rs:86-151`

**Current:** Uses in-memory `world.model_catalog`

**Status:** Acceptable for now - tests catalog logic, not actual DB

---

### 10. Worker Registry Operations

**File:** `src/steps/happy_path.rs:263-279`

**Current:** Uses in-memory `world.workers`

**Status:** Acceptable for now - queen-rbee has real registry

---

### 11. Configuration File Handling

**File:** `src/steps/cli_commands.rs:8-18`

**Current:** Uses `tracing::debug!()` for config validation

**Status:** Low priority - config loading is tested elsewhere

---

### 12. Error Response Formatting

**File:** `src/steps/error_responses.rs`

**Current:** All `tracing::debug!()` stubs

**Status:** Low priority - error format is tested in unit tests

---

## Summary Statistics

| Priority | Category | Count | Impact |
|----------|----------|-------|--------|
| 🔴 P0 | Critical | 3 | 10-18 failing scenarios |
| 🟡 P1 | Important | 5 | Better test quality |
| 🟢 P2 | Nice to have | 4 | Low priority |
| **Total** | | **12** | |

---

## Recommended Implementation Order

### Phase 5A: Fix Exit Codes (2-3 hours)

1. **Edge case exit codes** (src/steps/edge_cases.rs)
   - Replace all `world.last_exit_code = Some(1)` with real commands
   - Test each edge case individually
   - Expected: +5-10 passing scenarios

2. **CLI command execution** (src/steps/cli_commands.rs)
   - Execute real commands where possible
   - Verify stdout/stderr content
   - Expected: +3-5 passing scenarios

### Phase 5B: Fix Worker Integration (1-2 hours)

3. **Worker ready callback** (src/steps/happy_path.rs)
   - Wait for real callback from mock-worker
   - Query queen-rbee registry to verify
   - Expected: +2-3 passing scenarios

4. **Worker HTTP verification** (src/steps/happy_path.rs)
   - Check mock-worker port is listening
   - Test health endpoint
   - Expected: +1-2 passing scenarios

### Phase 5C: Optional Improvements (2-3 hours)

5. **HTTP queries** - Make real calls to mock rbee-hive
6. **SSH operations** - Use real SSH mock server
7. **Model download** - Simulate with real file I/O
8. **Preflight checks** - Check real system resources

---

## Testing Strategy

### For Each Mock Replacement:

1. **Identify the mock**
   ```bash
   grep -n "Mock:" src/steps/*.rs
   ```

2. **Write the real implementation**
   - Use actual commands/HTTP calls
   - Capture real output/exit codes
   - Store in world state

3. **Test individually**
   ```bash
   LLORCH_BDD_FEATURE_PATH=tests/features/test-001.feature cargo run --bin bdd-runner
   ```

4. **Verify improvement**
   - Check scenario pass count increases
   - Ensure no regressions

---

## Code Signature Requirement

All replacements MUST be signed:

```rust
// TEAM-060: Replaced mock with real command execution
let result = tokio::process::Command::new("ssh")
    .arg("-o").arg("ConnectTimeout=1")
    .arg("unreachable.invalid")
    .output()
    .await?;
world.last_exit_code = result.status.code();
```

---

## Expected Outcome

**Current:** 42/62 passing (68%)  
**After P0 fixes:** 52-60/62 passing (84-97%)  
**After P1 fixes:** 58-62/62 passing (94-100%)

**Target:** 62/62 passing (100%) ✅

---

**TEAM-059 identified the mocks. TEAM-060 replaces them.** 🎯

**Start with P0, then P1, then celebrate 62/62!** 🎉
