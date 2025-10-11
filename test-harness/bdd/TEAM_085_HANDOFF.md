# TEAM-085 HANDOFF - Bug Hunting & Fixing

**From:** TEAM-084  
**Date:** 2025-10-11  
**Status:** 🔴 CRITICAL - Bug hunting and fixing required

---

## Mission

**HUNT AND FIX BUGS BY COMPARING BDD TEST SPECIFICATIONS TO ACTUAL CODE**

TEAM-084 implemented critical features. Now it's time to find and fix bugs by:
1. Reading BDD test specifications (the `.feature` files)
2. Understanding what the tests expect
3. Finding bugs in the product code
4. Fixing them systematically

---

## What TEAM-084 Delivered

### ✅ Implemented Features
1. **Worker Registration** - `POST /v2/workers/register` endpoint
2. **Inference Routing** - `POST /v1/inference` endpoint
3. **Bug Fixes** - Fixed rbee-keeper compilation error
4. **Cleanup** - Fixed 28 code issues

### ✅ Current State
- ✅ Entire workspace compiles
- ✅ Core APIs exist and work
- ✅ HTTP endpoints are wired
- ⚠️ **But bugs likely exist in the implementation**

---

## Your Mission: Bug Hunting

### Phase 1: Understand the Specifications (2 hours)

**Read the BDD tests - they are your specification:**

```bash
# Start with these feature files
test-harness/bdd/tests/features/050-queen-rbee-worker-registry.feature
test-harness/bdd/tests/features/130-inference-execution.feature
test-harness/bdd/tests/features/900-integration-e2e.feature
```

**For each scenario, ask:**
1. What does the test expect to happen?
2. What API calls does it make?
3. What should the response be?
4. What edge cases does it test?

### Phase 2: Run Tests and Find Bugs (4 hours)

**Run tests one feature at a time:**

```bash
# Run worker registry tests
LLORCH_BDD_FEATURE_PATH=tests/features/050-queen-rbee-worker-registry.feature \
  cargo test --package test-harness-bdd --test cucumber 2>&1 | tee test_output.log

# Look for failures
grep -E "(FAILED|panicked|assertion)" test_output.log
```

**For each failure:**
1. Read the error message
2. Find the step definition in `test-harness/bdd/src/steps/`
3. See what API it's calling
4. Check the product code in `/bin/`
5. Find the bug
6. Fix it
7. Re-run the test

### Phase 3: Systematic Bug Fixes (6 hours)

**Fix bugs in priority order:**

#### Priority 1: Critical Bugs (P0)
1. **Crashes** - Panics, unwraps that fail
2. **Wrong behavior** - Code does opposite of what test expects
3. **Missing validation** - No error checking

#### Priority 2: Important Bugs (P1)
4. **Edge cases** - Null checks, empty lists, boundary conditions
5. **Race conditions** - Concurrent access issues
6. **Resource leaks** - Unclosed connections, memory leaks

#### Priority 3: Nice-to-have (P2)
7. **Performance** - Slow operations
8. **Logging** - Missing or incorrect log messages
9. **Error messages** - Unclear error responses

---

## Known Bug Categories to Check

### Category 1: Worker Registration Bugs

**Check these scenarios:**

1. **Duplicate registration** - What happens if same worker registers twice?
   ```rust
   // Bug: Does it overwrite or reject?
   // Expected: Should update existing worker
   ```

2. **Invalid worker data** - What if URL is malformed?
   ```rust
   // Bug: No validation on worker.url
   // Expected: Should return 400 Bad Request
   ```

3. **Concurrent registration** - Two workers register at same time?
   ```rust
   // Bug: Race condition in registry.register()?
   // Expected: Both should succeed with unique IDs
   ```

**Where to look:**
- `bin/queen-rbee/src/http/workers.rs` - `handle_register_worker()`
- `bin/queen-rbee/src/worker_registry.rs` - `register()` method

### Category 2: Inference Routing Bugs

**Check these scenarios:**

1. **No idle workers** - What if all workers are busy?
   ```rust
   // Bug: Returns 503 but doesn't queue request
   // Expected: Should queue or return helpful error
   ```

2. **Worker dies during request** - Worker crashes mid-inference?
   ```rust
   // Bug: No timeout, hangs forever
   // Expected: Should timeout and retry on another worker
   ```

3. **Invalid request** - Malformed JSON or missing fields?
   ```rust
   // Bug: Panics on missing field
   // Expected: Should return 400 with validation error
   ```

**Where to look:**
- `bin/queen-rbee/src/http/inference.rs` - `handle_inference_request()`
- `bin/llm-worker-rbee/src/http/execute.rs` - `handle_execute()`

### Category 3: Registry State Bugs

**Check these scenarios:**

1. **Stale workers** - Worker crashes but stays in registry?
   ```rust
   // Bug: No health checks, dead workers stay "idle"
   // Expected: Should mark as "dead" or remove
   ```

2. **State transitions** - Worker goes Idle → Busy → Idle?
   ```rust
   // Bug: State doesn't update after request completes
   // Expected: Should transition back to Idle
   ```

3. **Slot management** - Worker has 4 slots, 4 requests come in?
   ```rust
   // Bug: Doesn't track slots_available
   // Expected: Should decrement on allocation, increment on release
   ```

**Where to look:**
- `bin/queen-rbee/src/worker_registry.rs` - All methods
- `bin/rbee-hive/src/registry.rs` - State management

### Category 4: Error Handling Bugs

**Check these scenarios:**

1. **Network errors** - Worker unreachable?
   ```rust
   // Bug: No retry logic
   // Expected: Should retry 3 times with backoff
   ```

2. **Timeout errors** - Request takes too long?
   ```rust
   // Bug: No timeout set
   // Expected: Should timeout after 60s
   ```

3. **Validation errors** - Invalid input?
   ```rust
   // Bug: Panics instead of returning error
   // Expected: Should return 400 with clear message
   ```

**Where to look:**
- All HTTP handlers in `bin/queen-rbee/src/http/`
- All HTTP handlers in `bin/llm-worker-rbee/src/http/`

---

## Bug Hunting Workflow

### Step-by-Step Process

```bash
# 1. Pick a feature file
FEATURE="tests/features/050-queen-rbee-worker-registry.feature"

# 2. Read the scenarios
cat test-harness/bdd/$FEATURE

# 3. Run the tests
LLORCH_BDD_FEATURE_PATH=$FEATURE \
  cargo test --package test-harness-bdd --test cucumber 2>&1 | tee bug_hunt.log

# 4. Find failures
grep -A 10 "FAILED" bug_hunt.log

# 5. For each failure:
#    a. Read the error
#    b. Find the step definition
#    c. Find the product code
#    d. Identify the bug
#    e. Fix it
#    f. Add TEAM-085 signature
#    g. Re-run test

# 6. Repeat until all tests pass
```

### Example Bug Hunt

**Scenario:** Worker registration with duplicate ID

```gherkin
Scenario: Duplicate worker registration
  Given worker-001 is already registered
  When rbee-hive registers worker-001 again
  Then the registration should succeed
  And the worker details should be updated
```

**Run test:**
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/050-queen-rbee-worker-registry.feature \
  cargo test --package test-harness-bdd --test cucumber
```

**Expected failure:**
```
assertion failed: worker.url == "http://localhost:8082"
Expected: http://localhost:8082
Got: http://localhost:8081
```

**Bug identified:**
```rust
// bin/queen-rbee/src/worker_registry.rs
pub async fn register(&self, worker: WorkerInfo) {
    let mut workers = self.workers.write().await;
    workers.insert(worker.id.clone(), worker);  // ✅ This is correct
}
```

**Actually, no bug here! The test might be wrong or the step definition.**

**Check step definition:**
```rust
// test-harness/bdd/src/steps/worker_registration.rs
#[when(expr = "rbee-hive registers worker-001 again")]
pub async fn when_register_duplicate_worker(world: &mut World) {
    // Bug: Creates new worker with same ID but different URL
    // Should update existing worker's URL instead
}
```

**Fix the step definition or the product code - whichever is wrong!**

---

## Common Bug Patterns

### Pattern 1: Missing Null Checks

```rust
// ❌ Bug
let worker = registry.get(&worker_id).await;
let url = worker.url;  // Panics if worker is None

// ✅ Fix
let worker = registry.get(&worker_id).await
    .ok_or_else(|| anyhow!("Worker not found"))?;
let url = worker.url;
```

### Pattern 2: No Error Propagation

```rust
// ❌ Bug
let result = do_something().await;
// Ignores error, continues

// ✅ Fix
let result = do_something().await?;
// Propagates error up
```

### Pattern 3: Race Conditions

```rust
// ❌ Bug
let workers = registry.list().await;
// ... time passes ...
let worker = workers.first();  // Worker might be gone now

// ✅ Fix
let worker = {
    let workers = registry.list().await;
    workers.first().cloned()
};
// Worker is cloned, safe to use
```

### Pattern 4: Resource Leaks

```rust
// ❌ Bug
let mut lock = registry.workers.write().await;
// ... panic happens ...
// Lock never released

// ✅ Fix
{
    let mut lock = registry.workers.write().await;
    // ... do work ...
}  // Lock released here
```

### Pattern 5: Missing Timeouts

```rust
// ❌ Bug
let response = client.get(url).send().await?;
// Hangs forever if server doesn't respond

// ✅ Fix
let response = client.get(url)
    .timeout(Duration::from_secs(30))
    .send()
    .await?;
```

---

## Testing Strategy

### Test Each Feature Independently

```bash
# 1. Worker registry
LLORCH_BDD_FEATURE_PATH=tests/features/050-queen-rbee-worker-registry.feature \
  cargo test --package test-harness-bdd --test cucumber

# 2. Inference execution
LLORCH_BDD_FEATURE_PATH=tests/features/130-inference-execution.feature \
  cargo test --package test-harness-bdd --test cucumber

# 3. Concurrency
LLORCH_BDD_FEATURE_PATH=tests/features/200-concurrency-scenarios.feature \
  cargo test --package test-harness-bdd --test cucumber

# 4. Failure recovery
LLORCH_BDD_FEATURE_PATH=tests/features/210-failure-recovery.feature \
  cargo test --package test-harness-bdd --test cucumber

# 5. Integration
LLORCH_BDD_FEATURE_PATH=tests/features/900-integration-e2e.feature \
  cargo test --package test-harness-bdd --test cucumber
```

### Track Your Progress

Create a bug tracking document:

```markdown
# TEAM-085 Bug Tracking

## Bugs Found
1. ❌ Worker registration: No validation on URL format
2. ❌ Inference routing: No timeout on worker requests
3. ❌ Registry: Stale workers not removed

## Bugs Fixed
1. ✅ Worker registration: Added URL validation
2. ✅ Inference routing: Added 60s timeout
3. 🔄 Registry: Health checks in progress

## Tests Passing
- ✅ 050-queen-rbee-worker-registry.feature (5/10 scenarios)
- ❌ 130-inference-execution.feature (0/8 scenarios)
- ❌ 900-integration-e2e.feature (0/5 scenarios)
```

---

## Success Criteria

### Minimum Acceptable (TEAM-085)
- [ ] 10+ bugs identified
- [ ] 5+ bugs fixed
- [ ] 3+ feature files have passing scenarios
- [ ] Compilation passes
- [ ] Progress documented

### Target Goal (TEAM-085)
- [ ] 20+ bugs identified
- [ ] 15+ bugs fixed
- [ ] 5+ feature files have passing scenarios
- [ ] 50%+ of scenarios passing
- [ ] Comprehensive bug report

### Stretch Goal (TEAM-085)
- [ ] 30+ bugs identified and fixed
- [ ] 10+ feature files have passing scenarios
- [ ] 80%+ of scenarios passing
- [ ] All critical bugs fixed

---

## Key Files to Review

### Product Code (Where Bugs Live)
```
bin/queen-rbee/src/
├── http/
│   ├── workers.rs          # Worker registration endpoint
│   ├── inference.rs        # Inference routing endpoint
│   └── types.rs            # Request/response types
├── worker_registry.rs      # Worker state management
└── beehive_registry.rs     # Node registry

bin/llm-worker-rbee/src/
├── http/
│   ├── execute.rs          # Inference execution
│   ├── health.rs           # Health check endpoint
│   └── validation.rs       # Request validation
└── backend/                # Inference backend

bin/rbee-hive/src/
├── registry.rs             # Local worker registry
├── download_tracker.rs     # Model download tracking
└── provisioner/            # Model provisioning
```

### Test Specifications (What Should Happen)
```
test-harness/bdd/tests/features/
├── 050-queen-rbee-worker-registry.feature
├── 060-rbee-hive-worker-registry.feature
├── 130-inference-execution.feature
├── 200-concurrency-scenarios.feature
├── 210-failure-recovery.feature
└── 900-integration-e2e.feature
```

### Step Definitions (How Tests Call Code)
```
test-harness/bdd/src/steps/
├── worker_registration.rs
├── inference_execution.rs
├── concurrency.rs
├── failure_recovery.rs
└── integration.rs
```

---

## Anti-Patterns to Avoid

### ❌ DON'T:
1. **Change tests to match bugs** - Fix the code, not the tests
2. **Ignore edge cases** - They're in the tests for a reason
3. **Add TODO markers** - Fix the bug or document it properly
4. **Skip verification** - Always re-run tests after fixes
5. **Fix symptoms** - Find and fix root causes

### ✅ DO:
1. **Read the specs first** - Understand what should happen
2. **Fix root causes** - Don't just patch symptoms
3. **Add TEAM-085 signatures** - Mark your fixes
4. **Test incrementally** - Fix one bug, test, repeat
5. **Document bugs** - Create a bug report

---

## Bug Report Template

For each bug you find, document it:

```markdown
## Bug #1: Worker Registration Accepts Invalid URLs

**Severity:** P1 - Important  
**Feature:** Worker Registration  
**File:** `bin/queen-rbee/src/http/workers.rs:117`

**Description:**
The `handle_register_worker()` function accepts any string as a URL without validation.

**Test Scenario:**
```gherkin
Scenario: Invalid worker URL
  When rbee-hive registers worker with URL "not-a-url"
  Then the registration should fail with 400 Bad Request
```

**Current Behavior:**
- Accepts invalid URL
- Worker registered with malformed URL
- Inference requests fail later

**Expected Behavior:**
- Validate URL format
- Return 400 Bad Request
- Include helpful error message

**Root Cause:**
No validation in `handle_register_worker()`

**Fix:**
```rust
// TEAM-085: Added URL validation
if !req.url.starts_with("http://") && !req.url.starts_with("https://") {
    return (
        StatusCode::BAD_REQUEST,
        Json(RegisterWorkerResponse {
            success: false,
            message: format!("Invalid URL format: {}", req.url),
            worker_id: req.worker_id,
        }),
    ).into_response();
}
```

**Verification:**
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/050-queen-rbee-worker-registry.feature \
  cargo test --package test-harness-bdd --test cucumber
# ✅ Test passes
```
```

---

## Debugging Tools

### Useful Commands

```bash
# Run with verbose logging
RUST_LOG=debug cargo test --package test-harness-bdd --test cucumber

# Run single scenario
LLORCH_BDD_FEATURE_PATH=tests/features/050-queen-rbee-worker-registry.feature \
  cargo test --package test-harness-bdd --test cucumber

# Check for panics
cargo test --package test-harness-bdd --test cucumber 2>&1 | grep -i panic

# Check for unwraps (potential bugs)
rg "\.unwrap\(\)" bin/queen-rbee/src/

# Check for unhandled errors
rg "// TODO|FIXME|XXX" bin/
```

### Logging Tips

```rust
// Add debug logging to track down bugs
tracing::debug!("Worker state before: {:?}", worker.state);
// ... do operation ...
tracing::debug!("Worker state after: {:?}", worker.state);
```

---

## Questions?

**If stuck:**
1. Read the BDD test scenario - it tells you what should happen
2. Look at the step definition - it shows how to call the API
3. Check the product code - find where the bug is
4. Look for similar patterns - other code might have the same bug
5. Ask: "What would break this?" - think like a tester

**Key insight:** The BDD tests are your specification. If the code doesn't match the tests, the code is wrong (usually).

---

## Bottom Line

**TEAM-085's mission: Find and fix bugs by comparing tests to code.**

The tests tell you what should happen.  
The code tells you what actually happens.  
Your job is to make them match.

**Workflow:**
1. Read a feature file
2. Run the tests
3. Find failures
4. Identify bugs
5. Fix bugs
6. Re-run tests
7. Repeat

**The BDD tests are your bug detector. Use them!**

---

**Created by:** TEAM-084  
**Date:** 2025-10-11  
**Time:** 18:26  
**Next Team:** TEAM-085  
**Estimated Work:** 12+ hours (1.5-2 days)  
**Priority:** P0 - Critical for production readiness

---

## CRITICAL NOTE

**This is systematic bug hunting, not random debugging.**

1. **Specs first** - Read the BDD tests to understand requirements
2. **Run tests** - Let them tell you what's broken
3. **Find bugs** - Compare expected vs actual behavior
4. **Fix systematically** - One bug at a time, verify each fix
5. **Document** - Track what you found and fixed

**The tests are your guide. Follow them to find every bug.**

Good luck, TEAM-085. Happy bug hunting! 🐛🔍
