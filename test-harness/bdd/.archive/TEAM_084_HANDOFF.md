# TEAM-084 HANDOFF - Implement Product Features & Connect to BDD Tests

**From:** TEAM-083  
**Date:** 2025-10-11  
**Status:** 🔴 CRITICAL - Product features needed for BDD tests

---

## Mission

**IMPLEMENT MISSING PRODUCT FEATURES IN `/bin/` BINARIES AND CONNECT THEM TO BDD TESTS**

The BDD test suite is 93.5% complete and ready to validate your code. However, **many product features don't exist yet** in the binaries. Your job is to:

1. **Implement missing features** in `/bin/` binaries
2. **Run BDD tests** to see what fails
3. **Fix failures** by implementing real functionality
4. **Iterate** until tests pass

---

## Current State

### BDD Tests (test-harness/bdd)
- ✅ **20 feature files** with 140 scenarios
- ✅ **686 step definitions** (~93.5% wired to APIs)
- ✅ **Integration tests** created (900-integration-e2e.feature)
- ⚠️ **Tests expect features that don't exist yet**

### Product Binaries (/bin)
- ⏳ **llm-worker-rbee/** - Partially implemented
- ⏳ **queen-rbee/** - Partially implemented
- ⏳ **rbee-hive/** - Partially implemented
- ⏳ **rbee-keeper/** - Partially implemented

---

## Priority 1: Implement Core Features (CRITICAL) 🔴

### 1.1 llm-worker-rbee (8 hours)

**Missing features:**

1. **Inference execution with slot management**
   ```rust
   // bin/llm-worker-rbee/src/inference.rs
   pub struct SlotManager {
       slots_total: usize,
       slots_available: AtomicUsize,
   }
   
   impl SlotManager {
       pub async fn allocate_slot(&self) -> Result<SlotHandle, Error> {
           // TEAM-084: Implement slot allocation
           // BDD test: when_concurrent_slot_requests()
       }
   }
   ```

2. **SSE streaming for token output**
   ```rust
   // bin/llm-worker-rbee/src/streaming.rs
   pub async fn stream_tokens(
       tx: tokio::sync::mpsc::Sender<SseEvent>,
       tokens: Vec<String>
   ) -> Result<(), Error> {
       // TEAM-084: Implement SSE token streaming
       // BDD test: then_tokens_streamed_to_client()
   }
   ```

3. **Model loading and GGUF validation**
   ```rust
   // bin/llm-worker-rbee/src/model_loader.rs
   pub async fn load_gguf_model(path: PathBuf) -> Result<Model, Error> {
       // TEAM-084: Implement GGUF loading
       // BDD test: when_worker_loads_gguf_model()
   }
   ```

4. **Health check endpoint**
   ```rust
   // bin/llm-worker-rbee/src/health.rs
   pub async fn health_check() -> Result<HealthStatus, Error> {
       // TEAM-084: Implement health endpoint
       // BDD test: when_poll_endpoint()
   }
   ```

**BDD tests to validate:**
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/130-inference-execution.feature \
  cargo test --package test-harness-bdd -- --nocapture
```

### 1.2 queen-rbee (6 hours)

**Missing features:**

1. **Worker registry (in-memory or SQLite)**
   ```rust
   // bin/queen-rbee/src/registry.rs
   pub struct WorkerRegistry {
       workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
   }
   
   impl WorkerRegistry {
       pub async fn register(&self, worker: WorkerInfo) -> Result<(), Error> {
           // TEAM-084: Implement worker registration
           // BDD test: when_rbee_hive_reports_worker()
       }
       
       pub async fn get(&self, id: &str) -> Option<WorkerInfo> {
           // TEAM-084: Implement worker lookup
           // BDD test: then_routes_to_worker_integration()
       }
   }
   ```

2. **Request routing logic**
   ```rust
   // bin/queen-rbee/src/router.rs
   pub async fn route_request(
       registry: &WorkerRegistry,
       request: InferenceRequest
   ) -> Result<String, Error> {
       // TEAM-084: Implement request routing
       // BDD test: then_routes_to_worker_integration()
   }
   ```

3. **Failover detection**
   ```rust
   // bin/queen-rbee/src/failover.rs
   pub async fn detect_crash(
       registry: &WorkerRegistry,
       worker_id: &str
   ) -> Result<bool, Error> {
       // TEAM-084: Implement crash detection
       // BDD test: then_detects_crash_within()
   }
   ```

**BDD tests to validate:**
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/050-queen-rbee-worker-registry.feature \
  cargo test --package test-harness-bdd -- --nocapture
```

### 1.3 rbee-hive (8 hours)

**Missing features:**

1. **Local worker registry**
   ```rust
   // bin/rbee-hive/src/registry.rs
   pub struct WorkerRegistry {
       db_path: PathBuf,
   }
   
   impl WorkerRegistry {
       pub async fn list(&self) -> Vec<WorkerInfo> {
           // TEAM-084: Implement worker listing
           // BDD test: given_worker_ready_idle()
       }
   }
   ```

2. **Model provisioner with download tracking**
   ```rust
   // bin/rbee-hive/src/provisioner.rs
   pub struct ModelProvisioner {
       base_dir: PathBuf,
   }
   
   impl ModelProvisioner {
       pub fn find_local_model(&self, reference: &str) -> Option<PathBuf> {
           // TEAM-084: Implement model lookup
           // BDD test: given_model_catalog_contains()
       }
   }
   ```

3. **Download tracker**
   ```rust
   // bin/rbee-hive/src/download_tracker.rs
   pub struct DownloadTracker {
       active_downloads: Arc<RwLock<HashMap<String, DownloadProgress>>>,
   }
   
   impl DownloadTracker {
       pub async fn start_download(&self) -> String {
           // TEAM-084: Implement download tracking
           // BDD test: when_initiate_download()
       }
   }
   ```

**BDD tests to validate:**
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/030-model-provisioner.feature \
  cargo test --package test-harness-bdd -- --nocapture
```

### 1.4 rbee-keeper (4 hours)

**Missing features:**

1. **CLI commands**
   ```rust
   // bin/rbee-keeper/src/cli.rs
   pub async fn cmd_inference(args: InferenceArgs) -> Result<(), Error> {
       // TEAM-084: Implement inference CLI command
       // BDD test: when_client_sends_request_integration()
   }
   ```

2. **HTTP client for API calls**
   ```rust
   // bin/rbee-keeper/src/client.rs
   pub async fn send_inference_request(
       url: &str,
       payload: InferenceRequest
   ) -> Result<InferenceResponse, Error> {
       // TEAM-084: Implement HTTP client
       // BDD test: when_client_sends_inference_request()
   }
   ```

**BDD tests to validate:**
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/150-cli-commands.feature \
  cargo test --package test-harness-bdd -- --nocapture
```

---

## Priority 2: Run BDD Tests & Fix Failures (CRITICAL) 🔴

### 2.1 Run Integration Tests First

**Start with the integration tests:**
```bash
LLORCH_BDD_FEATURE_PATH=tests/features/900-integration-e2e.feature \
  cargo test --package test-harness-bdd -- --nocapture
```

**Expected result:** Most tests will fail because features don't exist

### 2.2 Implement Features Iteratively

**For each failing test:**
1. Read the error message
2. Find the step definition in `test-harness/bdd/src/steps/`
3. See what API it's calling
4. Implement that API in the appropriate binary
5. Re-run the test
6. Repeat until test passes

**Example workflow:**
```bash
# Run test
LLORCH_BDD_FEATURE_PATH=tests/features/900-integration-e2e.feature \
  cargo test --package test-harness-bdd -- --nocapture

# See failure: "worker-001 not found in registry"
# → Implement WorkerRegistry::get() in queen-rbee

# Re-run test
LLORCH_BDD_FEATURE_PATH=tests/features/900-integration-e2e.feature \
  cargo test --package test-harness-bdd -- --nocapture

# See next failure: "inference endpoint not found"
# → Implement inference endpoint in llm-worker-rbee

# Continue...
```

### 2.3 Update Step Definitions If Needed

**If you change an API signature:**
```rust
// Old API in step definition
let result = registry.get("worker-001").await;

// New API you implemented
let result = registry.get("worker-001").await?;

// Update step definition to match
let result = registry.get("worker-001").await
    .expect("Failed to get worker");
```

**Add TEAM-084 signatures:**
```rust
// TEAM-084: Updated to match new registry API
let result = registry.get("worker-001").await
    .expect("Failed to get worker");
```

---

## Priority 3: Wire Remaining Stub Functions (LOW) 🟢

**9 functions still need wiring (~6.5%):**

These are lower priority - focus on implementing product features first.

```bash
# Find remaining stubs
rg "tracing::warn.*test environment" test-harness/bdd/src/steps/
```

**Wire these after core features are working.**

---

## Workflow

### Daily Loop

1. **Morning:** Pick one binary (e.g., llm-worker-rbee)
2. **Implement:** Add 1-2 features from Priority 1
3. **Test:** Run relevant BDD tests
4. **Fix:** Implement missing pieces until tests pass
5. **Commit:** Add TEAM-084 signatures and commit
6. **Repeat:** Move to next feature

### Example Day 1

**Goal:** Get inference execution working

```bash
# 1. Implement SlotManager in llm-worker-rbee
vim bin/llm-worker-rbee/src/inference.rs

# 2. Run BDD test
LLORCH_BDD_FEATURE_PATH=tests/features/130-inference-execution.feature \
  cargo test --package test-harness-bdd -- --nocapture

# 3. See what fails, implement missing pieces
# 4. Repeat until test passes
```

---

## Success Criteria

### Minimum Acceptable (TEAM-084)
- [ ] 5+ core features implemented in binaries
- [ ] 3+ BDD test scenarios passing
- [ ] Compilation passes for all binaries
- [ ] Progress documented

### Target Goal (TEAM-084)
- [ ] 15+ features implemented across all binaries
- [ ] 10+ BDD test scenarios passing
- [ ] Integration tests (900-integration-e2e.feature) passing
- [ ] Documentation updated

### Stretch Goal (TEAM-084)
- [ ] All core features implemented
- [ ] 50+ BDD test scenarios passing
- [ ] All integration tests passing
- [ ] CI/CD pipeline green

---

## Anti-Patterns to Avoid

❌ **DON'T:**
- Implement features without running BDD tests
- Change BDD test expectations to match incomplete features
- Skip error handling in product code
- Leave TODO markers in production code
- Implement features that aren't tested

✅ **DO:**
- Run BDD tests frequently (after each feature)
- Implement features that BDD tests expect
- Add proper error handling
- Add TEAM-084 signatures to all changes
- Document what you implemented

---

## Key Files

### BDD Tests (Your Specification)
- `test-harness/bdd/tests/features/*.feature` - What to implement
- `test-harness/bdd/src/steps/*.rs` - How tests call your APIs

### Product Code (What You're Building)
- `bin/llm-worker-rbee/src/` - Worker implementation
- `bin/queen-rbee/src/` - Registry implementation
- `bin/rbee-hive/src/` - Orchestrator implementation
- `bin/rbee-keeper/src/` - CLI implementation

### Shared Libraries
- `bin/shared-crates/` - Shared types and utilities

---

## Verification Commands

```bash
# Check compilation for all binaries
cargo check --workspace

# Run specific BDD test
LLORCH_BDD_FEATURE_PATH=tests/features/900-integration-e2e.feature \
  cargo test --package test-harness-bdd -- --nocapture

# Run all BDD tests (will take time)
cargo test --package test-harness-bdd -- --nocapture

# Count TEAM-084 signatures
rg "TEAM-084:" bin/ test-harness/bdd/src/steps/
```

---

## Questions?

**If stuck:**
1. Read the BDD test scenario - it tells you what to implement
2. Look at the step definition - it shows how to call your API
3. Check existing code in `/bin/` for patterns
4. Implement the simplest thing that makes the test pass
5. Refactor after tests pass

**Key insight:** The BDD tests are your specification. Implement exactly what they expect, nothing more, nothing less.

---

## Bottom Line

**TEAM-084's mission: Build the product features that the BDD tests expect.**

The test suite is ready. The specifications are clear. Now it's time to **implement the actual product code**.

**Workflow:**
1. Pick a feature from Priority 1
2. Implement it in the appropriate binary
3. Run the BDD test
4. Fix failures
5. Repeat

**The BDD tests will guide you every step of the way.**

---

**Created by:** TEAM-083  
**Date:** 2025-10-11  
**Next Team:** TEAM-084  
**Estimated Work:** 26+ hours (3-4 days)  
**Priority:** P0 - CRITICAL - Product features needed for BDD validation

---

## CRITICAL NOTE

**This is a shift from "wiring tests" to "implementing product features".**

Previous teams (TEAM-076 through TEAM-083) focused on **writing and wiring BDD tests**.

**TEAM-084 must focus on IMPLEMENTING PRODUCT CODE in `/bin/` binaries.**

The tests are done. Now build the product.
