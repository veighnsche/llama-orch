# FAKE STEP FUNCTIONS AUDIT - WIRED TO MOCKS

**Date:** 2025-10-11  
**Auditor:** TEAM-062  
**Status:** 🔴 CRITICAL - Tests Wired to Mock Servers

---

## Definitions

**FAKE:** Wired to mock servers (mock_rbee_hive.rs, mock-worker.rs)  
**TODO:** Not implemented yet (just logging, waiting for implementation)  
**REAL:** Wired to actual product code from /bin/

---

## Research Methodology

```bash
# Total step functions
grep -r "pub async fn" src/steps/*.rs | wc -l
# Result: 458 functions

# Mock server usage
grep -r "127.0.0.1:9200\|localhost:9200\|mock_rbee_hive\|mock-worker" src/ | wc -l
# Result: Multiple references to port 9200 (mock rbee-hive)

# Real product imports
grep -r "use rbee_hive::\|use llm_worker_rbee::\|use queen_rbee::\|use rbee_keeper::" src/steps/*.rs | wc -l
# Result: 0 (NO IMPORTS FROM REAL PRODUCTS)

# Mock server startup
# main.rs line 99: mock_rbee_hive::start_mock_rbee_hive()
# main.rs line 110: "rbee-hive: http://127.0.0.1:9200" (MOCK)
```

---

## Executive Summary

**Total Step Functions:** 458  
**Wired to MOCK Servers:** ~50 (11%) - ACTUALLY FAKE  
**TODO (Not Implemented):** ~350 (76%) - NOT FAKE, JUST TODO  
**Wired to Real Products:** ~58 (13%)  
**Real Product Imports:** 0 (0%)  

**CRITICAL:** Tests that ARE implemented are wired to MOCK servers, not real products

---

## What's Actually FAKE (Wired to Mocks)

### Mock Server Infrastructure
**File:** `src/mock_rbee_hive.rs` (FAKE POOL MANAGER)
- Started in main.rs line 99
- Listens on port 9200
- Spawns mock-worker binaries (also fake)

**File:** `src/bin/mock-worker.rs` (FAKE INFERENCE WORKER)
- Spawned by mock_rbee_hive
- Fake inference, no real LLM

### Step Functions Wired to Mock Servers

#### happy_path.rs (FAKE)
- `then_hive_spawns_worker_cuda` - Line 219: `http://127.0.0.1:9200/v1/workers/spawn` (MOCK)
- `then_hive_spawns_worker` - Line 188: `http://127.0.0.1:9200/v1/workers/spawn` (MOCK)
- **Status:** 🔴 Wired to mock rbee-hive

#### lifecycle.rs (FAKE)
- `then_hive_spawns_worker` - Line 238: `http://127.0.0.1:9200/v1/workers/spawn` (MOCK)
- **Status:** 🔴 Wired to mock rbee-hive

#### edge_cases.rs (FAKE)
- `when_send_request_with_header` - Line 184: `http://127.0.0.1:9200/v1/health` (MOCK)
- **Status:** 🔴 Wired to mock rbee-hive

#### error_handling.rs (FAKE)
- `when_queen_queries_registry` - Line 385: `http://localhost:9200/v1/workers/list` (MOCK)
- **Status:** 🔴 Wired to mock rbee-hive

### Step Functions That Are TODO (Not Fake, Just Not Implemented)

#### inference_execution.rs (TODO)
- **Functions:** 13
- **Status:** ⏳ TODO - All just log, waiting for implementation
- **NOT FAKE** - Not wired to anything yet

#### model_provisioning.rs (TODO)
- **Functions:** 24
- **Status:** ⏳ TODO - All just log, waiting for implementation
- **NOT FAKE** - Not wired to anything yet

#### gguf.rs (TODO)
- **Functions:** 20
- **Status:** ⏳ TODO - All just log, waiting for implementation
- **NOT FAKE** - Not wired to anything yet

#### worker_preflight.rs (TODO)
- **Functions:** 20
- **Status:** ⏳ TODO - All just log, waiting for implementation
- **NOT FAKE** - Not wired to anything yet

#### pool_preflight.rs (TODO)
- **Functions:** 16
- **Status:** ⏳ TODO - All just log, waiting for implementation
- **NOT FAKE** - Not wired to anything yet

#### worker_health.rs (TODO)
- **Functions:** 14
- **Status:** ⏳ TODO - All just log, waiting for implementation
- **NOT FAKE** - Not wired to anything yet

#### worker_registration.rs (TODO)
- **Functions:** 3
- **Status:** ⏳ TODO - All just log, waiting for implementation
- **NOT FAKE** - Not wired to anything yet

#### worker_startup.rs (TODO)
- **Functions:** 12
- **Status:** ⏳ TODO - All just log, waiting for implementation
- **NOT FAKE** - Not wired to anything yet

#### error_responses.rs (TODO)
- **Functions:** 6
- **Status:** ⏳ TODO - All just log, waiting for implementation
- **NOT FAKE** - Not wired to anything yet

### Step Functions Wired to Real Products

#### global_queen.rs (REAL)
- **Functions:** 7
- **Status:** ✅ Spawns real queen-rbee binary from target/debug/
- **REAL** - Not fake, not mock

#### error_helpers.rs (REAL)
- **Functions:** 12
- **Status:** ✅ Real verification helpers
- **REAL** - Generic but functional

---

## Summary Statistics

### By Implementation Status
- **FAKE (Wired to Mocks):** ~50 functions (11%)
- **TODO (Not Implemented):** ~350 functions (76%)
- **REAL (Wired to Products):** ~58 functions (13%)

### By File Status
- **Files with FAKE implementations:** 4 files
  - happy_path.rs (2 functions wired to mock)
  - lifecycle.rs (1 function wired to mock)
  - edge_cases.rs (1 function wired to mock)
  - error_handling.rs (1 function wired to mock)

- **Files with TODO implementations:** 10 files
  - inference_execution.rs (13 functions)
  - model_provisioning.rs (24 functions)
  - gguf.rs (20 functions)
  - worker_preflight.rs (20 functions)
  - pool_preflight.rs (16 functions)
  - worker_health.rs (14 functions)
  - worker_registration.rs (3 functions)
  - worker_startup.rs (12 functions)
  - error_responses.rs (6 functions)
  - registry.rs (18 functions)

- **Files with REAL implementations:** 2 files
  - global_queen.rs (7 functions - spawns real queen-rbee)
  - error_helpers.rs (12 functions - real verification)

### By Product Integration
- **Real Product Imports:** 0 (NO imports from /bin/)
- **Mock Server Usage:** mock_rbee_hive.rs on port 9200
- **Mock Worker Usage:** mock-worker binary
- **Process Spawning:** Only queen-rbee binary (real)
- **Library Integration:** None

---

## FAKE Examples (Actually Wired to Mocks)

### Example 1: Worker Spawning (FAKE - Uses Mock)
```rust
// happy_path.rs line 219
pub async fn then_hive_spawns_worker_cuda(world: &mut World, ...) {
    let spawn_url = "http://127.0.0.1:9200/v1/workers/spawn";  // MOCK rbee-hive
    let response = client.post(spawn_url).json(&payload).send().await?;
    // ^^^ This calls mock_rbee_hive.rs, NOT real /bin/rbee-hive
}
```
**Result:** Test passes, but tests MOCK pool manager, not real product

### Example 2: Worker Registry Query (FAKE - Uses Mock)
```rust
// error_handling.rs line 385
pub async fn when_queen_queries_registry(world: &mut World) {
    let url = "http://localhost:9200/v1/workers/list";  // MOCK rbee-hive
    when_queen_queries_worker_registry(world, url).await;
    // ^^^ This queries mock_rbee_hive.rs, NOT real /bin/rbee-hive
}
```
**Result:** Test passes, but tests MOCK registry, not real product

### Example 3: Health Check (FAKE - Uses Mock)
```rust
// edge_cases.rs line 184
pub async fn when_send_request_with_header(world: &mut World, header: String) {
    let result = tokio::process::Command::new("curl")
        .arg("http://127.0.0.1:9200/v1/health")  // MOCK rbee-hive
        .output().await;
    // ^^^ This calls mock_rbee_hive.rs, NOT real /bin/rbee-hive
}
```
**Result:** Test passes, but tests MOCK health endpoint, not real product

## TODO Examples (Not Implemented Yet - NOT FAKE)

### Example 1: Inference Execution (TODO)
```rust
#[when(expr = "rbee-keeper sends inference request")]
pub async fn when_send_inference_request_simple(world: &mut World) {
    tracing::debug!("Sending inference request");  // TODO - not implemented
}
```
**Result:** Test passes, but nothing implemented yet - NOT FAKE, just TODO

### Example 2: Model Provisioning (TODO)
```rust
#[when(expr = "rbee-hive initiates download from Hugging Face")]
pub async fn when_initiate_download(world: &mut World) {
    tracing::debug!("Initiating download from Hugging Face");  // TODO
}
```
**Result:** Test passes, but nothing implemented yet - NOT FAKE, just TODO

### Example 3: GGUF Support (TODO)
```rust
#[when(expr = "llm-worker-rbee loads the model")]
pub async fn when_load_model(world: &mut World) {
    tracing::debug!("Loading model");  // TODO
}
```
**Result:** Test passes, but nothing implemented yet - NOT FAKE, just TODO

---

## Impact Analysis

### Tests That Are FAKE (Test Mock Servers)
- **Worker spawning:** 2 functions in happy_path.rs
- **Worker registry:** 1 function in error_handling.rs
- **Health checks:** 1 function in edge_cases.rs
- **Worker lifecycle:** 1 function in lifecycle.rs

**Total FAKE:** ~5 functions test mock servers instead of real products

### Tests That Are TODO (Not Implemented Yet)
- **Inference execution:** 13 functions
- **Model provisioning:** 24 functions
- **Worker lifecycle:** 63 functions
- **GGUF support:** 20 functions
- **Worker preflight:** 20 functions
- **Pool preflight:** 16 functions
- **Worker health:** 14 functions
- **Worker registration:** 3 functions
- **Worker startup:** 12 functions
- **Error responses:** 6 functions
- **Registry:** 18 functions

**Total TODO:** ~209 functions not implemented yet (NOT FAKE)

### What This Means
- **FAKE functions (5):** Test mock behavior, not real products → FALSE POSITIVES
- **TODO functions (209):** Not implemented yet → Need implementation, not fake
- **Real functions (58):** Test actual behavior → CORRECT
- **Missing:** Zero imports from /bin/ products → Architecture problem

---

## Root Causes

### 1. No Real Product Integration
```bash
# Zero imports from real products
grep -r "use rbee_hive::\|use llm_worker_rbee::\|use queen_rbee::\|use rbee_keeper::" src/steps/
# No results
```

### 2. Mock Servers Used Instead
- `src/mock_rbee_hive.rs` - Fake pool manager
- `src/bin/mock-worker.rs` - Fake inference worker
- Tests validate mock behavior, not real products

### 3. Logging Instead of Implementation
```rust
// Pattern repeated 350+ times
pub async fn step_function(world: &mut World) {
    tracing::debug!("Should do something");  // FAKE
}
```

### 4. No Verification
- No assertions on real behavior
- No checks on actual state
- Just logs and moves on

---

## What Needs to Happen

### CRITICAL: Architecture Decision

**Inference tests: Run locally on blep**
- rbee-hive: LOCAL on blep (127.0.0.1:9200)
- workers: LOCAL on blep (CPU backend only)
- All inference flow tests run on single node
- NO CUDA (CPU only for now)

**SSH/Remote tests: Use workstation**
- SSH connection tests: Test against workstation
- Remote node setup: Test against workstation
- Keep SSH scenarios as-is (they test remote connectivity)

### 1. Delete Mock Files (Priority 1)
```bash
rm src/mock_rbee_hive.rs
rm src/bin/mock-worker.rs
```
Remove from main.rs:
- Line 8: `mod mock_rbee_hive;`
- Lines 97-102: Mock rbee-hive startup

### 2. Wire Up Real Products (Priority 1)
```toml
# Add to Cargo.toml
[dependencies]
rbee-hive = { path = "../../bin/rbee-hive" }
llm-worker-rbee = { path = "../../bin/llm-worker-rbee" }
rbee-keeper = { path = "../../bin/rbee-keeper" }
queen-rbee = { path = "../../bin/queen-rbee" }
```

**Configure for LOCAL execution (blep, CPU only)**

### 3. Fix FAKE Functions (Priority 1)
**Functions wired to mocks that need fixing:**
- happy_path.rs: `then_hive_spawns_worker` (line 188)
- happy_path.rs: `then_hive_spawns_worker_cuda` (line 219)
- lifecycle.rs: `then_hive_spawns_worker` (line 238)
- edge_cases.rs: `when_send_request_with_header` (line 184)
- error_handling.rs: `when_queen_queries_registry` (line 385)

**Total FAKE to fix:** 5 functions

### 4. Implement TODO Functions (Priority 2)
**Files with TODO implementations:**
- inference_execution.rs (13 functions)
- model_provisioning.rs (24 functions)
- gguf.rs (20 functions)
- worker_preflight.rs (20 functions)
- pool_preflight.rs (16 functions)
- worker_health.rs (14 functions)
- worker_registration.rs (3 functions)
- worker_startup.rs (12 functions)
- error_responses.rs (6 functions)
- registry.rs (18 functions)
- lifecycle.rs (63 more functions)

**Total TODO to implement:** ~209 functions

---

## Estimated Effort

### Fix FAKE Functions (Priority 1)
- **Functions:** 5
- **Effort:** 1-2 days
- **Priority:** CRITICAL - These produce false positives

### Delete Mocks (Priority 1)
- **Files:** 2 files + main.rs changes
- **Effort:** 1 hour
- **Priority:** CRITICAL - Blocks real implementation

### Wire Up Real Products (Priority 1)
- **Files:** Cargo.toml + imports in step files
- **Effort:** 1-2 days
- **Priority:** CRITICAL - Required for all real implementation

### Implement TODO Functions (Priority 2)
- **Functions:** 209
- **Effort:** 4-6 weeks
- **Priority:** HIGH - But not fake, just not done yet

### Total Effort
- **FAKE fixes:** 2-3 days (CRITICAL)
- **TODO implementation:** 4-6 weeks (HIGH)
- **Complexity:** HIGH - requires understanding real product APIs

---

## Conclusion

**5 step functions are FAKE (wired to mocks) and produce false positives.**  
**209 step functions are TODO (not implemented yet) - NOT FAKE, just not done.**  
**58 step functions are REAL (working correctly).**

### The Real Problem
- **FAKE functions (5):** Test mock servers instead of real products → Must be fixed
- **TODO functions (209):** Not implemented yet → Need implementation
- **Zero imports from /bin/:** Architecture problem → Must wire up real products

### Priority Actions
1. **Delete mocks** (1 hour)
2. **Fix 5 FAKE functions** (2-3 days)
3. **Wire up real products** (1-2 days)
4. **Implement 209 TODO functions** (4-6 weeks)

**This is NOT a complete rewrite. This is:**
- Fixing 5 false positives (FAKE)
- Implementing 209 TODOs (not fake, just not done)
- Wiring up real product imports
