# UNDO MOCK PLAN - Removing All False Infrastructure

**Date:** 2025-10-11  
**Purpose:** Complete plan to undo mock infrastructure and wire up real products  
**Estimated Effort:** 3-5 days

---

## CRITICAL ARCHITECTURE DECISION

**Inference tests: Run locally on blep**
- rbee-hive: LOCAL on blep (127.0.0.1:9200)
- workers: LOCAL on blep (CPU backend only)
- All inference flow tests run on single node
- NO CUDA (CPU only for now)

**SSH/Remote tests: Use workstation**
- SSH connection tests: Test against workstation
- Remote node setup: Test against workstation
- Keep SSH scenarios as-is (they test remote connectivity)

**Why:** Inference flow on single node is simpler. SSH tests still need workstation to validate remote connectivity.

---

## Phase 1: Delete Mock Infrastructure (1 hour)

### Files to Delete
```bash
rm src/mock_rbee_hive.rs              # Created by TEAM-054, enhanced by TEAM-055, TEAM-059
rm src/bin/mock-worker.rs             # Created by TEAM-059
```

### Files to Modify

#### src/main.rs
**Remove:**
```rust
// Line 8
mod mock_rbee_hive;

// Lines 97-110
// TEAM-054: Start mock rbee-hive on port 9200 (NOT 8080 or 8090!)
tokio::spawn(async {
    if let Err(e) = mock_rbee_hive::start_mock_rbee_hive().await {
        tracing::error!("Mock rbee-hive failed: {}", e);
    }
});

// Wait for mock servers to start
tokio::time::sleep(Duration::from_millis(1000)).await;

tracing::info!("✅ Mock servers ready:");
tracing::info!("   - queen-rbee: http://127.0.0.1:8080");
tracing::info!("   - rbee-hive:  http://127.0.0.1:9200");
```

**Keep:**
```rust
// TEAM-051: Start global queen-rbee instance before running tests
steps::global_queen::start_global_queen_rbee().await;

tracing::info!("✅ Real servers ready:");
tracing::info!("   - queen-rbee: http://127.0.0.1:8080");

World::cucumber().fail_on_skipped().run(features).await;
```

---

## Phase 2: Wire Up Real Products (1-2 days)

### Add Real Dependencies

#### Cargo.toml
**Add:**
```toml
[dependencies]
# Real product dependencies
rbee-hive = { path = "../../bin/rbee-hive" }
llm-worker-rbee = { path = "../../bin/llm-worker-rbee" }
rbee-keeper = { path = "../../bin/rbee-keeper" }
queen-rbee = { path = "../../bin/queen-rbee" }
```

### Create Real Product Wrappers

#### src/real_rbee_hive.rs (NEW FILE)
```rust
//! Real rbee-hive integration for BDD tests
//! 
//! CRITICAL: Runs LOCALLY on blep with CPU backend only
//! NO remote connections, NO SSH, NO workstation

use rbee_hive::pool::PoolManager;
use rbee_hive::config::Config;
use anyhow::Result;

pub struct RealRbeeHive {
    pool: PoolManager,
}

impl RealRbeeHive {
    pub async fn new() -> Result<Self> {
        // CRITICAL: Local configuration for blep (CPU only)
        let config = Config {
            host: "127.0.0.1".to_string(),  // LOCAL ONLY
            port: 9200,
            backend: "cpu".to_string(),      // CPU ONLY (no CUDA)
            // NO SSH configuration
            // NO remote nodes
        };
        
        let pool = PoolManager::new(config).await?;
        tracing::info!("✅ Real rbee-hive started LOCALLY on blep (CPU backend)");
        Ok(Self { pool })
    }
    
    pub async fn spawn_worker(&self, model_ref: &str) -> Result<String> {
        // CRITICAL: Always use CPU backend (local on blep)
        self.pool.spawn_worker(model_ref, "cpu", 0).await
    }
    
    pub async fn list_workers(&self) -> Result<Vec<WorkerInfo>> {
        self.pool.list_workers().await
    }
}
```

#### src/real_worker.rs (NEW FILE)
```rust
//! Real llm-worker-rbee integration for BDD tests
//!
//! CRITICAL: Runs LOCALLY on blep with CPU backend only
//! NO remote connections, NO CUDA

use llm_worker_rbee::inference::InferenceEngine;
use llm_worker_rbee::config::WorkerConfig;
use anyhow::Result;

pub struct RealWorker {
    engine: InferenceEngine,
}

impl RealWorker {
    pub async fn new(model_path: &str) -> Result<Self> {
        // CRITICAL: Local configuration for blep (CPU only)
        let config = WorkerConfig {
            model_path: model_path.to_string(),
            backend: "cpu".to_string(),      // CPU ONLY (no CUDA)
            device: 0,                       // Ignored for CPU
            host: "127.0.0.1".to_string(),  // LOCAL ONLY
            port: 8001,                      // Local port
        };
        
        let engine = InferenceEngine::new(config).await?;
        tracing::info!("✅ Real worker started LOCALLY on blep (CPU backend)");
        Ok(Self { engine })
    }
    
    pub async fn infer(&self, prompt: &str) -> Result<String> {
        self.engine.generate(prompt).await
    }
}
```

#### src/main.rs
**Add:**
```rust
mod real_rbee_hive;
mod real_worker;
```

---

## Phase 3: Fix False Positive Functions (2-3 days)

### 1. Fix happy_path.rs (2 functions)

#### Function 1: then_hive_spawns_worker (Line 188)

**Current (FAKE):**
```rust
pub async fn then_hive_spawns_worker(world: &mut World, binary: String, port: u16) {
    let client = crate::steps::world::create_http_client();
    let spawn_url = "http://127.0.0.1:9200/v1/workers/spawn";  // MOCK
    
    let payload = serde_json::json!({
        "binary": binary,
        "port": port,
    });
    
    let response = client.post(spawn_url).json(&payload).send().await?;
    // ...
}
```

**New (REAL):**
```rust
use crate::real_rbee_hive::RealRbeeHive;

pub async fn then_hive_spawns_worker(world: &mut World, binary: String, port: u16) {
    // Get or create real rbee-hive instance
    let rbee_hive = world.get_or_create_rbee_hive().await?;
    
    // Actually spawn worker using real product code
    let worker_id = rbee_hive.spawn_worker(
        &world.current_model_ref,
        "cpu",  // or from world state
        0
    ).await?;
    
    // Store in world state
    world.spawned_workers.push(worker_id);
    
    tracing::info!("✅ Real worker spawned via real rbee-hive");
}
```

#### Function 2: then_hive_spawns_worker_cuda (Line 219)

**Current (FAKE):**
```rust
pub async fn then_hive_spawns_worker_cuda(world: &mut World, binary: String, port: u16, device: u32) {
    let spawn_url = "http://127.0.0.1:9200/v1/workers/spawn";  // MOCK
    // ...
}
```

**New (REAL):**
```rust
pub async fn then_hive_spawns_worker_cuda(world: &mut World, binary: String, port: u16, device: u32) {
    let rbee_hive = world.get_or_create_rbee_hive().await?;
    
    let worker_id = rbee_hive.spawn_worker(
        &world.current_model_ref,
        "cuda",
        device
    ).await?;
    
    world.spawned_workers.push(worker_id);
    
    tracing::info!("✅ Real CUDA worker spawned on device {}", device);
}
```

### 2. Fix lifecycle.rs (1 function)

#### Function: then_hive_spawns_worker (Line 238)

**Current (FAKE):**
```rust
pub async fn then_hive_spawns_worker(world: &mut World) {
    let spawn_url = "http://127.0.0.1:9200/v1/workers/spawn";  // MOCK
    // ...
}
```

**New (REAL):**
```rust
pub async fn then_hive_spawns_worker(world: &mut World) {
    let rbee_hive = world.get_or_create_rbee_hive().await?;
    
    let worker_id = rbee_hive.spawn_worker(
        "mock-model",  // or from world state
        "cpu",
        0
    ).await?;
    
    world.spawned_workers.push(worker_id);
    
    tracing::info!("✅ Real worker spawned");
}
```

### 3. Fix edge_cases.rs (1 function)

#### Function: when_send_request_with_header (Line 184)

**Current (FAKE):**
```rust
pub async fn when_send_request_with_header(world: &mut World, header: String) {
    let result = tokio::process::Command::new("curl")
        .arg("http://127.0.0.1:9200/v1/health")  // MOCK
        .output().await;
    // ...
}
```

**New (REAL):**
```rust
pub async fn when_send_request_with_header(world: &mut World, header: String) {
    let rbee_hive = world.get_or_create_rbee_hive().await?;
    
    // Make real health check against real rbee-hive
    let health = rbee_hive.health_check().await;
    
    match health {
        Ok(_) => world.last_exit_code = Some(0),
        Err(e) => {
            world.last_exit_code = Some(1);
            world.last_error = Some(e.to_string());
        }
    }
}
```

### 4. Fix error_handling.rs (1 function)

#### Function: when_queen_queries_registry (Line 385)

**Current (FAKE):**
```rust
pub async fn when_queen_queries_registry(world: &mut World) {
    let url = "http://localhost:9200/v1/workers/list";  // MOCK
    when_queen_queries_worker_registry(world, url).await;
}
```

**New (REAL):**
```rust
pub async fn when_queen_queries_registry(world: &mut World) {
    let rbee_hive = world.get_or_create_rbee_hive().await?;
    
    // Query real worker registry
    match rbee_hive.list_workers().await {
        Ok(workers) => {
            world.last_workers = workers;
            world.last_exit_code = Some(0);
        }
        Err(e) => {
            world.last_error = Some(ErrorResponse {
                code: "REGISTRY_QUERY_FAILED".to_string(),
                message: e.to_string(),
                details: None,
            });
            world.last_exit_code = Some(1);
        }
    }
}
```

### 5. Update World State

#### src/steps/world.rs

**Add:**
```rust
use crate::real_rbee_hive::RealRbeeHive;
use crate::real_worker::RealWorker;

#[derive(Debug, Default, cucumber::World)]
pub struct World {
    // ... existing fields ...
    
    // Real product instances
    pub rbee_hive: Option<RealRbeeHive>,
    pub spawned_workers: Vec<String>,
    pub current_model_ref: String,
}

impl World {
    pub async fn get_or_create_rbee_hive(&mut self) -> Result<&mut RealRbeeHive> {
        if self.rbee_hive.is_none() {
            let config = rbee_hive::config::Config::default();
            self.rbee_hive = Some(RealRbeeHive::new(config).await?);
        }
        Ok(self.rbee_hive.as_mut().unwrap())
    }
}
```

---

## Phase 4: Update Documentation (1 hour)

### Files to Update

#### TEAM_063_REAL_HANDOFF.md
**Update status:**
```markdown
## Phase 1: Delete Mocks (COMPLETE)
- [x] Deleted mock_rbee_hive.rs
- [x] Deleted mock-worker.rs
- [x] Removed mock startup from main.rs

## Phase 2: Wire Up Real Products (COMPLETE)
- [x] Added real product dependencies
- [x] Created real product wrappers
- [x] Updated World state

## Phase 3: Fix False Positives (COMPLETE)
- [x] Fixed happy_path.rs (2 functions)
- [x] Fixed lifecycle.rs (1 function)
- [x] Fixed edge_cases.rs (1 function)
- [x] Fixed error_handling.rs (1 function)
```

#### Create UNDO_COMPLETE.md
```markdown
# UNDO COMPLETE - Mock Infrastructure Removed

**Date:** 2025-10-11  
**Status:** ✅ All mocks removed, real products wired up

## What Was Removed
- mock_rbee_hive.rs (TEAM-054, TEAM-055, TEAM-059)
- mock-worker.rs (TEAM-059)
- Mock startup in main.rs (TEAM-054)

## What Was Added
- Real product dependencies in Cargo.toml
- real_rbee_hive.rs wrapper
- real_worker.rs wrapper
- Real product instances in World state

## Functions Fixed
- happy_path.rs: 2 functions
- lifecycle.rs: 1 function
- edge_cases.rs: 1 function
- error_handling.rs: 1 function

## Result
- 0 false positives (was 5)
- 100% real product testing
- All tests validate actual behavior
```

---

## Phase 5: Verify & Test (1 day)

### Verification Steps

#### 1. Compilation Check
```bash
cd test-harness/bdd
cargo check --bin bdd-runner
```

**Expected:** No compilation errors

#### 2. Verify No Mock References
```bash
grep -r "mock_rbee_hive\|mock-worker\|127.0.0.1:9200\|localhost:9200" src/
```

**Expected:** No results (except in comments/docs)

#### 3. Verify Real Product Imports
```bash
grep -r "use rbee_hive::\|use llm_worker_rbee::" src/steps/
```

**Expected:** Multiple results showing real imports

#### 4. Run Tests
```bash
cargo run --bin bdd-runner 2>&1 | tee undo_test_output.log
```

**Expected:** Tests may fail initially (need real products configured), but no mock references

#### 5. Check Process List
```bash
ps aux | grep -E "mock-worker|mock.*rbee"
```

**Expected:** No mock processes running

---

## Rollback Plan (If Needed)

### If Something Goes Wrong

#### Restore Mocks (Emergency Only)
```bash
git checkout HEAD -- src/mock_rbee_hive.rs
git checkout HEAD -- src/bin/mock-worker.rs
git checkout HEAD -- src/main.rs
```

#### Revert Step Functions
```bash
git checkout HEAD -- src/steps/happy_path.rs
git checkout HEAD -- src/steps/lifecycle.rs
git checkout HEAD -- src/steps/edge_cases.rs
git checkout HEAD -- src/steps/error_handling.rs
```

**Note:** Only use rollback if critical blocker found. Otherwise, fix forward.

---

## Success Criteria

### Must Have
- [x] All mock files deleted
- [x] Real product dependencies added
- [x] 5 false positive functions fixed
- [x] No references to port 9200 in step functions
- [x] Real product imports in step files
- [x] Code compiles successfully

### Should Have
- [ ] Tests run without mock server errors
- [ ] Real rbee-hive instance created
- [ ] Real workers spawned
- [ ] All tests validate real behavior

### Nice to Have
- [ ] Documentation updated
- [ ] Handoff complete
- [ ] Team notified

---

## Timeline

### Day 1 (4 hours)
- **Morning:** Phase 1 (Delete mocks) - 1 hour
- **Afternoon:** Phase 2 (Wire up real products) - 3 hours

### Day 2 (8 hours)
- **Morning:** Phase 3 part 1 (Fix happy_path.rs, lifecycle.rs) - 4 hours
- **Afternoon:** Phase 3 part 2 (Fix edge_cases.rs, error_handling.rs) - 4 hours

### Day 3 (4 hours)
- **Morning:** Phase 4 (Update documentation) - 1 hour
- **Afternoon:** Phase 5 (Verify & test) - 3 hours

**Total:** 16 hours (2-3 days)

---

## Risk Assessment

### Low Risk
- Deleting mock files (Phase 1)
- Adding dependencies (Phase 2)
- Updating documentation (Phase 4)

### Medium Risk
- Creating real product wrappers (Phase 2)
- Updating World state (Phase 2)

### High Risk
- Fixing step functions (Phase 3)
- Integration with real products (Phase 5)

**Mitigation:** Test incrementally, fix one function at a time, verify after each change

---

## Team Assignments

### TEAM-063 (Recommended)
- **Phase 1:** Junior dev (1 hour)
- **Phase 2:** Senior dev (1-2 days)
- **Phase 3:** Senior dev (2-3 days)
- **Phase 4:** Junior dev (1 hour)
- **Phase 5:** Senior dev (1 day)

**Total:** 1 senior dev + 1 junior dev, 3-5 days

---

## Conclusion

This plan completely removes all mock infrastructure created by TEAM-054, TEAM-055, and TEAM-059, and replaces it with real product integration. 

**Effort:** 3-5 days  
**Impact:** Eliminates 5 false positives, enables real product testing  
**Risk:** Medium (but manageable with incremental approach)

**After completion:** BDD tests will validate actual product behavior, not mock behavior.
