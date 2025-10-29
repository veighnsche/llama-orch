# Mode 3 (Integrated) - Investigation & Migration Guide

**Created by:** TEAM-265  
**Date:** Oct 23, 2025  
**Status:** Investigation Guide  
**For:** Next team implementing integrated mode

---

## Mission

Implement Mode 3 (Integrated) where queen-rbee calls rbee-hive crates directly instead of via HTTP, achieving 50-100x performance improvement for localhost operations.

---

## Phase 1: Investigation (Required Before Implementation)

### 1.1 Understand Current Architecture

**Read these files first:**
1. `bin/.plan/QUEEN_TO_HIVE_COMMUNICATION_MODES.md` - Overview of 3 modes
2. `bin/10_queen_rbee/src/hive_forwarder.rs` - Current HTTP forwarding
3. `bin/10_queen_rbee/src/job_router.rs` - Operation routing
4. `bin/20_rbee_hive/src/job_router.rs` - Hive's operation handlers

**Key questions to answer:**
- [ ] What operations does hive_forwarder currently forward?
- [ ] How does rbee-hive's job_router handle these operations?
- [ ] What's the HTTP request/response format?
- [ ] What's the SSE streaming format?
- [ ] How are errors propagated?

**Investigation commands:**
```bash
# Find all operations that get forwarded
grep -r "should_forward_to_hive" bin/99_shared_crates/rbee-operations/src/

# See what hive's job_router does
cat bin/20_rbee_hive/src/job_router.rs

# Check operation types
cat bin/99_shared_crates/rbee-operations/src/lib.rs
```

---

### 1.2 Map HTTP Flow to Direct Calls

**Current HTTP flow (Mode 2):**
```text
queen-rbee
  ↓ hive_forwarder::forward_to_hive()
  ↓ JobClient::submit_and_stream()
  ↓ POST http://localhost:8600/v1/jobs
  ↓ rbee-hive receives HTTP
  ↓ rbee-hive::job_router::route_operation()
  ↓ Execute operation (e.g., worker_lifecycle::spawn_worker)
  ↓ Stream results via SSE
  ↓ queen-rbee receives SSE events
  ↓ Forward to client
```

**Target integrated flow (Mode 3):**
```text
queen-rbee
  ↓ hive_forwarder::forward_to_hive()
  ↓ execute_integrated() [NEW]
  ↓ Direct call to rbee-hive crates
  ↓ worker_lifecycle::spawn_worker() [DIRECT]
  ↓ Collect results in memory
  ↓ Convert to narration events
  ↓ Forward to client (no HTTP/SSE)
```

**Investigation tasks:**
- [ ] List all operations that get forwarded (WorkerSpawn, WorkerList, etc.)
- [ ] For each operation, find the corresponding rbee-hive crate function
- [ ] Document the function signature and return type
- [ ] Document how results are currently streamed via SSE
- [ ] Identify any state that's shared between operations

**Create a mapping table:**
```markdown
| Operation | HTTP Endpoint | Hive Crate | Function | Return Type |
|-----------|---------------|------------|----------|-------------|
| WorkerSpawn | POST /v1/jobs | worker-lifecycle | spawn_worker() | Result<WorkerId> |
| WorkerList | POST /v1/jobs | worker-lifecycle | list_workers() | Result<Vec<Worker>> |
| ... | ... | ... | ... | ... |
```

---

### 1.3 Identify Dependencies

**Check what rbee-hive crates are needed:**
```bash
# List rbee-hive crates
ls -la bin/25_rbee_hive_crates/

# Check their Cargo.toml files
cat bin/25_rbee_hive_crates/*/Cargo.toml
```

**Expected crates:**
- `rbee-hive-worker-lifecycle` - Worker operations
- `rbee-hive-model-catalog` - Model operations
- `rbee-hive-model-provisioner` - Model download
- Others?

**Investigation tasks:**
- [ ] List all rbee-hive crates
- [ ] Check which ones are needed for forwarded operations
- [ ] Document their public APIs
- [ ] Check for circular dependencies (queen → hive → queen?)
- [ ] Identify any shared state (registries, catalogs, etc.)

---

### 1.4 Analyze State Management

**Critical question:** How does rbee-hive manage state?

**Current state in rbee-hive:**
- Worker registry (in-memory)
- Model catalog (SQLite?)
- Download tracker
- Device detection cache
- Others?

**Investigation tasks:**
- [ ] Find all stateful components in rbee-hive
- [ ] Check if they're thread-safe (Arc<Mutex<...>>?)
- [ ] Determine if state can be shared with queen
- [ ] Check for singleton patterns or global state
- [ ] Identify initialization requirements

**Commands:**
```bash
# Find registries and state
grep -r "Arc<Mutex" bin/20_rbee_hive/src/
grep -r "lazy_static" bin/20_rbee_hive/src/
grep -r "once_cell" bin/20_rbee_hive/src/

# Check main.rs for initialization
cat bin/20_rbee_hive/src/main.rs
```

---

### 1.5 Understand Narration Flow

**Current narration in HTTP mode:**
```text
rbee-hive operation
  ↓ NARRATE.emit() with job_id
  ↓ Goes to SSE sink
  ↓ Routed to job-specific channel
  ↓ Streamed via HTTP SSE
  ↓ queen-rbee receives and forwards
```

**Target narration in integrated mode:**
```text
Direct function call
  ↓ NARRATE.emit() with job_id
  ↓ Goes to SSE sink (same as before!)
  ↓ Routed to job-specific channel
  ↓ queen-rbee reads from channel
  ↓ Forwards to client (no HTTP)
```

**Investigation tasks:**
- [ ] Verify narration already uses job_id routing
- [ ] Check if SSE sink works without HTTP
- [ ] Test if narration events can be collected in-memory
- [ ] Verify job-specific channels work in same process

**Key insight:** Narration might "just work" because it's already job-scoped!

---

## Phase 2: Design Decisions

### 2.1 Dependency Management

**Option A: Optional Dependencies (Recommended)**
```toml
[dependencies]
# Only included when local-hive feature is enabled
rbee-hive-worker-lifecycle = { path = "...", optional = true }
rbee-hive-model-catalog = { path = "...", optional = true }

[features]
local-hive = [
    "rbee-hive-worker-lifecycle",
    "rbee-hive-model-catalog",
]
```

**Pros:**
- Clean separation (distributed vs integrated builds)
- Smaller binary for distributed mode
- No circular dependency risk

**Cons:**
- Need to maintain two code paths
- More #[cfg(feature = "local-hive")] guards

**Option B: Always Include (Not Recommended)**
```toml
[dependencies]
rbee-hive-worker-lifecycle = { path = "..." }
rbee-hive-model-catalog = { path = "..." }
```

**Pros:**
- Simpler code (no feature guards)
- Single code path

**Cons:**
- Larger binary even for distributed mode
- Potential circular dependencies
- Defeats purpose of mode separation

**Decision:** Use Option A (optional dependencies)

---

### 2.2 State Initialization

**Challenge:** rbee-hive crates expect initialized state (registries, catalogs, etc.)

**Option A: Lazy Initialization**
```rust
#[cfg(feature = "local-hive")]
static HIVE_STATE: Lazy<HiveState> = Lazy::new(|| {
    HiveState {
        worker_registry: Arc::new(WorkerRegistry::new()),
        model_catalog: Arc::new(ModelCatalog::new()),
        // ... other state
    }
});
```

**Pros:**
- Simple, initialized on first use
- No startup overhead if not used

**Cons:**
- Hidden initialization
- Hard to handle initialization errors

**Option B: Explicit Initialization**
```rust
#[cfg(feature = "local-hive")]
pub struct IntegratedHive {
    worker_registry: Arc<WorkerRegistry>,
    model_catalog: Arc<ModelCatalog>,
}

impl IntegratedHive {
    pub fn new() -> Result<Self> {
        // Explicit initialization with error handling
    }
}
```

**Pros:**
- Clear initialization point
- Proper error handling
- Testable

**Cons:**
- Need to pass state around
- More boilerplate

**Decision:** Use Option B (explicit initialization) - better for production

---

### 2.3 Error Handling

**Challenge:** Direct calls return Result<T>, but HTTP mode streams errors via SSE.

**Current HTTP mode:**
```rust
// Error is sent as SSE event
NARRATE.error("worker_spawn_failed").emit();
// Then HTTP returns 500
```

**Integrated mode options:**

**Option A: Convert to Narration**
```rust
match worker_lifecycle::spawn_worker(...).await {
    Ok(worker_id) => {
        NARRATE.success("worker_spawned").context(&worker_id).emit();
    }
    Err(e) => {
        NARRATE.error("worker_spawn_failed").context(e.to_string()).emit();
        return Err(e);
    }
}
```

**Option B: Propagate Directly**
```rust
// Just return the error, let caller handle it
worker_lifecycle::spawn_worker(...).await?
```

**Decision:** Use Option A (convert to narration) - maintains consistency with HTTP mode

---

### 2.4 Result Streaming

**Challenge:** HTTP mode streams results progressively. Integrated mode is synchronous.

**HTTP mode:**
```text
POST /v1/jobs → 202 Accepted {job_id}
GET /v1/jobs/{job_id}/stream → SSE events
  data: [worker-lc ] spawn_start    : Starting worker...
  data: [worker-lc ] spawn_progress : Loading model...
  data: [worker-lc ] spawn_complete : Worker started
  data: [DONE]
```

**Integrated mode options:**

**Option A: Collect All Events**
```rust
// Collect all narration events
let events = execute_integrated(operation).await?;
// Then stream them to client
for event in events {
    emit_to_client(event);
}
```

**Option B: Real-time Streaming**
```rust
// Narration events go to SSE sink as they happen
// Client reads from SSE sink (same as HTTP mode)
execute_integrated(operation).await?;
```

**Decision:** Use Option B (real-time streaming) - narration already handles this!

---

## Phase 3: Implementation Plan

### 3.1 Step 1: Add Optional Dependencies

**File:** `bin/10_queen_rbee/Cargo.toml`

```toml
[dependencies]
# Existing dependencies...

# TEAM-XXX: Optional dependencies for integrated mode
rbee-hive-worker-lifecycle = { path = "../25_rbee_hive_crates/worker-lifecycle", optional = true }
rbee-hive-model-catalog = { path = "../25_rbee_hive_crates/model-catalog", optional = true }
rbee-hive-model-provisioner = { path = "../25_rbee_hive_crates/model-provisioner", optional = true }

[features]
# TEAM-XXX: Integrated mode feature
local-hive = [
    "rbee-hive-worker-lifecycle",
    "rbee-hive-model-catalog",
    "rbee-hive-model-provisioner",
]
```

**Verification:**
```bash
# Build without feature (should work)
cargo build --bin queen-rbee

# Build with feature (should work)
cargo build --bin queen-rbee --features local-hive

# Check binary size difference
ls -lh target/debug/queen-rbee
```

---

### 3.2 Step 2: Create Integrated State

**File:** `bin/10_queen_rbee/src/integrated_hive.rs` (NEW)

```rust
//! Integrated hive state for Mode 3 (local-hive feature)
//!
//! TEAM-XXX: Created for direct function call mode
//!
//! This module manages the state needed to call rbee-hive crates directly
//! without HTTP. Only compiled when local-hive feature is enabled.

#[cfg(feature = "local-hive")]
use rbee_hive_worker_lifecycle::WorkerRegistry;
#[cfg(feature = "local-hive")]
use rbee_hive_model_catalog::ModelCatalog;
#[cfg(feature = "local-hive")]
use std::sync::Arc;

#[cfg(feature = "local-hive")]
pub struct IntegratedHive {
    pub worker_registry: Arc<WorkerRegistry>,
    pub model_catalog: Arc<ModelCatalog>,
    // Add other state as needed
}

#[cfg(feature = "local-hive")]
impl IntegratedHive {
    /// Initialize integrated hive state
    ///
    /// This should be called once at queen startup when local-hive feature is enabled.
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            worker_registry: Arc::new(WorkerRegistry::new()),
            model_catalog: Arc::new(ModelCatalog::new()?),
        })
    }
}

#[cfg(not(feature = "local-hive"))]
pub struct IntegratedHive; // Empty stub for non-integrated builds

#[cfg(not(feature = "local-hive"))]
impl IntegratedHive {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self)
    }
}
```

**Add to main.rs:**
```rust
mod integrated_hive;

// In main():
let integrated_hive = integrated_hive::IntegratedHive::new()?;
let integrated_hive = Arc::new(integrated_hive);

// Pass to router state
let job_state = http::SchedulerState {
    registry: job_server,
    config: config.clone(),
    hive_registry: worker_registry.clone(),
    integrated_hive: integrated_hive.clone(), // NEW
};
```

---

### 3.3 Step 3: Implement execute_integrated()

**File:** `bin/10_queen_rbee/src/hive_forwarder.rs`

```rust
#[cfg(feature = "local-hive")]
async fn execute_integrated(
    job_id: &str,
    operation: Operation,
    integrated_hive: Arc<IntegratedHive>,
) -> Result<()> {
    use rbee_hive_worker_lifecycle;
    use rbee_hive_model_catalog;
    
    match operation {
        Operation::WorkerSpawn { hive_id, model_id, backend, device_id } => {
            NARRATE
                .action("integrated_spawn")
                .job_id(job_id)
                .human("Spawning worker via integrated mode...")
                .emit();
            
            // Direct function call (no HTTP!)
            let worker_id = rbee_hive_worker_lifecycle::spawn_worker(
                integrated_hive.worker_registry.clone(),
                model_id,
                backend,
                device_id,
            ).await?;
            
            NARRATE
                .action("integrated_spawn")
                .job_id(job_id)
                .context(&worker_id)
                .human("Worker {} spawned successfully")
                .emit();
            
            Ok(())
        }
        
        Operation::WorkerList { hive_id } => {
            let workers = rbee_hive_worker_lifecycle::list_workers(
                integrated_hive.worker_registry.clone()
            ).await?;
            
            for worker in workers {
                NARRATE
                    .action("integrated_list")
                    .job_id(job_id)
                    .context(&worker.id)
                    .human("Worker: {} ({})", worker.id, worker.state)
                    .emit();
            }
            
            Ok(())
        }
        
        // TODO: Implement other operations
        _ => {
            Err(anyhow::anyhow!("Operation not yet implemented in integrated mode"))
        }
    }
}
```

---

### 3.4 Step 4: Update forward_to_hive()

**File:** `bin/10_queen_rbee/src/hive_forwarder.rs`

```rust
pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
    config: Arc<RbeeConfig>,
    #[cfg(feature = "local-hive")]
    integrated_hive: Option<Arc<IntegratedHive>>,
) -> Result<()> {
    // ... existing mode detection ...
    
    // TEAM-XXX: Route to integrated mode if available
    #[cfg(feature = "local-hive")]
    if is_localhost && has_integrated {
        if let Some(hive) = integrated_hive {
            NARRATE
                .action("forward_mode")
                .job_id(job_id)
                .human("Using integrated mode (direct calls)")
                .emit();
            
            return execute_integrated(job_id, operation, hive).await;
        }
    }
    
    // Fall back to HTTP mode
    // ... existing HTTP forwarding code ...
}
```

---

### 3.5 Step 5: Update job_router.rs

**File:** `bin/10_queen_rbee/src/job_router.rs`

```rust
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub config: Arc<RbeeConfig>,
    pub hive_registry: Arc<WorkerRegistry>,
    #[cfg(feature = "local-hive")]
    pub integrated_hive: Arc<IntegratedHive>, // NEW
}

// In route_operation():
op if op.should_forward_to_hive() => {
    hive_forwarder::forward_to_hive(
        &job_id,
        op,
        state.config.clone(),
        #[cfg(feature = "local-hive")]
        Some(state.integrated_hive.clone()),
    ).await?;
}
```

---

## Phase 4: Testing Strategy

### 4.1 Unit Tests

**Test each operation in isolation:**

```rust
#[cfg(all(test, feature = "local-hive"))]
mod integrated_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_worker_spawn_integrated() {
        let hive = IntegratedHive::new().unwrap();
        let job_id = "test-job-123";
        
        let operation = Operation::WorkerSpawn {
            hive_id: "localhost".to_string(),
            model_id: "test-model".to_string(),
            backend: Some("cpu".to_string()),
            device_id: None,
        };
        
        let result = execute_integrated(job_id, operation, Arc::new(hive)).await;
        assert!(result.is_ok());
    }
}
```

---

### 4.2 Integration Tests

**Compare HTTP vs Integrated:**

```bash
# Test 1: HTTP mode (baseline)
cargo build --bin queen-rbee
./target/debug/queen-rbee &
./target/debug/rbee-hive &
time rbee-keeper worker list --hive-id localhost

# Test 2: Integrated mode
cargo build --bin queen-rbee --features local-hive
./target/debug/queen-rbee &
# No rbee-hive needed!
time rbee-keeper worker list --hive-id localhost

# Compare times - should be 50-100x faster
```

---

### 4.3 Performance Benchmarks

**Create benchmark:**

```rust
#[cfg(all(test, feature = "local-hive"))]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn bench_worker_list_http(c: &mut Criterion) {
        c.bench_function("worker_list_http", |b| {
            b.iter(|| {
                // HTTP mode
            });
        });
    }
    
    fn bench_worker_list_integrated(c: &mut Criterion) {
        c.bench_function("worker_list_integrated", |b| {
            b.iter(|| {
                // Integrated mode
            });
        });
    }
    
    criterion_group!(benches, bench_worker_list_http, bench_worker_list_integrated);
    criterion_main!(benches);
}
```

---

## Phase 5: Potential Issues & Solutions

### Issue 1: Circular Dependencies

**Problem:** queen-rbee depends on rbee-hive crates, which might depend on shared crates that queen also uses.

**Solution:**
- Use optional dependencies
- Ensure shared crates are truly shared (in 99_shared_crates/)
- Check for version conflicts

**Investigation:**
```bash
cargo tree --features local-hive | grep -i cycle
```

---

### Issue 2: State Synchronization

**Problem:** If both HTTP and integrated modes are used simultaneously, state might diverge.

**Solution:**
- Don't support mixed mode (all localhost ops use same mode)
- Document that local-hive feature means "integrated only"
- Remove localhost from hives.conf when using local-hive

---

### Issue 3: Narration Event Ordering

**Problem:** Integrated mode might emit events in different order than HTTP mode.

**Solution:**
- Test event ordering carefully
- Ensure job_id routing works correctly
- Verify SSE sink handles same-process events

---

### Issue 4: Error Propagation

**Problem:** HTTP mode returns 500 errors. Integrated mode returns Result<T>.

**Solution:**
- Convert errors to narration events
- Maintain same error format as HTTP mode
- Test error cases thoroughly

---

## Phase 6: Rollout Plan

### 6.1 Alpha Release (Internal Testing)

**Goals:**
- Verify compilation with local-hive feature
- Test basic operations (WorkerList, WorkerSpawn)
- Measure performance improvement

**Checklist:**
- [ ] Compiles with and without local-hive feature
- [ ] Basic operations work in integrated mode
- [ ] Performance is 50-100x better
- [ ] No crashes or panics

---

### 6.2 Beta Release (Limited Users)

**Goals:**
- Test all operations
- Verify error handling
- Collect user feedback

**Checklist:**
- [ ] All forwarded operations implemented
- [ ] Error handling matches HTTP mode
- [ ] Documentation updated
- [ ] Migration guide for users

---

### 6.3 Stable Release

**Goals:**
- Production-ready integrated mode
- Full documentation
- Performance benchmarks published

**Checklist:**
- [ ] All tests passing
- [ ] Benchmarks show expected improvement
- [ ] User guide updated
- [ ] CHANGELOG.md updated

---

## Phase 7: Documentation Updates

### Files to Update

1. **`QUEEN_TO_HIVE_COMMUNICATION_MODES.md`**
   - Change Mode 3 status from "TODO" to "Implemented"
   - Add actual performance numbers
   - Add usage examples

2. **`hive_forwarder.rs`**
   - Update module docs
   - Remove TODO comments
   - Add examples

3. **`.arch/CHANGELOG.md`**
   - Add entry for Mode 3 implementation
   - Document performance improvements

4. **`README.md`** (if exists)
   - Mention local-hive feature
   - Explain when to use it

---

## Success Criteria

### Must Have
- [ ] All forwarded operations work in integrated mode
- [ ] Performance is 50-100x better than HTTP mode
- [ ] No regressions in HTTP mode
- [ ] All tests pass
- [ ] Documentation complete

### Nice to Have
- [ ] Benchmarks published
- [ ] Migration guide for users
- [ ] Performance monitoring/metrics
- [ ] A/B testing framework

---

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| 1. Investigation | 2-4 hours | None |
| 2. Design | 2-3 hours | Phase 1 |
| 3. Implementation | 8-16 hours | Phase 2 |
| 4. Testing | 4-8 hours | Phase 3 |
| 5. Issue Resolution | 4-8 hours | Phase 4 |
| 6. Rollout | 2-4 hours | Phase 5 |
| 7. Documentation | 2-4 hours | Phase 6 |
| **Total** | **24-47 hours** | Sequential |

---

## Questions to Answer During Investigation

1. **Architecture:**
   - [ ] Can rbee-hive crates be called directly without HTTP?
   - [ ] Is there any HTTP-specific logic that needs to be bypassed?
   - [ ] Are there any singleton patterns that conflict?

2. **State:**
   - [ ] How is state initialized in rbee-hive?
   - [ ] Can state be shared between queen and integrated hive?
   - [ ] Are there any race conditions to worry about?

3. **Narration:**
   - [ ] Does narration work without HTTP/SSE?
   - [ ] Can events be collected in-memory?
   - [ ] Is job_id routing sufficient?

4. **Performance:**
   - [ ] What's the actual overhead of HTTP mode?
   - [ ] What's the expected overhead of integrated mode?
   - [ ] Are there any bottlenecks we're not considering?

5. **Compatibility:**
   - [ ] Can HTTP and integrated modes coexist?
   - [ ] Do we need to support mixed mode?
   - [ ] What's the migration path for users?

---

## Final Checklist

Before declaring Mode 3 complete:

- [ ] Investigation phase complete (all questions answered)
- [ ] Design decisions documented
- [ ] Implementation complete (all operations)
- [ ] Tests written and passing
- [ ] Performance benchmarks show 50-100x improvement
- [ ] Documentation updated
- [ ] User migration guide written
- [ ] Code reviewed
- [ ] No regressions in HTTP mode
- [ ] Feature flag works correctly
- [ ] Handoff document created

---

## References

- `bin/.plan/QUEEN_TO_HIVE_COMMUNICATION_MODES.md` - Mode overview
- `bin/10_queen_rbee/src/hive_forwarder.rs` - Current implementation
- `bin/20_rbee_hive/src/job_router.rs` - Hive operation handlers
- `bin/99_shared_crates/rbee-operations/src/lib.rs` - Operation types

---

**Good luck, next team! This is a high-impact feature that will make localhost operations 50-100x faster. Take your time with the investigation phase - understanding the current architecture is key to a successful implementation.**
