# TEAM-305: Fix Circular Dependency (JobRegistry)

**Status:** 🚨 CRITICAL - TECHNICAL DEBT  
**Priority:** P0 (Blocking)  
**Estimated Duration:** 2-3 hours  
**Dependencies:** TEAM-304 (DONE signal fix)  
**Blocks:** Proper E2E testing with real JobRegistry

---

## Mission

Fix the circular dependency between `job-server` and `narration-core` so that test binaries can use the real `JobRegistry` instead of a simplified `HashMap`.

---

## Problem Statement

**Current State:**
- ❌ `job-server` depends on `narration-core`
- ❌ `narration-core` test binaries need `job-server`
- ❌ Circular dependency prevents this
- ❌ Test binaries use simplified `HashMap` instead

**Impact:**
- Test binaries don't use real production code
- Job lifecycle not tested with narration
- Job state transitions not tested
- False confidence in E2E tests
- **Production coverage: 85% instead of 95%**

**Technical Debt:**
- See `.plan/TEAM_303_TECHNICAL_DEBT.md` for full analysis

---

## Solution: Extract Job Registry Interface

### Option A: Create job-registry-interface Crate (RECOMMENDED)

**New crate structure:**
```
bin/99_shared_crates/job-registry-interface/
├── Cargo.toml
└── src/
    └── lib.rs
```

**Dependency graph:**
```
job-registry-interface (new, no dependencies)
    ↑
    ├── job-server (implements trait)
    ├── narration-core (depends on interface for test binaries)
    └── test binaries (use real JobRegistry via interface)
```

---

## Implementation Tasks

### Task 1: Create job-registry-interface Crate (1 hour)

**File:** `bin/99_shared_crates/job-registry-interface/Cargo.toml`

```toml
[package]
name = "job-registry-interface"
version = "0.1.0"
edition = "2021"
publish = false

[dependencies]
tokio = { workspace = true, features = ["sync"] }
serde_json = { workspace = true }
chrono = "0.4"
```

**File:** `bin/99_shared_crates/job-registry-interface/src/lib.rs`

```rust
// TEAM-305: Job registry interface for breaking circular dependency

use tokio::sync::mpsc::UnboundedReceiver;

/// Job state in the registry
#[derive(Debug, Clone)]
pub enum JobState {
    Queued,
    Running,
    Completed,
    Failed(String),
}

/// Job registry trait
///
/// TEAM-305: Interface allows narration-core test binaries to use
/// real JobRegistry without circular dependency
pub trait JobRegistry<T>: Send + Sync {
    /// Create a new job and return job_id
    fn create_job(&self) -> String;
    
    /// Set payload for a job (for deferred execution)
    fn set_payload(&self, job_id: &str, payload: serde_json::Value);
    
    /// Take payload from a job (consumes it)
    fn take_payload(&self, job_id: &str) -> Option<serde_json::Value>;
    
    /// Check if job exists
    fn has_job(&self, job_id: &str) -> bool;
    
    /// Get job state
    fn get_job_state(&self, job_id: &str) -> Option<JobState>;
    
    /// Update job state
    fn update_state(&self, job_id: &str, state: JobState);
    
    /// Set token receiver for streaming
    fn set_token_receiver(&self, job_id: &str, receiver: UnboundedReceiver<T>);
    
    /// Take the token receiver for a job (consumes it)
    fn take_token_receiver(&self, job_id: &str) -> Option<UnboundedReceiver<T>>;
    
    /// Remove a job from the registry
    fn remove_job(&self, job_id: &str);
    
    /// Get count of jobs in registry
    fn job_count(&self) -> usize;
    
    /// Get all job IDs
    fn job_ids(&self) -> Vec<String>;
}
```

### Task 2: Update job-server to Implement Trait (30 min)

**File:** `bin/99_shared_crates/job-server/Cargo.toml`

```toml
[dependencies]
# ... existing dependencies ...
job-registry-interface = { path = "../job-registry-interface" }  # TEAM-305
```

**File:** `bin/99_shared_crates/job-server/src/lib.rs`

```rust
// TEAM-305: Implement JobRegistry trait
use job_registry_interface::{JobRegistry as JobRegistryTrait, JobState as JobStateInterface};

// Re-export interface types
pub use job_registry_interface::{JobRegistry as JobRegistryTrait, JobState};

impl<T> JobRegistryTrait<T> for JobRegistry<T>
where
    T: Send + 'static,
{
    fn create_job(&self) -> String {
        self.create_job()
    }
    
    fn set_payload(&self, job_id: &str, payload: serde_json::Value) {
        self.set_payload(job_id, payload)
    }
    
    fn take_payload(&self, job_id: &str) -> Option<serde_json::Value> {
        self.take_payload(job_id)
    }
    
    fn has_job(&self, job_id: &str) -> bool {
        self.has_job(job_id)
    }
    
    fn get_job_state(&self, job_id: &str) -> Option<JobState> {
        self.get_job_state(job_id)
    }
    
    fn update_state(&self, job_id: &str, state: JobState) {
        self.update_state(job_id, state)
    }
    
    fn set_token_receiver(&self, job_id: &str, receiver: UnboundedReceiver<T>) {
        self.set_token_receiver(job_id, receiver)
    }
    
    fn take_token_receiver(&self, job_id: &str) -> Option<UnboundedReceiver<T>> {
        self.take_token_receiver(job_id)
    }
    
    fn remove_job(&self, job_id: &str) {
        self.remove_job(job_id);
    }
    
    fn job_count(&self) -> usize {
        self.job_count()
    }
    
    fn job_ids(&self) -> Vec<String> {
        self.job_ids()
    }
}
```

### Task 3: Update narration-core Test Binaries (1 hour)

**File:** `bin/99_shared_crates/narration-core/Cargo.toml`

```toml
[dependencies]
# ... existing dependencies ...
job-registry-interface = { path = "../job-registry-interface", optional = true }  # TEAM-305

[dev-dependencies]
# ... existing dependencies ...
job-server = { path = "../job-server" }  # TEAM-305: Now we can use it!

[features]
axum = ["dep:axum", "dep:reqwest", "dep:futures", "dep:job-registry-interface"]  # TEAM-305
```

**File:** `narration-core/tests/bin/fake_queen.rs`

```rust
// TEAM-305: Use real JobRegistry via interface
use job_registry_interface::JobRegistry as JobRegistryTrait;
use job_server::JobRegistry;
use std::sync::Arc;

/// Queen state
struct QueenState {
    registry: Arc<JobRegistry<String>>,  // TEAM-305: Real JobRegistry!
    hive_url: Option<String>,
}

async fn create_job_handler(
    State(state): State<Arc<QueenState>>,
    headers: HeaderMap,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    // TEAM-305: Use real JobRegistry methods
    let job_id = state.registry.create_job();
    state.registry.set_payload(&job_id, payload.clone());
    
    // ... rest of implementation ...
}

#[tokio::main]
async fn main() {
    // TEAM-305: Create real JobRegistry
    let state = Arc::new(QueenState {
        registry: Arc::new(JobRegistry::new()),
        hive_url,
    });
    
    // ... rest of implementation ...
}
```

**File:** `narration-core/tests/bin/fake_hive.rs`

```rust
// TEAM-305: Same changes as fake_queen.rs
use job_registry_interface::JobRegistry as JobRegistryTrait;
use job_server::JobRegistry;

struct HiveState {
    registry: Arc<JobRegistry<String>>,  // TEAM-305: Real JobRegistry!
}

// ... update all methods to use real JobRegistry ...
```

### Task 4: Verify and Test (30 min)

**Build test binaries:**
```bash
cargo build --bin fake-queen-rbee --bin fake-rbee-hive --features axum
```

**Run E2E tests:**
```bash
cargo test -p observability-narration-core --test e2e_real_processes --features axum -- --ignored
```

**Verify:**
- [ ] No circular dependency errors
- [ ] Test binaries compile
- [ ] Test binaries use real JobRegistry
- [ ] All E2E tests pass

---

## Verification Checklist

- [ ] job-registry-interface crate created
- [ ] job-server implements trait
- [ ] narration-core depends on interface (not job-server)
- [ ] Test binaries use real JobRegistry
- [ ] No circular dependency
- [ ] All tests pass
- [ ] Production coverage increased to 95%

---

## Success Criteria

1. **No Circular Dependency**
   - narration-core → job-registry-interface
   - job-server → job-registry-interface
   - No circular reference

2. **Real JobRegistry in Tests**
   - Test binaries use `job_server::JobRegistry`
   - Not simplified `HashMap`
   - Full job lifecycle tested

3. **Tests Pass**
   - All existing tests still pass
   - E2E tests use real JobRegistry
   - Job state transitions tested

---

## Production Coverage Impact

### Before (TEAM-303)
- Test binaries use HashMap
- **Coverage: 85%**
- Missing: job lifecycle integration

### After (TEAM-305)
- Test binaries use real JobRegistry
- **Coverage: 95%**
- Includes: job lifecycle integration

---

## Handoff to TEAM-306

Document in `.plan/TEAM_305_HANDOFF.md`:

1. **What Was Fixed**
   - Circular dependency resolved
   - Test binaries use real JobRegistry
   - Production coverage increased to 95%

2. **Architecture**
   - job-registry-interface provides abstraction
   - Clean dependency graph
   - No circular references

3. **Next Steps**
   - TEAM-306: Context propagation tests
   - TEAM-307: Failure scenario tests
   - TEAM-308: Fix all broken tests

---

**TEAM-305 Mission:** Fix circular dependency and restore real JobRegistry in tests

**Priority:** P0 - CRITICAL TECHNICAL DEBT

**Estimated Time:** 2-3 hours
