# Implementation Guide for Next Team
# Created by: TEAM-077
# Date: 2025-10-11
# Status: ACTIONABLE PLAN

## Final Structure: 14 M1 Feature Files

```
010-ssh-registry-management (10) ✅ EXISTS
020-model-catalog (13) ✅ EXISTS - rename from model-provisioning
025-worker-provisioning ⚠️ NEW - Build from git + catalog
030-queen-rbee-worker-registry ⚠️ NEW - Global registry
040-rbee-hive-worker-registry (9) ✅ EXISTS - rename from 060
050-ssh-preflight-validation ⚠️ NEW - queen-rbee → rbee-hive SSH checks
060-rbee-hive-preflight-validation ⚠️ NEW - rbee-hive readiness checks
070-worker-resource-preflight (10) ✅ EXISTS - rename from 030
080-worker-rbee-lifecycle (11) ✅ EXISTS - rename from 040
090-rbee-hive-lifecycle (7) ✅ EXISTS - rename from 070
100-queen-rbee-lifecycle (3) ✅ EXISTS - rename from 080
110-inference-execution (11) ✅ EXISTS - rename from 050
120-input-validation (6) ✅ EXISTS - rename from 090
130-cli-commands (9) ✅ EXISTS - rename from 100
140-end-to-end-flows (2) ✅ EXISTS - rename from 110
```

## ⚠️ ORDER MATTERS - Follow Phases Sequentially!

---

## Phase 1: Rename Files (30 min)

**Why first:** Establish structure before adding new files.

```bash
cd test-harness/bdd/tests/features

# Rename in REVERSE order to avoid conflicts
mv 110-end-to-end-flows.feature 140-end-to-end-flows.feature
mv 100-cli-commands.feature 130-cli-commands.feature
mv 090-input-validation.feature 120-input-validation.feature
mv 080-queen-rbee-lifecycle.feature 100-queen-rbee-lifecycle.feature
mv 070-rbee-hive-lifecycle.feature 090-rbee-hive-lifecycle.feature
mv 050-inference-execution.feature 110-inference-execution.feature
mv 060-pool-management.feature 040-rbee-hive-worker-registry.feature
mv 040-worker-lifecycle.feature 080-worker-rbee-lifecycle.feature
mv 030-worker-preflight-checks.feature 070-worker-resource-preflight.feature
mv 020-model-provisioning.feature 020-model-catalog.feature
```

**Update feature names inside files:**
- 020: "Model Catalog"
- 040: "rbee-hive Worker Registry"
- 080: "worker-rbee Daemon Lifecycle"

**✅ Checkpoint:** `cargo check --bin bdd-runner`

---

## Phase 2: Create New Features (2 hours)

### 2.1: Create 025-worker-provisioning.feature

**Scenarios (7):**
1. Build worker from git succeeds
2. Worker binary registered in catalog
3. Query available worker types
4. Build triggered when binary not found
5. EH-020a: Cargo build failure
6. EH-020b: Missing CUDA toolkit
7. EH-020c: Binary verification failed

**Reference:** `bin/.specs/.gherkin/test-001.md` Phase 3b

### 2.2: Create 030-queen-rbee-worker-registry.feature

**Scenarios (6):**
1. Register worker from rbee-hive
2. Query all workers
3. Filter by capability
4. Update worker state
5. Remove worker
6. Stale worker cleanup

**Key:** In-memory, just HTTP endpoints!

### 2.3: Create 050-ssh-preflight-validation.feature

**Scenarios (6):**
1. SSH connection validation succeeds
2. EH-001a: SSH connection timeout
3. EH-001b: SSH authentication failure
4. SSH command execution test
5. Network latency check (<100ms)
6. rbee-hive binary exists on remote node

**Stakeholder:** DevOps / SSH operations
**Component:** queen-rbee
**Timing:** Phase 2a (before starting rbee-hive)

### 2.4: Create 060-rbee-hive-preflight-validation.feature

**Scenarios (4):**
1. rbee-hive HTTP API health check succeeds
2. rbee-hive version compatibility check
3. Backend catalog populated (CUDA/Metal/CPU detected)
4. Sufficient resources available (RAM, disk)

**Stakeholder:** Platform readiness team
**Component:** rbee-hive
**Timing:** Phase 3a (before spawning workers)

### 2.5: Verify 070-worker-resource-preflight.feature

**This file already exists!** Just verify scenarios:
1. EH-004a/b: RAM availability checks
2. EH-005a: VRAM availability - GPU FAIL FAST!
3. EH-006a/b: Disk space checks
4. EH-009a/b: Backend availability - GPU FAIL FAST!

**Stakeholder:** Resource management team
**Component:** rbee-hive (checks before spawning worker)
**Timing:** Phase 8 (before spawning specific worker)

**✅ Checkpoint:** `cargo check --bin bdd-runner`

---

## Phase 3: Add Step Definitions (3 hours)

### 3.1: Create Step Files

**Files to create:**
- `src/steps/worker_provisioning.rs`
- `src/steps/queen_rbee_registry.rs`
- `src/steps/ssh_preflight.rs`
- `src/steps/rbee_hive_preflight.rs`
- (worker_resource_preflight.rs already exists!)

**Update `src/steps/mod.rs`:**
```rust
pub mod worker_provisioning;
pub mod queen_rbee_registry;
pub mod ssh_preflight;
pub mod rbee_hive_preflight;
pub mod worker_resource_preflight;  // Already exists
```

### 3.2: Stub All Functions First

```rust
#[given(expr = "worker binary {string} not in catalog")]
pub async fn given_worker_not_in_catalog(world: &mut World, worker_type: String) {
    tracing::warn!("STUB: implement later");
    // TODO
}
```

**✅ Checkpoint:** `cargo check --bin bdd-runner`

### 3.3: Key Functions Needed

**worker_provisioning.rs:**
- `given_worker_binary_not_in_catalog`
- `when_rbee_hive_builds_worker`
- `then_worker_binary_registered`
- `when_cargo_build_fails` (EH-020a)

**queen_rbee_registry.rs:**
- `given_queen_rbee_has_workers`
- `when_rbee_hive_reports_worker`
- `when_query_all_workers`
- `when_filter_by_capability`
- `then_queen_rbee_returns_workers`

**ssh_preflight.rs:**
- `when_ssh_preflight_checks`
- `then_ssh_preflight_passes`
- `when_ssh_connection_timeout` (EH-001a)
- `when_ssh_authentication_fails` (EH-001b)
- `when_check_network_latency`

**rbee_hive_preflight.rs:**
- `when_rbee_hive_preflight_checks`
- `then_rbee_hive_preflight_passes`
- `when_check_http_health`
- `when_check_version_compatibility`
- `when_check_backend_catalog`

**worker_resource_preflight.rs (already exists):**
- Verify existing functions for EH-004a/b, EH-005a, EH-006a/b, EH-009a/b

**✅ Checkpoint:** `cargo test --bin bdd-runner` (will fail, that's OK)

---

## Phase 4: Implement Product Code (5 hours)

### 4.1: Worker Provisioning (Priority 1)

**Create:**
- `bin/rbee-hive/src/worker_binaries_catalog.rs` - SQLite catalog
- `bin/rbee-hive/src/worker_provisioner.rs` - Cargo build logic
- `bin/rbee-hive/src/api/worker_binaries.rs` - HTTP endpoints

**Key API:**
```rust
POST /v1/worker-binaries/provision
{
  "worker_type": "llm-cuda-worker-rbee",
  "features": ["cuda"]
}
```

**Implementation:**
```rust
// cargo build --release --bin <type> --features <features>
// Register in catalog
// Return binary path
```

### 4.2: Queen-rbee Registry (Priority 2)

**Create:**
- `bin/queen-rbee/src/worker_registry.rs` - In-memory HashMap
- `bin/queen-rbee/src/api/workers.rs` - HTTP endpoints

**Key APIs:**
```rust
POST /v1/workers/register
GET /v1/workers/list?capability=<capability>
DELETE /v1/workers/{id}
```

### 4.3: Preflight Checkers (Priority 3)

**Create 3 separate checkers:**

#### 4.3.1: SSH Preflight Checker
- **File:** `bin/queen-rbee/src/preflight/ssh.rs`
- **Stakeholder:** DevOps
- **Checks:** Connection, auth, latency, binary exists

#### 4.3.2: rbee-hive Preflight Checker
- **File:** `bin/queen-rbee/src/preflight/rbee_hive.rs`
- **Stakeholder:** Platform team
- **Checks:** HTTP health, version, backend catalog, resources

#### 4.3.3: Worker Resource Preflight
- **File:** `bin/rbee-hive/src/preflight/worker_resources.rs` (enhance existing)
- **Stakeholder:** Resource management
- **Checks:** RAM, VRAM (GPU FAIL FAST!), disk, backend

**✅ Checkpoint:** `cargo test --bin bdd-runner` (pass rate should improve)

---

## Phase 5: Wire Up Tests (2 hours)

Replace stub functions with real API calls:

```rust
#[when(expr = "rbee-hive builds worker from git")]
pub async fn when_rbee_hive_builds_worker(world: &mut World) {
    let provisioner = world.get_provisioner();
    let result = provisioner.provision("llm-worker-rbee", &["cuda"]).await;
    world.build_result = Some(result);
}
```

**✅ Checkpoint:** `cargo test --bin bdd-runner` (target: 80%+ pass rate)

---

## Phase 6: Documentation (30 min)

### Handoff Document (MAX 2 PAGES!)

```markdown
# TEAM-XXX SUMMARY

**Deliverables:**
- ✅ 14 M1 feature files (4 new, 10 renamed)
- ✅ XX step functions implemented
- ✅ Worker provisioning (cargo build from git)
- ✅ queen-rbee registry (in-memory)
- ✅ 3 separate preflight validation features (SSH, rbee-hive, worker)

**Pass Rate:** XX% → YY% (+ZZ%)

**Code Examples:**
[Show 2-3 key functions]

**Verification:**
cargo test --bin bdd-runner
```

---

## Critical Warnings

### GPU FAIL FAST Policy

**Enforce in all GPU errors:**
- ❌ NO CPU fallback
- ✅ Exit code 1
- ✅ Message: "GPU FAIL FAST! NO CPU FALLBACK!"

**Scenarios:**
- EH-005a: VRAM exhausted
- EH-009a/b: Backend/CUDA not available

### Compilation Checkpoints

**You MUST compile after each phase:**
```bash
cargo check --bin bdd-runner
```

**If fails, STOP and fix!**

### Dependency Order

1. Rename files before creating new ones
2. Create features before step definitions  
3. Stub functions before implementation
4. Foundational features before dependent ones

---

## Quick Reference

**Test Commands:**
```bash
# All tests
cargo test --bin bdd-runner -- --nocapture

# Specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/025-worker-provisioning.feature \
  cargo test --bin bdd-runner -- --nocapture

# Check compilation
cargo check --bin bdd-runner
```

**File Structure:**
```
test-harness/bdd/
├── tests/features/           # .feature files
├── src/steps/                # Step definitions
└── src/world.rs              # Test state
```

**Time Estimate:** ~13 hours total

**Target:** 80%+ pass rate for M1 features

---

## Summary

**14 M1 Feature Files = Better Stakeholder Clarity!**

**Each preflight level has:**
- ✅ Different stakeholder (DevOps, Platform, Resource Mgmt)
- ✅ Different component (queen-rbee, rbee-hive, rbee-hive)
- ✅ Different timing in flow (Phase 2a, 3a, 8)
- ✅ Clear separation of concerns

**New Features (4):**
1. 025-worker-provisioning
2. 030-queen-rbee-worker-registry
3. 050-ssh-preflight-validation
4. 060-rbee-hive-preflight-validation

**Total Time:** ~13 hours

---

**TEAM-077 says:** MORE FILES = BETTER CLARITY! Follow phases in order! Check compilation after each phase! Enforce GPU FAIL FAST! 🐝
