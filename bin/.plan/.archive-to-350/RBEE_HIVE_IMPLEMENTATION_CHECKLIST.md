# rbee-hive Implementation Checklist

**Master Plan:** TEAM-267 to TEAM-275  
**Created by:** TEAM-266  
**Date:** Oct 23, 2025

---

## 📋 Complete Checklist

### Phase 1: Model Catalog Types (TEAM-267)

**Crate:** `rbee-hive-model-catalog`

- [ ] **Types (types.rs)**
  - [ ] ModelEntry struct with all fields
  - [ ] ModelStatus enum (Ready, Downloading, Failed)
  - [ ] ModelMetadata struct
  - [ ] ModelEntry::new() constructor
  - [ ] ModelEntry::with_metadata() builder
  - [ ] ModelEntry::is_ready() helper

- [ ] **Catalog (catalog.rs)**
  - [ ] ModelCatalog struct with Arc<Mutex<HashMap>>
  - [ ] new() constructor
  - [ ] add() method
  - [ ] get() method
  - [ ] remove() method
  - [ ] list() method
  - [ ] list_by_status() method
  - [ ] update_status() method
  - [ ] contains() method
  - [ ] len() and is_empty() methods

- [ ] **Tests**
  - [ ] test_catalog_add_get()
  - [ ] test_catalog_duplicate_add()
  - [ ] test_catalog_remove()
  - [ ] test_catalog_list()
  - [ ] test_catalog_update_status()

- [ ] **Documentation**
  - [ ] Public API documented in lib.rs
  - [ ] README.md created
  - [ ] Examples added

- [ ] **Verification**
  - [ ] `cargo check --package rbee-hive-model-catalog` passes
  - [ ] `cargo test --package rbee-hive-model-catalog` passes
  - [ ] No warnings

---

### Phase 2: Model Catalog Operations (TEAM-268)

**Files:** `rbee-hive/src/job_router.rs`, `rbee-hive/src/main.rs`

- [ ] **Dependencies**
  - [ ] Add rbee-hive-model-catalog to Cargo.toml

- [ ] **State Initialization**
  - [ ] Initialize ModelCatalog in main.rs
  - [ ] Add model_catalog to JobState struct

- [ ] **ModelList Operation**
  - [ ] Implement in job_router.rs
  - [ ] Add narration events (start, result, entries)
  - [ ] Handle empty catalog case
  - [ ] Format output as table

- [ ] **ModelGet Operation**
  - [ ] Implement in job_router.rs
  - [ ] Add narration events (start, found, details)
  - [ ] Handle not found error
  - [ ] Output JSON details

- [ ] **ModelDelete Operation**
  - [ ] Implement in job_router.rs
  - [ ] Add narration events (start, catalog removal)
  - [ ] Handle not found error
  - [ ] Note file deletion TODO

- [ ] **Verification**
  - [ ] `cargo check --bin rbee-hive` passes
  - [ ] Manual test: ModelList works
  - [ ] Manual test: ModelGet works
  - [ ] Manual test: ModelDelete works

---

### Phase 3: Model Provisioner (TEAM-269)

**Crate:** `rbee-hive-model-provisioner`

- [ ] **Core Implementation**
  - [ ] ModelProvisioner struct
  - [ ] new() constructor with cache_dir setup
  - [ ] download_model() async function
  - [ ] delete_model_files() async function
  - [ ] Progress tracking via ModelCatalog status

- [ ] **Download Logic**
  - [ ] Check if model already exists
  - [ ] Create model entry with Downloading status
  - [ ] Create model directory
  - [ ] TODO: Actual HuggingFace download
  - [ ] Update status to Ready
  - [ ] Emit narration events

- [ ] **Integration**
  - [ ] Add to rbee-hive Cargo.toml
  - [ ] Initialize in main.rs
  - [ ] Add to JobState
  - [ ] Implement ModelDownload in job_router.rs

- [ ] **Narration**
  - [ ] download_start event
  - [ ] download_progress event
  - [ ] download_complete event
  - [ ] download_error event

- [ ] **Verification**
  - [ ] `cargo check --package rbee-hive-model-provisioner` passes
  - [ ] `cargo check --bin rbee-hive` passes
  - [ ] Manual test: ModelDownload creates directory
  - [ ] Manual test: Progress events emitted

---

### Phase 4: Worker Registry (TEAM-270)

**Crate:** `rbee-hive-worker-lifecycle` (registry module)

- [ ] **Types**
  - [ ] WorkerEntry struct with all fields
  - [ ] WorkerStatus enum (Starting, Ready, Busy, Stopped, Failed)
  - [ ] started_at timestamp

- [ ] **Registry**
  - [ ] WorkerRegistry struct with Arc<Mutex<HashMap>>
  - [ ] new() constructor
  - [ ] register() method
  - [ ] get() method
  - [ ] remove() method
  - [ ] list() method
  - [ ] update_status() method

- [ ] **Tests**
  - [ ] test_registry_register()
  - [ ] test_registry_duplicate()
  - [ ] test_registry_get()
  - [ ] test_registry_remove()
  - [ ] test_registry_list()
  - [ ] test_registry_update_status()

- [ ] **Verification**
  - [ ] `cargo check --package rbee-hive-worker-lifecycle` passes
  - [ ] `cargo test --package rbee-hive-worker-lifecycle` passes
  - [ ] No warnings

---

### Phase 5: Worker Lifecycle - Spawn (TEAM-271)

**Crate:** `rbee-hive-worker-lifecycle` (spawn module)

- [ ] **Core Implementation**
  - [ ] WorkerSpawner struct
  - [ ] new() constructor
  - [ ] spawn_worker() async function
  - [ ] find_available_port() helper
  - [ ] Process spawning with tokio::process::Command

- [ ] **Spawn Logic**
  - [ ] Emit spawn_start narration
  - [ ] Find available port (9100-9200 range)
  - [ ] Spawn worker process with args
  - [ ] Get process PID
  - [ ] Create WorkerEntry
  - [ ] Register in WorkerRegistry
  - [ ] Wait for worker ready (health check TODO)
  - [ ] Update status to Ready
  - [ ] Emit spawn_complete narration

- [ ] **Integration**
  - [ ] Initialize in rbee-hive main.rs
  - [ ] Add to JobState
  - [ ] Implement WorkerSpawn in job_router.rs

- [ ] **Narration**
  - [ ] spawn_start event
  - [ ] spawn_port event
  - [ ] spawn_process event
  - [ ] spawn_complete event
  - [ ] spawn_error event

- [ ] **Known Issues**
  - [ ] Document worker binary requirement
  - [ ] Add fallback for missing binary
  - [ ] Note health check TODO

- [ ] **Verification**
  - [ ] `cargo check --package rbee-hive-worker-lifecycle` passes
  - [ ] `cargo check --bin rbee-hive` passes
  - [ ] Manual test: WorkerSpawn attempts to spawn
  - [ ] Manual test: Port allocation works
  - [ ] Manual test: Worker registered in registry

---

### Phase 6: Worker Lifecycle - Management (TEAM-272)

**Files:** `rbee-hive/src/job_router.rs`

- [ ] **WorkerList Operation**
  - [ ] Implement in job_router.rs
  - [ ] Call worker_registry.list()
  - [ ] Add narration events (start, result, entries)
  - [ ] Handle empty registry case
  - [ ] Format output as table

- [ ] **WorkerGet Operation**
  - [ ] Implement in job_router.rs
  - [ ] Call worker_registry.get()
  - [ ] Add narration events (start, details)
  - [ ] Handle not found error
  - [ ] Output JSON details

- [ ] **WorkerDelete Operation**
  - [ ] Implement in job_router.rs
  - [ ] Get worker from registry
  - [ ] Kill process (TODO: actual implementation)
  - [ ] Remove from registry
  - [ ] Add narration events (start, kill, complete)
  - [ ] Handle not found error

- [ ] **Verification**
  - [ ] `cargo check --bin rbee-hive` passes
  - [ ] Manual test: WorkerList works
  - [ ] Manual test: WorkerGet works
  - [ ] Manual test: WorkerDelete works

---

### Phase 7: Hive Job Router Integration (TEAM-273)

**Files:** `rbee-hive/src/job_router.rs`, `rbee-hive/src/main.rs`

- [ ] **State Verification**
  - [ ] JobState includes all fields:
    - [ ] registry (JobRegistry)
    - [ ] model_catalog (ModelCatalog)
    - [ ] model_provisioner (ModelProvisioner)
    - [ ] worker_registry (WorkerRegistry)
    - [ ] worker_spawner (WorkerSpawner)

- [ ] **Initialization Verification**
  - [ ] All state initialized in main.rs
  - [ ] All dependencies added to Cargo.toml
  - [ ] Proper Arc wrapping

- [ ] **Operation Verification**
  - [ ] WorkerSpawn - calls real function
  - [ ] WorkerList - calls real function
  - [ ] WorkerGet - calls real function
  - [ ] WorkerDelete - calls real function
  - [ ] ModelDownload - calls real function
  - [ ] ModelList - calls real function
  - [ ] ModelGet - calls real function
  - [ ] ModelDelete - calls real function

- [ ] **TODO Cleanup**
  - [ ] All TODO markers removed from job_router.rs
  - [ ] All operations implemented
  - [ ] No placeholder comments

- [ ] **Compilation**
  - [ ] `cargo clean`
  - [ ] `cargo build --bin rbee-hive` succeeds
  - [ ] No warnings
  - [ ] No errors

- [ ] **Smoke Tests**
  - [ ] rbee-hive starts successfully
  - [ ] /health endpoint responds
  - [ ] /capabilities endpoint responds
  - [ ] /v1/jobs endpoint accepts requests

---

### Phase 8: HTTP Mode Testing & Validation (TEAM-274)

**Deliverable:** Test report document

- [ ] **Model Operations Testing**
  - [ ] Test ModelList (empty catalog)
  - [ ] Test ModelDownload
  - [ ] Test ModelList (with models)
  - [ ] Test ModelGet
  - [ ] Test ModelDelete
  - [ ] Verify narration events for each

- [ ] **Worker Operations Testing**
  - [ ] Test WorkerList (empty registry)
  - [ ] Test WorkerSpawn (may fail without binary)
  - [ ] Test WorkerList (with workers)
  - [ ] Test WorkerGet
  - [ ] Test WorkerDelete
  - [ ] Verify narration events for each

- [ ] **Error Handling Testing**
  - [ ] Test getting non-existent model
  - [ ] Test getting non-existent worker
  - [ ] Test deleting non-existent model
  - [ ] Test deleting non-existent worker
  - [ ] Test duplicate model add
  - [ ] Test duplicate worker register
  - [ ] Verify error narration events

- [ ] **Performance Benchmarks**
  - [ ] Measure ModelList latency (100 iterations)
  - [ ] Measure WorkerList latency (100 iterations)
  - [ ] Measure ModelGet latency
  - [ ] Measure WorkerGet latency
  - [ ] Document baseline: ~1.1ms per operation

- [ ] **Known Limitations Documentation**
  - [ ] Worker binary missing
  - [ ] Model download stub (no HuggingFace)
  - [ ] Process cleanup incomplete
  - [ ] File deletion incomplete

- [ ] **Test Report**
  - [ ] Create TEAM_274_TEST_REPORT.md
  - [ ] Document all test results
  - [ ] Include performance baselines
  - [ ] List known limitations
  - [ ] Provide recommendations

---

### Phase 9: Mode 3 Implementation (TEAM-275)

**Files:** Multiple in queen-rbee

- [ ] **Dependencies**
  - [ ] Add optional dependencies to Cargo.toml
  - [ ] Add local-hive feature flag
  - [ ] Include all 3 rbee-hive crates

- [ ] **IntegratedHive Struct**
  - [ ] Create integrated_hive.rs module
  - [ ] IntegratedHive struct with all fields
  - [ ] new() constructor
  - [ ] Initialize all hive components

- [ ] **execute_integrated() Function**
  - [ ] Create in hive_forwarder.rs
  - [ ] Implement WorkerSpawn
  - [ ] Implement WorkerList
  - [ ] Implement WorkerGet
  - [ ] Implement WorkerDelete
  - [ ] Implement ModelDownload
  - [ ] Implement ModelList
  - [ ] Implement ModelGet
  - [ ] Implement ModelDelete
  - [ ] Add narration for all operations

- [ ] **Routing Updates**
  - [ ] Update forward_to_hive() signature
  - [ ] Add integrated_hive parameter
  - [ ] Implement mode detection
  - [ ] Route to execute_integrated() when appropriate
  - [ ] Fallback to HTTP mode

- [ ] **Initialization**
  - [ ] Initialize IntegratedHive in main.rs
  - [ ] Add to JobState
  - [ ] Handle feature flag conditionally

- [ ] **Testing**
  - [ ] Build with --features local-hive
  - [ ] Test all 8 operations in integrated mode
  - [ ] Verify narration events work
  - [ ] Verify SSE routing works

- [ ] **Performance Benchmarks**
  - [ ] Measure ModelList latency (integrated)
  - [ ] Measure WorkerList latency (integrated)
  - [ ] Compare to HTTP baseline
  - [ ] Verify 100x+ speedup

- [ ] **Documentation Updates**
  - [ ] Update QUEEN_TO_HIVE_COMMUNICATION_MODES.md
  - [ ] Update README.md with feature flag
  - [ ] Update CHANGELOG.md
  - [ ] Add build instructions

- [ ] **Verification**
  - [ ] `cargo build --features local-hive` succeeds
  - [ ] HTTP mode still works (no breaking changes)
  - [ ] Integrated mode works for localhost
  - [ ] Remote mode still uses HTTP
  - [ ] All tests passing

---

## 🎯 Summary Progress

### By Phase

- [ ] Phase 1: Model Catalog Types (TEAM-267)
- [ ] Phase 2: Model Catalog Operations (TEAM-268)
- [ ] Phase 3: Model Provisioner (TEAM-269)
- [ ] Phase 4: Worker Registry (TEAM-270)
- [ ] Phase 5: Worker Lifecycle - Spawn (TEAM-271)
- [ ] Phase 6: Worker Lifecycle - Management (TEAM-272)
- [ ] Phase 7: Hive Job Router Integration (TEAM-273)
- [ ] Phase 8: HTTP Testing & Validation (TEAM-274)
- [ ] Phase 9: Mode 3 Implementation (TEAM-275)

### By Crate

- [ ] rbee-hive-model-catalog (Phases 1-2)
- [ ] rbee-hive-model-provisioner (Phase 3)
- [ ] rbee-hive-worker-lifecycle (Phases 4-6)
- [ ] rbee-hive (Phases 2-3, 5-7)
- [ ] queen-rbee (Phase 9)

### By Operation

- [ ] ModelList
- [ ] ModelGet
- [ ] ModelDelete
- [ ] ModelDownload
- [ ] WorkerList
- [ ] WorkerGet
- [ ] WorkerDelete
- [ ] WorkerSpawn

---

## 📊 Effort Tracking

| Phase | Team | Estimated Hours | Actual Hours | Status |
|-------|------|----------------|--------------|--------|
| 1 | 267 | 20-24 | | ⬜ TODO |
| 2 | 268 | 16-20 | | ⬜ TODO |
| 3 | 269 | 24-32 | | ⬜ TODO |
| 4 | 270 | 20-24 | | ⬜ TODO |
| 5 | 271 | 32-40 | | ⬜ TODO |
| 6 | 272 | 24-32 | | ⬜ TODO |
| 7 | 273 | 16-20 | | ⬜ TODO |
| 8 | 274 | 16-24 | | ⬜ TODO |
| 9 | 275 | 30-58 | | ⬜ TODO |
| **Total** | | **198-274** | | |

---

## 🚨 Critical Path

```
Phase 1 → Phase 2 → Phase 3
                  ↓
Phase 4 → Phase 5 → Phase 6
                  ↓
Phase 7 → Phase 8 → Phase 9
```

**Blockers:**
- Phase 2 blocked by Phase 1
- Phase 3 blocked by Phase 2
- Phase 5 blocked by Phase 4
- Phase 6 blocked by Phase 5
- Phase 7 blocked by Phases 3 & 6
- Phase 8 blocked by Phase 7
- Phase 9 blocked by Phase 8

---

## ✅ Definition of Done

### Per Phase
- [ ] All checklist items complete
- [ ] Code compiles with no warnings
- [ ] Tests passing (if applicable)
- [ ] Narration events working
- [ ] Handoff document created
- [ ] Next team unblocked

### Overall (All Phases)
- [ ] All 8 operations working via HTTP
- [ ] Mode 3 working for localhost
- [ ] 100x+ speedup measured
- [ ] No breaking changes
- [ ] Full documentation
- [ ] Test report complete

---

**Use this checklist to track progress across all 9 phases!**
