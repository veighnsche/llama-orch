# START HERE: rbee-hive Implementation Plan (TEAM-267 to TEAM-275)

**Created by:** TEAM-266  
**Date:** Oct 23, 2025  
**Status:** 🟡 IN PROGRESS (Phases 1-2 complete)  
**Prerequisite:** Mode 3 (Integrated) implementation

---

## 🎯 Mission Overview

Implement all rbee-hive crates to enable worker and model management functionality. This is a **PREREQUISITE** for Mode 3 (Integrated) implementation.

**Current Status:** All rbee-hive crates are empty stubs (13-16 lines each)  
**Target:** Fully functional worker and model management via HTTP  
**Estimated Effort:** 180-220 hours (4-6 weeks)  
**Teams Required:** 9 teams (TEAM-267 to TEAM-275)

---

## 📋 Quick Reference

| Phase | Team | Focus | Effort | Status |
|-------|------|-------|--------|--------|
| **Phase 1** | 267 | Model Catalog Types & Storage | 20-24h | ✅ COMPLETE |
| **Phase 2** | 268 | Model Catalog Operations | 16-20h | ✅ COMPLETE |
| **Phase 3** | 269 | Model Provisioner Core | 24-32h | 🔴 TODO |
| **Phase 4** | 270 | Worker Registry & Types | 20-24h | 🔴 TODO |
| **Phase 5** | 271 | Worker Lifecycle - Spawn | 32-40h | 🔴 TODO |
| **Phase 6** | 272 | Worker Lifecycle - Management | 24-32h | 🔴 TODO |
| **Phase 7** | 273 | Hive Job Router Integration | 16-20h | 🔴 TODO |
| **Phase 8** | 274 | HTTP Mode Testing & Validation | 16-24h | 🔴 TODO |
| **Phase 9** | 275 | Mode 3 Implementation | 30-58h | 🔴 TODO |

**Total:** 198-274 hours

---

## 🎯 Success Criteria

### Phase 1-7: HTTP Mode Working
- ✅ All 8 operations work via HTTP (POST /v1/jobs)
- ✅ WorkerSpawn creates actual worker processes
- ✅ WorkerList returns running workers
- ✅ ModelDownload fetches models from HuggingFace
- ✅ ModelList shows available models
- ✅ All operations emit proper narration events
- ✅ Integration tests passing

### Phase 8: Validation
- ✅ End-to-end tests for all operations
- ✅ Performance benchmarks established
- ✅ Error handling verified
- ✅ Documentation complete

### Phase 9: Mode 3
- ✅ Integrated mode working for localhost
- ✅ 110x speedup for list/get operations
- ✅ No breaking changes to HTTP mode
- ✅ Feature flag working correctly

---

## 🚀 Getting Started

### For TEAM-267 (First Team)

1. **Read these documents:**
   - This file (START_HERE)
   - `STORAGE_ARCHITECTURE.md` ← **IMPORTANT: Read this first!**
   - `TEAM_267_MODEL_CATALOG_TYPES.md`
   - `TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md` (context)

2. **Understand the architecture:**
   - rbee-hive receives operations via POST /v1/jobs
   - Operations are defined in `rbee-operations/src/lib.rs`
   - Job router dispatches to handlers in `rbee-hive/src/job_router.rs`
   - Your crates provide the actual implementation

3. **Start with Phase 1:**
   - Implement model catalog types
   - Create storage layer (in-memory + optional SQLite)
   - Write unit tests
   - Document public API

4. **Hand off to TEAM-268:**
   - Complete checklist in your guide
   - Document what works
   - Note any blockers for next team

### For Subsequent Teams

1. Read your team's guide (TEAM_XXX_*.md)
2. Read previous team's handoff
3. Verify previous team's work compiles
4. Complete your phase
5. Hand off to next team

---

## 📁 File Structure

```
bin/.plan/
├── START_HERE_267_TO_275_RBEE_HIVE_IMPLEMENTATION.md  ← You are here
├── TEAM_267_MODEL_CATALOG_TYPES.md                    ← Phase 1
├── TEAM_268_MODEL_CATALOG_OPERATIONS.md               ← Phase 2
├── TEAM_269_MODEL_PROVISIONER.md                      ← Phase 3
├── TEAM_270_WORKER_REGISTRY.md                        ← Phase 4
├── TEAM_271_WORKER_LIFECYCLE_SPAWN.md                 ← Phase 5
├── TEAM_272_WORKER_LIFECYCLE_MGMT.md                  ← Phase 6
├── TEAM_273_HIVE_JOB_ROUTER.md                        ← Phase 7
├── TEAM_274_HTTP_TESTING.md                           ← Phase 8
└── TEAM_275_MODE_3_IMPLEMENTATION.md                  ← Phase 9
```

---

## 🏗️ Architecture Overview

### Current State (HTTP Only)

```text
rbee-keeper (CLI)
  ↓ POST /v1/jobs
queen-rbee (Scheduler)
  ↓ hive_forwarder::forward_to_hive()
  ↓ POST http://localhost:8600/v1/jobs
rbee-hive (Worker Manager)
  ↓ job_router::route_operation()
  ↓ match operation
  ↓ ⚠️ TODO STUBS (your work!)
  ↓ rbee-hive crates (worker-lifecycle, model-catalog, etc.)
```

### Target State (After Phase 9)

```text
rbee-keeper (CLI)
  ↓ POST /v1/jobs
queen-rbee (Scheduler)
  ↓ hive_forwarder::forward_to_hive()
  ↓ Mode detection
  ├─ Remote: HTTP to remote hive
  ├─ Localhost HTTP: POST http://localhost:8600/v1/jobs
  └─ Integrated: Direct function calls (110x faster!)
     ↓ execute_integrated()
     ↓ rbee-hive crates (in-process)
```

---

## 📊 Dependencies Between Phases

```text
Phase 1 (Model Catalog Types)
  ↓
Phase 2 (Model Catalog Operations)
  ↓
Phase 3 (Model Provisioner) ──┐
  ↓                            │
Phase 4 (Worker Registry)      │
  ↓                            │
Phase 5 (Worker Spawn) ────────┤
  ↓                            │
Phase 6 (Worker Management)    │
  ↓                            │
Phase 7 (Job Router) ←─────────┘
  ↓
Phase 8 (Testing)
  ↓
Phase 9 (Mode 3)
```

**Critical Path:** Phases 1-7 must be completed in order. Phase 8 validates everything. Phase 9 adds Mode 3.

---

## 🎓 Key Concepts

### 1. Operation Flow

Every operation follows this pattern:

```rust
// 1. Client submits operation
POST /v1/jobs
Body: {"operation": "worker_spawn", "hive_id": "localhost", ...}

// 2. Job created, SSE stream URL returned
Response: {"job_id": "xyz", "sse_url": "/v1/jobs/xyz/stream"}

// 3. Client connects to SSE stream
GET /v1/jobs/xyz/stream

// 4. Operation executes, events stream back
data: [hv-router ] route_job       : Executing operation: worker_spawn
data: [worker-lc ] spawn_start     : Spawning worker...
data: [worker-lc ] spawn_complete  : Worker spawned: worker-123
data: [DONE]

// 5. Client receives all events in real-time
```

### 2. Narration Pattern

All operations MUST emit narration events:

```rust
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("worker-lc");

NARRATE
    .action("spawn_start")
    .job_id(&job_id)  // ← CRITICAL for SSE routing
    .context(&worker_id)
    .human("Spawning worker '{}'")
    .emit();
```

### 3. State Management

All state uses Arc<Mutex<>> pattern:

```rust
pub struct WorkerRegistry {
    workers: Arc<Mutex<HashMap<String, Worker>>>,
}

impl WorkerRegistry {
    pub fn register(&self, worker: Worker) -> Result<()> {
        let mut workers = self.workers.lock().unwrap();
        workers.insert(worker.id.clone(), worker);
        Ok(())
    }
}
```

### 4. Error Handling

Errors are converted to narration events:

```rust
match spawn_worker(...).await {
    Ok(worker_id) => {
        NARRATE
            .action("spawn_complete")
            .job_id(&job_id)
            .context(&worker_id)
            .human("✅ Worker spawned: {}")
            .emit();
    }
    Err(e) => {
        NARRATE
            .action("spawn_error")
            .job_id(&job_id)
            .human("❌ Failed to spawn worker: {}")
            .emit();
        return Err(e);
    }
}
```

---

## 🛠️ Development Workflow

### For Each Phase

1. **Read your guide** (TEAM_XXX_*.md)
2. **Check previous team's work:**
   ```bash
   cargo check --bin rbee-hive
   cargo test --package <previous-crate>
   ```
3. **Implement your phase:**
   - Write types/structs
   - Implement functions
   - Add narration events
   - Write unit tests
4. **Verify compilation:**
   ```bash
   cargo check --bin rbee-hive
   cargo test --package <your-crate>
   ```
5. **Update job_router.rs** (if applicable)
6. **Document your work:**
   - Create TEAM_XXX_HANDOFF.md
   - List what works
   - Note any issues
7. **Hand off to next team**

### Testing Commands

```bash
# Check compilation
cargo check --bin rbee-hive

# Run unit tests for a crate
cargo test --package rbee-hive-model-catalog

# Run integration tests
cargo test --bin rbee-hive

# Check all hive crates
cargo check --package rbee-hive-model-catalog
cargo check --package rbee-hive-model-provisioner
cargo check --package rbee-hive-worker-lifecycle
```

---

## 📚 Required Reading

### Essential Documents

1. **STORAGE_ARCHITECTURE.md** ← **READ THIS FIRST!**
   - Filesystem-based catalog design
   - Cross-platform directory structure
   - Metadata YAML format
   - Models and worker binary locations

2. **TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md**
   - Comprehensive investigation of Mode 3
   - Operation mapping table
   - Architecture analysis

3. **QUEEN_TO_HIVE_COMMUNICATION_MODES.md**
   - Three communication modes
   - Mode selection logic
   - Performance characteristics

3. **bin/20_rbee_hive/src/job_router.rs**
   - Current TODO stubs
   - Operation routing
   - Job lifecycle

4. **bin/99_shared_crates/rbee-operations/src/lib.rs**
   - Operation enum definition
   - should_forward_to_hive() logic

### Reference Implementations

- **device-detection crate:** ✅ Fully implemented (use as reference)
- **daemon-lifecycle crate:** ✅ Process spawning patterns
- **job-server crate:** ✅ Job registry patterns
- **narration-core crate:** ✅ Narration patterns

---

## ⚠️ Common Pitfalls

### 1. Forgetting job_id in Narration

❌ **WRONG:**
```rust
NARRATE
    .action("spawn_start")
    .human("Spawning worker")
    .emit();
```

✅ **CORRECT:**
```rust
NARRATE
    .action("spawn_start")
    .job_id(&job_id)  // ← MUST include for SSE routing
    .human("Spawning worker")
    .emit();
```

### 2. Not Using Arc<Mutex<>> for Shared State

❌ **WRONG:**
```rust
pub struct WorkerRegistry {
    workers: HashMap<String, Worker>,  // ← Not thread-safe!
}
```

✅ **CORRECT:**
```rust
pub struct WorkerRegistry {
    workers: Arc<Mutex<HashMap<String, Worker>>>,
}
```

### 3. Hardcoding Paths

❌ **WRONG:**
```rust
let model_path = "/home/user/.cache/rbee/models";
```

✅ **CORRECT:**
```rust
let model_path = dirs::cache_dir()
    .ok_or_else(|| anyhow!("Cannot determine cache directory"))?
    .join("rbee")
    .join("models");
```

### 4. Not Handling Async Properly

❌ **WRONG:**
```rust
pub fn spawn_worker(...) -> Result<String> {
    // Blocking I/O in async context
    std::fs::create_dir_all(path)?;
}
```

✅ **CORRECT:**
```rust
pub async fn spawn_worker(...) -> Result<String> {
    // Non-blocking I/O
    tokio::fs::create_dir_all(path).await?;
}
```

---

## 🎯 Phase Summaries

### Phase 1: Model Catalog Types (TEAM-267)
**Goal:** Define model storage types and in-memory catalog  
**Deliverables:** ModelEntry, ModelCatalog struct, basic CRUD  
**Effort:** 20-24 hours

### Phase 2: Model Catalog Operations (TEAM-268)
**Goal:** Implement list/get/delete operations  
**Deliverables:** Full CRUD with narration, integration with job_router  
**Effort:** 16-20 hours

### Phase 3: Model Provisioner (TEAM-269)
**Goal:** Download models from HuggingFace  
**Deliverables:** download_model() function, progress tracking  
**Effort:** 24-32 hours

### Phase 4: Worker Registry (TEAM-270)
**Goal:** Worker state management  
**Deliverables:** WorkerRegistry, Worker types, CRUD operations  
**Effort:** 20-24 hours

### Phase 5: Worker Lifecycle - Spawn (TEAM-271)
**Goal:** Spawn worker processes  
**Deliverables:** spawn_worker() function, process management  
**Effort:** 32-40 hours (most complex phase)

### Phase 6: Worker Lifecycle - Management (TEAM-272)
**Goal:** List/get/delete workers  
**Deliverables:** Full worker CRUD operations  
**Effort:** 24-32 hours

### Phase 7: Hive Job Router Integration (TEAM-273)
**Goal:** Wire up all operations in job_router.rs  
**Deliverables:** Replace all TODO stubs with real calls  
**Effort:** 16-20 hours

### Phase 8: HTTP Testing (TEAM-274)
**Goal:** Validate all operations work via HTTP  
**Deliverables:** Integration tests, performance baselines  
**Effort:** 16-24 hours

### Phase 9: Mode 3 Implementation (TEAM-275)
**Goal:** Add integrated mode for localhost  
**Deliverables:** execute_integrated(), 110x speedup  
**Effort:** 30-58 hours

---

## 📈 Progress Tracking

### Checklist for Each Phase

- [ ] Read team guide
- [ ] Read previous team's handoff
- [ ] Verify previous work compiles
- [ ] Implement required functionality
- [ ] Add narration events
- [ ] Write unit tests
- [ ] Update job_router.rs (if applicable)
- [ ] Verify compilation
- [ ] Create handoff document
- [ ] Mark phase complete

### Overall Progress

```
Phase 1: [✅] Model Catalog Types (TEAM-267 complete)
Phase 2: [✅] Model Catalog Operations (TEAM-268 complete)
Phase 3: [ ] Model Provisioner
Phase 4: [ ] Worker Registry
Phase 5: [ ] Worker Lifecycle - Spawn
Phase 6: [ ] Worker Lifecycle - Management
Phase 7: [ ] Hive Job Router Integration
Phase 8: [ ] HTTP Testing
Phase 9: [ ] Mode 3 Implementation
```

---

## 🎓 Learning Resources

### Rust Patterns
- **Arc<Mutex<>>:** Shared mutable state
- **async/await:** Non-blocking I/O
- **Result<T, E>:** Error handling
- **tokio::spawn:** Background tasks

### Project Patterns
- **Narration:** observability_narration_core
- **Job lifecycle:** job-server crate
- **Process spawning:** daemon-lifecycle crate
- **Config management:** rbee-config crate

### External Resources
- **HuggingFace Hub API:** https://huggingface.co/docs/hub/api
- **tokio docs:** https://tokio.rs/
- **anyhow error handling:** https://docs.rs/anyhow/

---

## 🚨 Blockers & Escalation

### Known Blockers

1. **Pre-existing compilation error** in queen-rbee-worker-registry
   - Error: Missing `HiveRegistry` type
   - Impact: queen-rbee doesn't compile
   - Workaround: Focus on rbee-hive crates first

2. **No worker binary available**
   - Impact: Cannot actually spawn workers yet
   - Workaround: Use stub/mock for Phase 5, document requirement

### Escalation Path

If blocked:
1. Document the blocker in your handoff
2. Note what you completed despite blocker
3. Suggest workarounds for next team
4. Continue with what you can do

---

## 🎉 Success Metrics

### Phase 1-7 Complete
- ✅ All 8 operations work via HTTP
- ✅ `cargo check --bin rbee-hive` passes
- ✅ Unit tests passing for all crates
- ✅ Narration events flowing correctly

### Phase 8 Complete
- ✅ Integration tests passing
- ✅ Performance baselines established
- ✅ End-to-end workflows verified

### Phase 9 Complete
- ✅ Mode 3 working for localhost
- ✅ 110x speedup measured
- ✅ Feature flag working
- ✅ No breaking changes

---

## 📞 Questions?

Read these documents in order:
1. This file (START_HERE)
2. Your team's guide (TEAM_XXX_*.md)
3. TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md
4. Previous team's handoff

Still stuck? Document your question in your handoff for the next team.

---

**TEAM-266 signing off. Good luck, teams 267-275! You've got this! 🐝**
