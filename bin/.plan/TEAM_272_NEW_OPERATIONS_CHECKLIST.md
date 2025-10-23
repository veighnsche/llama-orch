# TEAM-272: New Operations Implementation Checklist

**Date:** Oct 23, 2025  
**Status:** 🚧 IN PROGRESS  
**Total Operations:** 28 (13 Hive + 15 Queen)

---

## 📋 Implementation Checklist

### ✅ COMPLETED (11 operations)

**TEAM-272:**
- [x] **WorkerSpawn** - Spawn worker process on hive
- [x] **ModelList** - List models on hive
- [x] **ModelGet** - Get model details on hive
- [x] **ModelDelete** - Delete model from hive
- [x] **Status** - Show live status (queen)

**TEAM-274:**
- [x] **WorkerBinaryList** - List worker binaries on hive
- [x] **WorkerBinaryGet** - Get worker binary details
- [x] **WorkerBinaryDelete** - Delete worker binary from hive
- [x] **WorkerProcessList** - List worker processes (local ps)
- [x] **WorkerProcessGet** - Get worker process details by PID
- [x] **WorkerProcessDelete** - Kill worker process by PID

---

## 🚧 HIVE OPERATIONS (2 remaining - TEAM-274 completed 6)

### Worker Binary Operations (2 remaining, 3 complete)

**✅ TEAM-274 Completed:**
- [x] WorkerBinaryList
- [x] WorkerBinaryGet  
- [x] WorkerBinaryDelete

**🚧 Remaining:**

#### [ ] WorkerDownload
**Priority:** HIGH  
**Estimated Effort:** 8-12 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `rbee-hive/src/job_router.rs` - Add handler
3. `rbee-keeper/src/main.rs` - Add CLI command
4. `rbee-hive-worker-catalog/` - Add download functionality

**Implementation Steps:**
- [ ] Add `WorkerDownload` handler in hive job_router
- [ ] Implement download logic in worker-catalog
- [ ] Add CLI command: `rbee worker download --hive <hive> --type <worker-type>`
- [ ] Add narration events
- [ ] Test download from GitHub releases or build artifacts
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Can download worker binary to hive
- [ ] Binary stored in worker catalog
- [ ] Compilation passes
- [ ] CLI command works

---

#### [ ] WorkerBuild
**Priority:** MEDIUM  
**Estimated Effort:** 12-16 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `rbee-hive/src/job_router.rs` - Add handler
3. `rbee-keeper/src/main.rs` - Add CLI command
4. `rbee-hive-worker-catalog/` - Add build functionality

**Implementation Steps:**
- [ ] Add `WorkerBuild` handler in hive job_router
- [ ] Implement build logic (cargo build --release)
- [ ] Add CLI command: `rbee worker build --hive <hive> --type <worker-type>`
- [ ] Handle build dependencies
- [ ] Stream build output via SSE
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Can build worker binary on hive
- [ ] Build output streamed to client
- [ ] Binary stored in worker catalog
- [ ] Compilation passes

---

#### [ ] WorkerBinaryList
**Priority:** HIGH  
**Estimated Effort:** 4-6 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `rbee-hive/src/job_router.rs` - Add handler
3. `rbee-keeper/src/main.rs` - Add CLI command
4. `rbee-hive-worker-catalog/` - Use existing list()

**Implementation Steps:**
- [ ] Add `WorkerBinaryList` handler in hive job_router
- [ ] Call worker_catalog.list()
- [ ] Format as table
- [ ] Add CLI command: `rbee worker binary list --hive <hive>`
- [ ] Add narration events
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Lists all worker binaries on hive
- [ ] Shows binary name, size, path
- [ ] Formatted as table
- [ ] Compilation passes

---

#### [ ] WorkerBinaryGet
**Priority:** MEDIUM  
**Estimated Effort:** 4-6 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `rbee-hive/src/job_router.rs` - Add handler
3. `rbee-keeper/src/main.rs` - Add CLI command
4. `rbee-hive-worker-catalog/` - Use existing get()

**Implementation Steps:**
- [ ] Add `WorkerBinaryGet` handler in hive job_router
- [ ] Call worker_catalog.get()
- [ ] Format as JSON
- [ ] Add CLI command: `rbee worker binary get --hive <hive> --type <type>`
- [ ] Add narration events
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Gets worker binary details
- [ ] Shows path, size, metadata
- [ ] Formatted as JSON
- [ ] Compilation passes

---

#### [ ] WorkerBinaryDelete
**Priority:** LOW  
**Estimated Effort:** 4-6 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `rbee-hive/src/job_router.rs` - Add handler
3. `rbee-keeper/src/main.rs` - Add CLI command
4. `rbee-hive-worker-catalog/` - Add delete functionality

**Implementation Steps:**
- [ ] Add `WorkerBinaryDelete` handler in hive job_router
- [ ] Implement delete logic in worker-catalog
- [ ] Add CLI command: `rbee worker binary delete --hive <hive> --type <type>`
- [ ] Add confirmation prompt
- [ ] Add narration events
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Can delete worker binary from hive
- [ ] Binary removed from catalog
- [ ] File deleted from disk
- [ ] Compilation passes

---

### Worker Process Operations (0 remaining, 3 complete)

**✅ TEAM-274 Completed:**
- [x] WorkerProcessList - List worker processes using local ps
- [x] WorkerProcessGet - Get worker process details by PID
- [x] WorkerProcessDelete - Kill worker process by PID

**All worker process operations complete! 🎉**

---

### ~~Worker Process Operations~~ (COMPLETE)

#### ~~[ ] WorkerProcessList~~ ✅ DONE
**Priority:** HIGH  
**Estimated Effort:** 6-8 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `rbee-hive/src/job_router.rs` - Add handler
3. `rbee-keeper/src/main.rs` - Add CLI command
4. `rbee-hive-worker-lifecycle/` - Add process listing

**Implementation Steps:**
- [ ] Add `WorkerProcessList` handler in hive job_router
- [ ] Implement local process listing (ps aux | grep worker)
- [ ] Parse process list
- [ ] Format as table (PID, worker_id, model, uptime)
- [ ] Add CLI command: `rbee worker process list --hive <hive>`
- [ ] Add narration events
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Lists all worker processes on hive (local ps)
- [ ] Shows PID, worker_id, model, uptime
- [ ] Formatted as table
- [ ] Compilation passes

---

#### [ ] WorkerProcessGet
**Priority:** MEDIUM  
**Estimated Effort:** 4-6 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `rbee-hive/src/job_router.rs` - Add handler
3. `rbee-keeper/src/main.rs` - Add CLI command
4. `rbee-hive-worker-lifecycle/` - Add process details

**Implementation Steps:**
- [ ] Add `WorkerProcessGet` handler in hive job_router
- [ ] Get process details by PID
- [ ] Format as JSON
- [ ] Add CLI command: `rbee worker process get --hive <hive> --pid <pid>`
- [ ] Add narration events
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Gets worker process details by PID
- [ ] Shows PID, command, memory, CPU
- [ ] Formatted as JSON
- [ ] Compilation passes

---

#### [ ] WorkerProcessDelete
**Priority:** HIGH  
**Estimated Effort:** 4-6 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `rbee-hive/src/job_router.rs` - Add handler
3. `rbee-keeper/src/main.rs` - Add CLI command
4. `rbee-hive-worker-lifecycle/` - ✅ DONE (delete_worker exists)

**Implementation Steps:**
- [ ] Add `WorkerProcessDelete` handler in hive job_router
- [ ] Call delete_worker(job_id, worker_id, pid)
- [ ] Add CLI command: `rbee worker process delete --hive <hive> --pid <pid>`
- [ ] Add confirmation prompt
- [ ] Add narration events
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Can kill worker process by PID
- [ ] Uses SIGTERM → SIGKILL pattern
- [ ] Confirmation prompt shown
- [ ] Compilation passes

---

### Model Operations (1 remaining)

#### [ ] ModelDownload
**Priority:** HIGH  
**Estimated Effort:** 16-24 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `rbee-hive/src/job_router.rs` - Update handler (stub exists)
3. `rbee-keeper/src/main.rs` - CLI command exists
4. `rbee-hive-model-provisioner/` - Implement download logic

**Implementation Steps:**
- [ ] Implement model provisioner download logic
- [ ] Add HuggingFace API integration
- [ ] Stream download progress via SSE
- [ ] Handle resume on failure
- [ ] Validate model after download
- [ ] Update job_router handler
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Can download models from HuggingFace
- [ ] Progress streamed to client
- [ ] Resume on failure
- [ ] Model validated after download
- [ ] Compilation passes

---

## 🚧 QUEEN OPERATIONS (10 remaining)

### Hive Management Operations (Already Implemented)

- [x] **HiveList** - ✅ DONE
- [x] **HiveGet** - ✅ DONE
- [x] **HiveInstall** - ✅ DONE
- [x] **HiveUninstall** - ✅ DONE
- [x] **HiveStart** - ✅ DONE
- [x] **HiveStop** - ✅ DONE
- [x] **HiveStatus** - ✅ DONE
- [x] **HiveRefreshCapabilities** - ✅ DONE
- [x] **SshTest** - ✅ DONE
- [x] **HiveImportSsh** - ✅ DONE

### Active Worker Operations (3 operations)

#### [ ] ActiveWorkerList
**Priority:** HIGH  
**Estimated Effort:** 12-16 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `queen-rbee/src/job_router.rs` - Update handler (stub exists)
3. `rbee-keeper/src/main.rs` - Add CLI command
4. `queen-rbee-worker-registry/` - Implement list_active()

**Implementation Steps:**
- [ ] Implement worker registry list_active()
- [ ] Update queen job_router handler
- [ ] Format as table (worker_id, hive_id, model, status, last_heartbeat)
- [ ] Add CLI command: `rbee worker list` (no --hive flag)
- [ ] Add narration events
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Lists all active workers across all hives
- [ ] Shows worker_id, hive_id, model, status
- [ ] Filtered by heartbeat (last 30s)
- [ ] Formatted as table
- [ ] Compilation passes

---

#### [ ] ActiveWorkerGet
**Priority:** MEDIUM  
**Estimated Effort:** 6-8 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `queen-rbee/src/job_router.rs` - Update handler (stub exists)
3. `rbee-keeper/src/main.rs` - Add CLI command
4. `queen-rbee-worker-registry/` - Implement get()

**Implementation Steps:**
- [ ] Implement worker registry get()
- [ ] Update queen job_router handler
- [ ] Format as JSON
- [ ] Add CLI command: `rbee worker get <worker-id>`
- [ ] Add narration events
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Gets active worker details by worker_id
- [ ] Shows full worker state (PID, port, model, hive, status)
- [ ] Formatted as JSON
- [ ] Compilation passes

---

#### [ ] ActiveWorkerRetire
**Priority:** LOW  
**Estimated Effort:** 8-12 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `queen-rbee/src/job_router.rs` - Update handler (stub exists)
3. `rbee-keeper/src/main.rs` - Add CLI command
4. `queen-rbee-worker-registry/` - Implement retire()

**Implementation Steps:**
- [ ] Implement worker registry retire()
- [ ] Mark worker as "retiring" (no new requests)
- [ ] Update queen job_router handler
- [ ] Add CLI command: `rbee worker retire <worker-id>`
- [ ] Add confirmation prompt
- [ ] Add narration events
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Can retire active worker
- [ ] Worker marked as "retiring" in registry
- [ ] No new inference requests routed to worker
- [ ] Existing requests complete
- [ ] Compilation passes

---

### Inference Operations (1 operation)

#### [ ] Infer
**Priority:** CRITICAL  
**Estimated Effort:** 40-60 hours  
**Files to Create/Modify:**
1. `rbee-operations/src/lib.rs` - ✅ DONE (already added)
2. `queen-rbee/src/job_router.rs` - Update handler (stub exists)
3. `rbee-keeper/src/main.rs` - CLI command exists
4. `queen-rbee/src/inference_scheduler.rs` - NEW (scheduler logic)
5. `queen-rbee-worker-registry/` - Query for available workers

**Implementation Steps:**
- [ ] Design inference scheduler
- [ ] Implement worker selection algorithm
  - [ ] Filter by model
  - [ ] Filter by device (if specified)
  - [ ] Filter by worker_id (if specified)
  - [ ] Load balancing (round-robin or least-loaded)
- [ ] Implement direct HTTP to worker
- [ ] Stream tokens back via SSE
- [ ] Handle worker failures (retry on different worker)
- [ ] Update queen job_router handler
- [ ] Add narration events
- [ ] Document in handoff

**Acceptance Criteria:**
- [ ] Can route inference to active workers
- [ ] Scheduler selects best worker
- [ ] Tokens streamed back to client
- [ ] Handles worker failures gracefully
- [ ] Compilation passes
- [ ] End-to-end inference works

---

## 📊 Progress Summary

### Overall Progress
- **Total Operations:** 28
- **Completed:** 11 (39%) ⬆️ +6 from TEAM-274
- **Remaining:** 17 (61%)

### By Component
**Hive Operations:**
- Completed: 10/13 (77%) ⬆️ +6 from TEAM-274
- Remaining: 3/13 (23%)

**Queen Operations:**
- Completed: 11/15 (73%)
- Remaining: 4/15 (27%)

### By Priority
**CRITICAL:** 1 operation (Infer - deferred)  
**HIGH:** 3 operations (WorkerDownload, ModelDownload, ActiveWorkerList) - ✅ 3 complete (WorkerProcessList, WorkerProcessDelete)  
**MEDIUM:** 1 operation (WorkerBuild, ActiveWorkerGet) - ✅ 2 complete (WorkerBinaryGet, WorkerProcessGet)  
**LOW:** 1 operation (ActiveWorkerRetire) - ✅ 1 complete (WorkerBinaryDelete)

---

## 🎯 Recommended Implementation Order

### ~~Phase 1: Worker Binary Management~~ (PARTIALLY COMPLETE - TEAM-274)
1. ✅ WorkerBinaryList (4-6h) - TEAM-274
2. [ ] WorkerDownload (8-12h) - **NEXT PRIORITY**
3. ✅ WorkerBinaryGet (4-6h) - TEAM-274
4. ✅ WorkerBinaryDelete (4-6h) - TEAM-274
5. [ ] WorkerBuild (12-16h)

**Total:** 32-46 hours  
**Completed:** 12-18 hours (TEAM-274)  
**Remaining:** 20-28 hours

### ~~Phase 2: Worker Process Management~~ ✅ COMPLETE (TEAM-274)
1. ✅ WorkerProcessList (6-8h) - TEAM-274
2. ✅ WorkerProcessGet (4-6h) - TEAM-274
3. ✅ WorkerProcessDelete (4-6h) - TEAM-274

**Total:** 14-20 hours ✅ ALL DONE

### Phase 3: Active Worker Tracking (Week 3)
1. ActiveWorkerList (12-16h)
2. ActiveWorkerGet (6-8h)
3. ActiveWorkerRetire (8-12h)

**Total:** 26-36 hours

### Phase 4: Model & Inference (Weeks 4-5)
1. ModelDownload (16-24h)
2. Infer (40-60h)

**Total:** 56-84 hours

---

## 🔑 Key Dependencies

### Worker Registry (CRITICAL)
**Required for:** ActiveWorkerList, ActiveWorkerGet, ActiveWorkerRetire, Infer  
**Estimated Effort:** 20-30 hours  
**Priority:** CRITICAL

**Implementation:**
- [ ] Create worker-registry crate
- [ ] Implement heartbeat tracking
- [ ] Implement list_active(), get(), retire()
- [ ] Store worker state (PID, port, model, hive, status)
- [ ] Clean up stale workers (no heartbeat > 30s)

### Model Provisioner (HIGH)
**Required for:** ModelDownload  
**Estimated Effort:** 16-24 hours  
**Priority:** HIGH

**Implementation:**
- [ ] Create model-provisioner crate
- [ ] Implement HuggingFace API integration
- [ ] Implement download with progress
- [ ] Implement resume on failure
- [ ] Implement validation

### Inference Scheduler (CRITICAL)
**Required for:** Infer  
**Estimated Effort:** 40-60 hours  
**Priority:** CRITICAL

**Implementation:**
- [ ] Design scheduler algorithm
- [ ] Implement worker selection
- [ ] Implement direct HTTP to worker
- [ ] Implement token streaming
- [ ] Implement failure handling

---

## 📝 Notes

### Architecture Reminders
1. **Hive is stateless** - Only executes local operations
2. **Queen is stateful** - Tracks workers via heartbeats
3. **Workers send heartbeats to queen** - Not to hive
4. **Inference goes through queen** - Queen selects worker

### Testing Strategy
- Unit tests for each operation
- Integration tests for end-to-end flows
- Manual testing with real workers
- Performance testing for inference

### Documentation Requirements
- Update ADDING_NEW_OPERATIONS.md for each new operation
- Create handoff documents for each phase
- Update architecture diagrams
- Document CLI commands

---

## ✅ Completion Criteria

**Phase 1 Complete When:**
- [ ] All worker binary operations implemented
- [ ] CLI commands working
- [ ] Tests passing
- [ ] Documentation updated

**Phase 2 Complete When:**
- [ ] All worker process operations implemented
- [ ] CLI commands working
- [ ] Tests passing
- [ ] Documentation updated

**Phase 3 Complete When:**
- [ ] Worker registry implemented
- [ ] All active worker operations implemented
- [ ] CLI commands working
- [ ] Tests passing
- [ ] Documentation updated

**Phase 4 Complete When:**
- [ ] Model provisioner implemented
- [ ] Inference scheduler implemented
- [ ] End-to-end inference working
- [ ] CLI commands working
- [ ] Tests passing
- [ ] Documentation updated

**ALL OPERATIONS COMPLETE WHEN:**
- [ ] All 28 operations implemented
- [ ] All CLI commands working
- [ ] All tests passing
- [ ] All documentation updated
- [ ] End-to-end workflows tested
- [ ] Performance benchmarks met

---

**Total Estimated Effort:** 128-186 hours (3-4 weeks full-time)

**TEAM-272 new operations checklist complete! Ready for implementation! 🚀**
