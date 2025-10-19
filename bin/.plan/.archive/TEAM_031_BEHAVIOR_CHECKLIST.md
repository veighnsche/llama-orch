# TEAM-031 Behavior Checklist - Architecture Verification

**Date:** 2025-10-10T00:54:00+02:00  
**Team:** TEAM-031  
**Mission:** Verify ALL behaviors after TEAM-030's architecture redesign

---

## ⚠️ CRITICAL: Read These First

**Before starting ANY work:**
1. ✅ Read `.windsurf/rules/dev-bee-rules.md` - Development standards
2. ✅ Read `bin/.plan/TEAM_030_FINAL_SUMMARY.md` - What changed
3. ✅ Read `bin/.specs/ARCHITECTURE_MODES.md` - Architecture overview
4. ✅ Read this entire checklist

**Dev-Bee Rules Summary:**
- ❌ NO background jobs (`cargo test &`)
- ✅ Blocking output only (`cargo test -- --nocapture`)
- ❌ NO multiple .md files for one task
- ✅ Add TEAM-031 signatures to all changes
- ✅ Complete ALL priorities in order (don't skip!)

---

## Architecture to Verify

### Worker Registry (In-Memory)
- **Storage:** `Arc<RwLock<HashMap<String, WorkerInfo>>>`
- **Location:** rbee-hive process memory
- **Persistence:** None (ephemeral)
- **Database:** None (no SQLite)

### Model Catalog (SQLite)
- **Storage:** SQLite database
- **Location:** `~/.rbee/models.db`
- **Persistence:** Survives restarts
- **Database:** SQLite (persistent)

---

## Behavior Checklist

### Category 1: Build & Compilation ✅

- [ ] **B1.1:** Workspace compiles without errors
  ```bash
  cargo check --workspace
  # Expected: Finished `dev` profile
  ```

- [ ] **B1.2:** rbee-hive builds successfully
  ```bash
  cargo build -p rbee-hive
  # Expected: Finished, binary at target/debug/rbee-hive
  ```

- [ ] **B1.3:** rbee-keeper builds successfully
  ```bash
  cargo build -p rbee-keeper
  # Expected: Finished, binary at target/debug/rbee
  ```

- [ ] **B1.4:** queen-rbee builds successfully
  ```bash
  cargo build -p queen-rbee
  # Expected: Finished, binary at target/debug/queen-rbee
  ```

- [ ] **B1.5:** No SQLite dependencies in worker registry code
  ```bash
  grep -r "sqlx" bin/rbee-hive/src/ || echo "No SQLite found - CORRECT"
  # Expected: No SQLite imports (except in model catalog usage)
  ```

---

### Category 2: Unit Tests ✅

- [ ] **B2.1:** All rbee-hive tests pass
  ```bash
  cargo test -p rbee-hive -- --nocapture
  # Expected: 11 passed; 0 failed
  ```

- [ ] **B2.2:** Worker registry tests pass (in-memory)
  ```bash
  cargo test -p rbee-hive registry -- --nocapture
  # Expected: All registry tests pass
  ```

- [ ] **B2.3:** Model provisioner tests pass
  ```bash
  cargo test -p rbee-hive provisioner -- --nocapture
  # Expected: Model name extraction tests pass
  ```

- [ ] **B2.4:** HTTP routes tests pass
  ```bash
  cargo test -p rbee-hive routes -- --nocapture
  # Expected: Router creation test passes
  ```

---

### Category 3: Worker Registry Behavior (In-Memory)

- [ ] **B3.1:** Worker registry initializes without database file
  ```bash
  # Start rbee-hive, check no workers.db created
  ./target/debug/rbee-hive daemon &
  sleep 2
  ls ~/.rbee/workers.db 2>/dev/null && echo "FAIL: workers.db exists" || echo "PASS: No workers.db"
  pkill rbee-hive
  ```

- [ ] **B3.2:** Workers can be registered in-memory
  ```bash
  # Test via HTTP API
  curl -X POST http://localhost:8080/v1/workers/ready \
    -H "Content-Type: application/json" \
    -d '{"worker_id":"test-123","url":"http://localhost:8081","model_ref":"test","backend":"cpu","device":0}'
  # Expected: 200 OK
  ```

- [ ] **B3.3:** Workers can be listed from in-memory registry
  ```bash
  curl http://localhost:8080/v1/workers/list | jq .
  # Expected: JSON array of workers
  ```

- [ ] **B3.4:** Worker state is lost on rbee-hive restart
  ```bash
  # 1. Register worker
  # 2. Restart rbee-hive
  # 3. List workers
  # Expected: Empty list (ephemeral)
  ```

- [ ] **B3.5:** No worker registry database file exists
  ```bash
  find ~/.rbee -name "workers.db" 2>/dev/null
  # Expected: No results
  ```

---

### Category 4: Model Catalog Behavior (SQLite)

- [ ] **B4.1:** Model catalog database is created
  ```bash
  ./target/debug/rbee-hive daemon &
  sleep 2
  ls ~/.rbee/models.db
  # Expected: File exists
  pkill rbee-hive
  ```

- [ ] **B4.2:** Model catalog persists across restarts
  ```bash
  # 1. Start rbee-hive (creates models.db)
  # 2. Stop rbee-hive
  # 3. Start rbee-hive again
  # 4. Check models.db still exists
  # Expected: Database file persists
  ```

- [ ] **B4.3:** Model catalog can track downloads
  ```bash
  # Check SQLite schema
  sqlite3 ~/.rbee/models.db ".schema models"
  # Expected: Table with reference, provider, local_path, size_bytes, downloaded_at
  ```

- [ ] **B4.4:** Model catalog prevents re-downloads
  ```bash
  # 1. Download model (or mock entry)
  # 2. Request same model again
  # Expected: Uses cached path, no re-download
  ```

---

### Category 5: Daemon Lifecycle

- [ ] **B5.1:** rbee-hive starts successfully
  ```bash
  ./target/debug/rbee-hive daemon > /tmp/rbee-hive.log 2>&1 &
  DAEMON_PID=$!
  sleep 3
  ps -p $DAEMON_PID
  # Expected: Process running
  kill $DAEMON_PID
  ```

- [ ] **B5.2:** Health endpoint responds
  ```bash
  curl http://localhost:8080/v1/health | jq .
  # Expected: {"status":"alive","version":"0.1.0","api_version":"v1"}
  ```

- [ ] **B5.3:** Daemon logs show in-memory registry init
  ```bash
  grep "Worker registry initialized (in-memory, ephemeral)" /tmp/rbee-hive.log
  # Expected: Log line found
  ```

- [ ] **B5.4:** Daemon logs show model catalog init
  ```bash
  grep "Model catalog initialized (SQLite, persistent)" /tmp/rbee-hive.log
  # Expected: Log line found
  ```

- [ ] **B5.5:** Graceful shutdown kills workers
  ```bash
  # 1. Start daemon
  # 2. Spawn worker (if possible)
  # 3. Send SIGTERM to daemon
  # 4. Check worker is killed
  # Expected: Worker process terminated
  ```

---

### Category 6: Worker Spawn Flow

- [ ] **B6.1:** Worker spawn checks model catalog
  ```bash
  # Monitor logs during spawn
  # Expected: "Checking model catalog for hf/..."
  ```

- [ ] **B6.2:** Worker spawn downloads missing model
  ```bash
  # Request model not in catalog
  # Expected: Download initiated, catalog updated
  ```

- [ ] **B6.3:** Worker spawn uses cached model
  ```bash
  # Request model already in catalog
  # Expected: "Model found in catalog: /path/to/model"
  ```

- [ ] **B6.4:** Worker is registered in in-memory registry
  ```bash
  # After spawn, check /v1/workers/list
  # Expected: Worker appears in list
  ```

- [ ] **B6.5:** Worker spawn returns correct response
  ```bash
  # Check spawn response structure
  # Expected: {worker_id, url, state: "loading"}
  ```

---

### Category 7: Ephemeral Mode (rbee-keeper)

- [ ] **B7.1:** rbee-keeper skips Phase 1 (no local registry)
  ```bash
  ./target/debug/rbee infer --node localhost --model test --prompt "hello" --max-tokens 5
  # Expected: "[Phase 1] Skipped (ephemeral mode)"
  ```

- [ ] **B7.2:** rbee-keeper connects to pool manager
  ```bash
  # Check Phase 2 in output
  # Expected: "Pool health: healthy"
  ```

- [ ] **B7.3:** No workers.db created by rbee-keeper
  ```bash
  # After running infer command
  ls ~/.rbee/workers.db 2>/dev/null && echo "FAIL" || echo "PASS"
  # Expected: PASS (no file)
  ```

---

### Category 8: Cascading Shutdown

- [ ] **B8.1:** SIGTERM handler is registered
  ```bash
  # Check daemon.rs code
  grep "tokio::signal::ctrl_c" bin/rbee-hive/src/commands/daemon.rs
  # Expected: Handler found
  ```

- [ ] **B8.2:** Shutdown sends HTTP to workers
  ```bash
  # Check shutdown_worker function
  grep "POST.*v1/shutdown" bin/rbee-hive/src/commands/daemon.rs
  # Expected: POST /v1/shutdown endpoint
  ```

- [ ] **B8.3:** Registry is cleared on shutdown
  ```bash
  # Check shutdown_all_workers function
  grep "registry.clear" bin/rbee-hive/src/commands/daemon.rs
  # Expected: clear() called
  ```

---

### Category 9: File System State

- [ ] **B9.1:** No workers.db file anywhere
  ```bash
  find ~ -name "workers.db" 2>/dev/null | grep -v reference
  # Expected: No results
  ```

- [ ] **B9.2:** models.db exists in correct location
  ```bash
  ls ~/.rbee/models.db
  # Expected: File exists
  ```

- [ ] **B9.3:** No orphaned database files
  ```bash
  find ~/.rbee -name "*.db" 2>/dev/null
  # Expected: Only models.db
  ```

---

### Category 10: Documentation Accuracy

- [ ] **B10.1:** ARCHITECTURE_MODES.md is accurate
  ```bash
  grep "Worker registry.*In-memory" bin/.specs/ARCHITECTURE_MODES.md
  grep "Model catalog.*SQLite" bin/.specs/ARCHITECTURE_MODES.md
  # Expected: Both found
  ```

- [ ] **B10.2:** test-001-mvp.md reflects architecture
  ```bash
  grep "TEAM-030.*in-memory" bin/.specs/.gherkin/test-001-mvp.md
  # Expected: Architecture documented
  ```

- [ ] **B10.3:** Test scripts have architecture headers
  ```bash
  grep "Worker registry.*In-memory" bin/.specs/.gherkin/test-001-mvp-*.sh
  # Expected: All scripts updated
  ```

---

### Category 11: E2E Test (BLOCKED - Model File Needed)

- [ ] **B11.1:** Preflight check passes
  ```bash
  ./bin/.specs/.gherkin/test-001-mvp-preflight.sh
  # Expected: All checks pass (except model file)
  ```

- [ ] **B11.2:** Download test model
  ```bash
  cd bin/llm-worker-rbee && ./download_test_model.sh
  # Expected: Model downloaded to .test-models/
  ```

- [ ] **B11.3:** Local E2E test passes
  ```bash
  ./bin/.specs/.gherkin/test-001-mvp-local.sh
  # Expected: All 8 phases complete
  ```

- [ ] **B11.4:** No lingering processes after test
  ```bash
  ps aux | grep -E "(rbee-hive|llm-worker-rbee)" | grep -v grep
  # Expected: No processes
  ```

- [ ] **B11.5:** VRAM is freed after test
  ```bash
  nvidia-smi  # or metal activity monitor
  # Expected: No model loaded
  ```

---

## Testing Procedure

### Phase 1: Static Verification (30 min)
1. Read all documentation
2. Check code signatures (TEAM-030)
3. Verify no SQLite in worker registry
4. Verify SQLite in model catalog
5. Review all modified files

### Phase 2: Build & Unit Tests (15 min)
1. Run all build checks (B1.1-B1.5)
2. Run all unit tests (B2.1-B2.4)
3. Document any failures

### Phase 3: Runtime Behavior (45 min)
1. Test worker registry (B3.1-B3.5)
2. Test model catalog (B4.1-B4.4)
3. Test daemon lifecycle (B5.1-B5.5)
4. Test worker spawn (B6.1-B6.5)
5. Test ephemeral mode (B7.1-B7.3)
6. Test shutdown (B8.1-B8.3)

### Phase 4: File System Verification (10 min)
1. Check database files (B9.1-B9.3)
2. Verify no orphaned files

### Phase 5: Documentation Review (15 min)
1. Verify all docs updated (B10.1-B10.3)
2. Check for inconsistencies

### Phase 6: E2E Testing (30 min)
1. Download model (B11.2)
2. Run preflight (B11.1)
3. Run E2E test (B11.3)
4. Verify cleanup (B11.4-B11.5)

**Total Estimated Time:** 2.5 hours

---

## Success Criteria

### Must Pass (Blocking)
- ✅ All builds succeed (B1.1-B1.4)
- ✅ All unit tests pass (B2.1-B2.4)
- ✅ No workers.db file exists (B3.1, B9.1)
- ✅ models.db file exists (B4.1, B9.2)
- ✅ Worker registry is in-memory (B3.2-B3.4)
- ✅ Model catalog is SQLite (B4.2-B4.4)

### Should Pass (Important)
- ✅ Daemon starts/stops cleanly (B5.1-B5.5)
- ✅ Worker spawn flow works (B6.1-B6.5)
- ✅ Ephemeral mode works (B7.1-B7.3)
- ✅ Shutdown cascades (B8.1-B8.3)
- ✅ Documentation accurate (B10.1-B10.3)

### Nice to Have (E2E)
- ✅ E2E test passes (B11.1-B11.5)
- ✅ Performance acceptable
- ✅ No memory leaks

---

## Failure Handling

### If Build Fails
1. Check error messages
2. Verify dependencies restored correctly
3. Check git status for uncommitted changes
4. Report to user with error details

### If Tests Fail
1. Run with `-- --nocapture` for full output
2. Check if failure is in new code or existing
3. Verify test expectations are correct
4. Fix or document issue

### If Behavior Incorrect
1. Check logs in /tmp/rbee-hive.log
2. Verify database files in ~/.rbee/
3. Check process list for orphans
4. Document unexpected behavior

---

## Deliverables

### Required Documents
1. **TEAM_031_TEST_RESULTS.md** - Test results for all behaviors
   - Format: Checklist with ✅/❌ for each behavior
   - Include error messages for failures
   - Include logs for critical behaviors

2. **TEAM_031_COMPLETION_SUMMARY.md** - Summary of verification
   - What was tested
   - What passed/failed
   - Any bugs found
   - Recommendations

### Optional Documents
- Performance metrics (if E2E works)
- Bug reports (if issues found)
- Architecture improvements (if discovered)

---

## Dev-Bee Rules Compliance

- [ ] Read dev-bee-rules.md ✅
- [ ] No background jobs (all blocking output) ✅
- [ ] No multiple .md files (max 2 for this task) ✅
- [ ] Add TEAM-031 signatures to changes ✅
- [ ] Complete ALL priorities in order ✅
- [ ] Follow existing TODO list ✅

---

## Notes for TEAM-031

**What TEAM-030 Did:**
- ✅ Removed worker-registry crate (SQLite)
- ✅ Kept model-catalog crate (SQLite)
- ✅ Enhanced in-memory worker registry
- ✅ Added cascading shutdown
- ✅ Updated all documentation

**What You Need to Do:**
1. Verify EVERY behavior in this checklist
2. Test both happy path and edge cases
3. Document ALL results (pass/fail)
4. Report any bugs found
5. Complete E2E test if model available

**Critical Behaviors:**
- Worker registry MUST be in-memory (no SQLite)
- Model catalog MUST be SQLite (persistent)
- No workers.db file MUST exist
- models.db file MUST exist

**Remember:**
- Don't skip behaviors
- Test thoroughly
- Document everything
- Follow dev-bee rules

---

**Good luck, TEAM-031! Test everything! 🧪**

**Created by:** TEAM-030  
**Date:** 2025-10-10T00:54:00+02:00  
**For:** TEAM-031 verification
