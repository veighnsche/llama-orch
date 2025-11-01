# TEAM-364 TODO LIST

**Team:** TEAM-364  
**Mission:** Telemetry Pipeline Testing & Critical Fixes  
**Date Started:** Oct 30, 2025  
**Status:** Phase 1 Complete (17% done)

---

## 🎯 OVERALL MISSION

Implement comprehensive testing for the telemetry pipeline and fix all critical issues that could cause hangs, data loss, or incorrect behavior.

---

## ✅ COMPLETED (Phase 1)

- [x] Investigation complete - all behaviors documented
- [x] Test infrastructure enabled (4 Cargo.toml files)
- [x] Critical Issue #1 fixed - nvidia-smi timeout
- [x] Critical Issue #6 fixed - heartbeat HTTP timeout
- [x] 12 worker telemetry tests created and passing
- [x] All team signatures updated to TEAM-364

---

## 🔄 IN PROGRESS (Phase 2)

### **Quick Wins - Estimated 1.5 hours**

- [ ] **Fix Issue #4: Auto Stale Cleanup (30 min)**
  - Location: `bin/10_queen_rbee/src/main.rs`
  - Add background task to call `cleanup_stale()` every 60s
  - Prevents dead workers from accumulating in registry

- [ ] **Fix Issue #5: Dynamic VRAM Detection (1 hour)**
  - Location: `bin/15_queen_rbee_crates/hive-registry/src/registry.rs:151`
  - Query actual GPU VRAM from nvidia-smi
  - Replace hardcoded 24GB limit with per-GPU detection

---

## 📋 TODO (Phase 3-6)

### **Phase 3: ProcessMonitor Tests (2-3 days)**

- [ ] **Implement helper functions for tests**
  - [ ] Mock nvidia-smi command
  - [ ] Test worker spawning helpers
  - [ ] Cgroup cleanup utilities

- [ ] **Worker spawn tests (Linux-only)**
  - [ ] test_spawn_creates_cgroup
  - [ ] test_spawn_applies_cpu_limit
  - [ ] test_spawn_applies_memory_limit
  - [ ] test_spawn_returns_valid_pid

- [ ] **Collection tests**
  - [ ] test_collect_reads_cgroup_stats
  - [ ] test_collect_queries_nvidia_smi
  - [ ] test_collect_parses_cmdline
  - [ ] test_collect_calculates_uptime
  - [ ] test_collect_handles_missing_gpu
  - [ ] test_collect_handles_dead_process
  - [ ] test_enumerate_walks_cgroup_tree

- [ ] **Error handling tests**
  - [ ] test_spawn_invalid_binary
  - [ ] test_spawn_invalid_cpu_limit

- [ ] **Platform tests**
  - [ ] test_spawn_fallback_on_non_linux
  - [ ] test_collect_fallback_on_non_linux

### **Phase 4: Integration Tests (3-4 days)**

- [ ] **Test harness infrastructure**
  - [ ] Binary compilation helpers
  - [ ] Process management (start/stop Queen/Hive)
  - [ ] Log collection
  - [ ] Cleanup on panic

- [ ] **E2E flow tests**
  - [ ] test_end_to_end_telemetry_flow
  - [ ] test_worker_dies_removed_from_registry
  - [ ] test_scheduling_queries
  - [ ] test_queen_restart_recovers
  - [ ] test_hive_restart_clears_workers

- [ ] **Fault injection tests**
  - [ ] test_queen_unreachable
  - [ ] test_nvidia_smi_timeout (verify fix)
  - [ ] test_cgroup_permission_denied
  - [ ] test_broadcast_channel_full

### **Phase 5: Performance Tests (2 days)**

- [ ] **Benchmarking infrastructure**
  - [ ] Set up criterion for precise measurements
  - [ ] Track metrics over time
  - [ ] CI integration

- [ ] **Benchmarks**
  - [ ] bench_collection_10_workers (target: <10ms)
  - [ ] bench_heartbeat_payload_size (target: <100KB for 100 workers)
  - [ ] bench_sse_latency (target: <100ms)
  - [ ] stress_100_workers (target: <1s collection)

### **Phase 6: CI/CD Integration (1 day)**

- [ ] **GitHub Actions workflow**
  - [ ] Create `.github/workflows/telemetry-tests.yml`
  - [ ] Run unit tests on every commit
  - [ ] Run integration tests on PR
  - [ ] Generate coverage reports

- [ ] **Documentation updates**
  - [ ] Update README with test instructions
  - [ ] Add troubleshooting guide
  - [ ] Document platform requirements

---

## 🚨 REMAINING CRITICAL ISSUES

### **Issue #2: CPU% Always Returns 0.0 (MEDIUM)**
- **Risk:** Scheduling can't use CPU metrics
- **Effort:** 2 hours
- **Location:** `bin/25_rbee_hive_crates/monitor/src/monitor.rs:337`
- **Fix:** Track previous `usage_usec`, calculate delta over time
- **Test:** Update `test_collect_reads_cgroup_stats`

### **Issue #3: I/O Rates Always Return 0.0 (LOW)**
- **Risk:** Not used for scheduling (low priority)
- **Effort:** 2 hours
- **Location:** `bin/25_rbee_hive_crates/monitor/src/monitor.rs:354`
- **Fix:** Track previous `io.stat`, calculate rate
- **Test:** Update `test_collect_reads_cgroup_stats`

### **Issue #4: No Automatic Stale Cleanup (MEDIUM)** ⬅️ NEXT
- **Risk:** Dead workers accumulate in registry
- **Effort:** 30 minutes
- **Location:** `bin/10_queen_rbee/src/main.rs`
- **Fix:** Spawn background task calling `cleanup_stale()` every 60s
- **Test:** `test_worker_dies_removed_from_registry`

### **Issue #5: Hardcoded 24GB VRAM Limit (MEDIUM)** ⬅️ NEXT
- **Risk:** Breaks on different GPUs
- **Effort:** 1 hour
- **Location:** `bin/15_queen_rbee_crates/hive-registry/src/registry.rs:151`
- **Fix:** Query GPU VRAM from nvidia-smi, store per-worker
- **Test:** `test_find_workers_with_capacity_checks_vram`

### **Issue #7: Collection Fails on First Error (MEDIUM)**
- **Risk:** One dead worker breaks all telemetry
- **Effort:** 1 hour
- **Location:** `bin/25_rbee_hive_crates/monitor/src/monitor.rs:256`
- **Fix:** Continue on error, collect what you can
- **Test:** `test_collect_all_workers_partial_failure`

---

## 📊 PROGRESS TRACKING

| Phase | Status | Tests | Effort | Complete |
|-------|--------|-------|--------|----------|
| Phase 1: Foundation | ✅ DONE | 12/12 | 0.5 days | 100% |
| Phase 2: Quick Wins | 🔄 NEXT | 0/2 | 0.2 days | 0% |
| Phase 3: ProcessMonitor | 📋 TODO | 0/14 | 2-3 days | 0% |
| Phase 4: Integration | 📋 TODO | 0/9 | 3-4 days | 0% |
| Phase 5: Performance | 📋 TODO | 0/4 | 2 days | 0% |
| Phase 6: CI/CD | 📋 TODO | 0/0 | 1 day | 0% |
| **TOTAL** | **17%** | **12/41** | **8-11 days** | **17%** |

---

## 🎯 ACCEPTANCE CRITERIA

### **Phase 2 Complete When:**
- [ ] Issue #4 fixed (auto stale cleanup)
- [ ] Issue #5 fixed (dynamic VRAM)
- [ ] Tests verify both fixes work
- [ ] No regressions in existing tests

### **Phase 3 Complete When:**
- [ ] All 14 ProcessMonitor tests passing
- [ ] Tests run on Linux (cgroup v2 required)
- [ ] Platform fallbacks tested
- [ ] No `#[ignore]` on unit tests

### **Phase 4 Complete When:**
- [ ] All 9 integration tests passing
- [ ] E2E flow verified
- [ ] Fault injection working
- [ ] Test harness reusable

### **Phase 5 Complete When:**
- [ ] Benchmarks establish baselines
- [ ] Stress tests pass (100 workers)
- [ ] Performance regression detection in CI

### **Phase 6 Complete When:**
- [ ] CI runs all tests
- [ ] Tests pass on every commit
- [ ] Coverage reports generated
- [ ] Documentation complete

---

## 📝 NOTES FOR NEXT TEAM

### **What's Working**
- Worker telemetry storage and scheduling queries (12 tests passing)
- nvidia-smi timeout protection (no more hangs)
- Heartbeat HTTP timeout (no more hangs)
- Thread-safe concurrent access to HiveRegistry

### **What Needs Attention**
- ProcessMonitor tests need Linux with cgroup v2
- Integration tests need compiled binaries
- Performance tests need baseline establishment
- CI/CD needs GitHub Actions setup

### **Platform Requirements**
- **Linux:** Full cgroup v2 support, nvidia-smi optional
- **macOS:** Fallback mode, no cgroups, limited testing
- **Windows:** Not supported yet

### **Test Execution**
```bash
# Run all passing tests
cargo test -p queen-rbee-hive-registry

# Run specific test file
cargo test -p queen-rbee-hive-registry --test worker_telemetry_tests

# Run with output
cargo test -p queen-rbee-hive-registry -- --nocapture
```

---

## 🔗 RELATED DOCUMENTS

- **Investigation:** `bin/.plan/TELEMETRY_INVESTIGATION.md`
- **Testing Summary:** `bin/.plan/TELEMETRY_TESTING_SUMMARY.md`
- **Testing Index:** `bin/.plan/TELEMETRY_TESTING_INDEX.md`
- **Fixes Complete:** `bin/.plan/TELEMETRY_FIXES_COMPLETE.md`
- **Phase Documents:** `bin/.plan/TEAM_364_PHASE_*.md`

---

**Last Updated:** Oct 30, 2025  
**Next Action:** Phase 2 - Fix Issue #4 (auto stale cleanup)  
**Estimated Time to Complete:** 8-11 days remaining
