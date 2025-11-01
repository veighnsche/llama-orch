# TEAM-364 PHASES 3-6: REMAINING WORK

**Team:** TEAM-364  
**Status:** 📋 PLANNED  
**Total Estimated Duration:** 8-11 days

---

## PHASE 3: PROCESSMONITOR TESTS (2-3 days)

### **Objectives**
Implement all 14 ProcessMonitor unit tests for worker spawn and telemetry collection

### **Requirements**
- Linux with cgroup v2 support
- Root/sudo access for cgroup creation
- nvidia-smi (optional, tests degrade gracefully)

### **Tasks**

**1. Helper Functions (4 hours)**
- Mock nvidia-smi command replacement
- Test worker spawning utilities
- Cgroup cleanup helpers
- Platform detection utilities

**2. Worker Spawn Tests (4 hours)**
- `test_spawn_creates_cgroup` - Verify cgroup directory created
- `test_spawn_applies_cpu_limit` - Verify cpu.max written correctly
- `test_spawn_applies_memory_limit` - Verify memory.max written
- `test_spawn_returns_valid_pid` - Verify PID valid and process exists

**3. Collection Tests (6 hours)**
- `test_collect_reads_cgroup_stats` - CPU, memory, uptime collection
- `test_collect_queries_nvidia_smi` - GPU stats collection
- `test_collect_parses_cmdline` - Model name extraction
- `test_collect_calculates_uptime` - Uptime calculation accuracy
- `test_collect_handles_missing_gpu` - Graceful degradation
- `test_collect_handles_dead_process` - Error handling
- `test_enumerate_walks_cgroup_tree` - Multi-worker enumeration

**4. Error Handling Tests (2 hours)**
- `test_spawn_invalid_binary` - Binary not found
- `test_spawn_invalid_cpu_limit` - Invalid limit format

**5. Platform Tests (2 hours)**
- `test_spawn_fallback_on_non_linux` - macOS/Windows fallback
- `test_collect_fallback_on_non_linux` - Platform detection

### **Acceptance Criteria**
- [ ] All 14 tests passing on Linux
- [ ] Platform fallbacks tested on macOS
- [ ] No `#[ignore]` on unit tests
- [ ] Tests run in CI (Linux runner)

---

## PHASE 4: INTEGRATION TESTS (3-4 days)

### **Objectives**
End-to-end tests for complete telemetry pipeline with real binaries

### **Requirements**
- Compiled queen-rbee and rbee-hive binaries
- Linux with cgroup v2
- Test harness for process management

### **Tasks**

**1. Test Harness Infrastructure (1 day)**
- Binary compilation helpers
- Process management (start/stop Queen/Hive)
- Log collection and analysis
- Cleanup on panic/failure
- Port allocation for test isolation

**2. E2E Flow Tests (1 day)**
- `test_end_to_end_telemetry_flow` - Full pipeline verification
- `test_worker_dies_removed_from_registry` - Stale cleanup (90s)
- `test_scheduling_queries` - All query types
- `test_queen_restart_recovers` - Queen restart handling
- `test_hive_restart_clears_workers` - Hive restart handling

**3. Fault Injection Tests (1 day)**
- `test_queen_unreachable` - Hive continues without Queen
- `test_nvidia_smi_timeout` - Verify timeout fix works
- `test_cgroup_permission_denied` - Permission error handling
- `test_broadcast_channel_full` - Slow consumer handling

**4. Integration Test Utilities (0.5 days)**
- Mock HTTP servers
- SSE client implementation
- Test worker spawning
- Daemon lifecycle management

### **Acceptance Criteria**
- [ ] All 9 integration tests passing
- [ ] E2E flow verified (spawn → SSE)
- [ ] Fault injection working
- [ ] Test harness reusable for future tests

---

## PHASE 5: PERFORMANCE TESTS (2 days)

### **Objectives**
Establish performance baselines and stress test the system

### **Tasks**

**1. Benchmarking Infrastructure (0.5 days)**
- Set up criterion for precise measurements
- Configure baseline tracking
- CI integration for regression detection

**2. Collection Benchmarks (0.5 days)**
- `bench_collection_10_workers` - Target: <10ms
- `bench_heartbeat_payload_size` - Target: <100KB for 100 workers

**3. Latency Benchmarks (0.5 days)**
- `bench_sse_latency` - Target: <100ms (spawn → UI)
- Measure each pipeline stage individually

**4. Stress Tests (0.5 days)**
- `stress_100_workers` - Target: <1s collection time
- Memory profiling under load
- CPU profiling under load
- Identify bottlenecks

### **Acceptance Criteria**
- [ ] Baselines established for all benchmarks
- [ ] Stress tests pass (100 workers)
- [ ] Performance regression detection in CI
- [ ] Bottlenecks identified and documented

---

## PHASE 6: CI/CD INTEGRATION (1 day)

### **Objectives**
Automate all tests in CI/CD pipeline

### **Tasks**

**1. GitHub Actions Workflow (4 hours)**

Create `.github/workflows/telemetry-tests.yml`:
```yaml
name: Telemetry Pipeline Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install cgroup v2
        run: |
          # Setup cgroup v2 if needed
      - name: Run unit tests
        run: |
          cargo test -p rbee-hive-monitor
          cargo test -p queen-rbee-hive-registry
          cargo test -p rbee-hive -- heartbeat
          cargo test -p queen-rbee -- heartbeat_stream

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build binaries
        run: |
          cargo build --bin queen-rbee --bin rbee-hive
      - name: Run integration tests
        run: |
          cargo test -p xtask --lib integration::telemetry_tests

  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: |
          cargo test -p xtask --lib integration::telemetry_tests -- bench
```

**2. Coverage Reports (2 hours)**
- Set up tarpaulin or cargo-llvm-cov
- Generate coverage reports
- Upload to codecov or similar

**3. Documentation Updates (2 hours)**
- Update README with test instructions
- Add troubleshooting guide
- Document platform requirements
- Add CI badge to README

### **Acceptance Criteria**
- [ ] CI runs all tests on every commit
- [ ] Integration tests run on PR
- [ ] Coverage reports generated
- [ ] Documentation complete

---

## 📊 OVERALL PROGRESS TRACKING

| Phase | Duration | Tests | Status |
|-------|----------|-------|--------|
| Phase 1: Foundation | 0.5 days | 12 | ✅ DONE |
| Phase 2: Quick Wins | 0.2 days | 2 fixes | 📋 NEXT |
| Phase 3: ProcessMonitor | 2-3 days | 14 | 📋 TODO |
| Phase 4: Integration | 3-4 days | 9 | 📋 TODO |
| Phase 5: Performance | 2 days | 4 | 📋 TODO |
| Phase 6: CI/CD | 1 day | N/A | 📋 TODO |
| **TOTAL** | **8.7-11.7 days** | **41 tests** | **17% done** |

---

## 🚨 REMAINING CRITICAL ISSUES

After Phase 2, these issues remain:

**Issue #2: CPU% Always Returns 0.0 (MEDIUM)**
- Fix in Phase 3 alongside ProcessMonitor tests
- Requires time-delta tracking

**Issue #3: I/O Rates Always Return 0.0 (LOW)**
- Fix in Phase 3 alongside ProcessMonitor tests
- Requires rate calculation

**Issue #7: Collection Fails on First Error (MEDIUM)**
- Fix in Phase 3 alongside ProcessMonitor tests
- Change error handling to continue on failure

---

## 📝 NOTES FOR FUTURE TEAMS

### **Platform Requirements**

**Linux (Full Support):**
- cgroup v2 filesystem mounted
- Root/sudo for cgroup creation
- nvidia-smi (optional)

**macOS (Limited Support):**
- No cgroup support
- Fallback spawn only
- Limited telemetry

**Windows (Not Supported):**
- No cgroup support
- Future work needed

### **Test Execution Tips**

```bash
# Run all tests
cargo test -p rbee-hive-monitor
cargo test -p queen-rbee-hive-registry

# Run specific test
cargo test -p rbee-hive-monitor test_spawn_creates_cgroup

# Run with output
cargo test -p rbee-hive-monitor -- --nocapture

# Run integration tests
cargo test -p xtask --lib integration::telemetry_tests

# Run benchmarks
cargo test -p xtask --lib -- bench --nocapture
```

### **Troubleshooting**

**Test hangs:**
- Check for missing cleanup (kill_worker)
- Add timeout: `#[tokio::test(timeout = "10s")]`

**cgroup permission denied:**
- Run with sudo: `sudo -E cargo test`
- Or add user to cgroup writable group

**nvidia-smi not found:**
- Tests degrade gracefully (GPU stats = 0)
- Expected on non-GPU systems

---

## 🎯 FINAL DELIVERABLES

When all phases complete:

- ✅ 41 tests passing (unit + integration + performance)
- ✅ 7 critical issues fixed
- ✅ CI/CD pipeline automated
- ✅ Performance baselines established
- ✅ Comprehensive documentation
- ✅ Platform support documented

---

**Document Created:** Oct 30, 2025  
**Total Remaining Effort:** 8-11 days  
**Current Progress:** 17% (Phase 1 complete)
