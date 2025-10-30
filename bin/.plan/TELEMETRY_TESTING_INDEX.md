# TELEMETRY PIPELINE TESTING - INDEX

**Date:** Oct 30, 2025  
**Status:** 📋 READY FOR IMPLEMENTATION

---

## 📚 QUICK NAVIGATION

### **Start Here**
1. 📖 [TELEMETRY_PIPELINE_COMPLETE.md](./TELEMETRY_PIPELINE_COMPLETE.md) - Original production documentation
2. 🔍 [TELEMETRY_INVESTIGATION.md](./TELEMETRY_INVESTIGATION.md) - Deep behavior analysis
3. 📋 [TELEMETRY_TESTING_SUMMARY.md](./TELEMETRY_TESTING_SUMMARY.md) - Implementation roadmap

### **Test Files**
- Unit Tests (31 total)
- Integration Tests (13 total)
- Performance Tests (4 total)
- Fault Injection Tests (5 total)

---

## 📂 DOCUMENT STRUCTURE

```
bin/.plan/
├─ TELEMETRY_PIPELINE_COMPLETE.md     [PRODUCTION DOC]
│  ├─ Complete data flow (6 stages)
│  ├─ Key files by component
│  ├─ Debugging guide
│  ├─ Configuration options
│  └─ Known limitations
│
├─ TELEMETRY_INVESTIGATION.md         [BEHAVIOR ANALYSIS]
│  ├─ Component behaviors (detailed)
│  ├─ All failure scenarios
│  ├─ 7 critical issues found
│  ├─ Data flow timing
│  └─ Testing requirements
│
├─ TELEMETRY_TESTING_SUMMARY.md       [IMPLEMENTATION PLAN]
│  ├─ Test breakdown (53 tests)
│  ├─ Critical issues with fixes
│  ├─ 6-phase implementation plan
│  ├─ Estimated effort (11-14 days)
│  └─ Acceptance criteria
│
├─ TELEMETRY_TESTING_INDEX.md         [THIS FILE]
│  └─ Navigation and quick reference
│
└─ TELEMETRY_INTEGRATION_TESTS.rs     [INTEGRATION STUBS]
   ├─ 9 end-to-end tests
   ├─ 4 performance benchmarks
   ├─ 5 fault injection tests
   └─ Helper function definitions
```

---

## 🧪 TEST FILE LOCATIONS

### **Unit Tests**

```
bin/25_rbee_hive_crates/monitor/tests/
├─ process_monitor_tests.rs            [14 tests]
│  ├─ Worker spawn with cgroups
│  ├─ Resource limit enforcement
│  ├─ Telemetry collection
│  ├─ GPU stats via nvidia-smi
│  ├─ Model detection from cmdline
│  └─ Platform fallbacks
│
└─ telemetry_collection_tests.rs       [8 tests]
   ├─ collect_all_workers()
   ├─ collect_group()
   ├─ collect_instance()
   └─ Error handling

bin/15_queen_rbee_crates/hive-registry/tests/
└─ worker_telemetry_tests.rs           [15 tests]
   ├─ Worker storage (update/get)
   ├─ Scheduling queries (idle/model/capacity)
   ├─ Thread safety (concurrent access)
   └─ Edge cases

bin/20_rbee_hive/tests/
└─ heartbeat_tests.rs                  [10 tests]
   ├─ HTTP POST to Queen
   ├─ Worker inclusion
   ├─ Interval timing (1s)
   ├─ Retry logic
   └─ Error handling

bin/10_queen_rbee/tests/
└─ heartbeat_stream_tests.rs           [11 tests]
   ├─ Queen heartbeat events (2.5s)
   ├─ Hive telemetry forwarding (1s)
   ├─ Multiple clients
   ├─ Broadcast channel handling
   └─ Event format validation
```

### **Integration Tests**

```
bin/.plan/
└─ TELEMETRY_INTEGRATION_TESTS.rs     [To be moved to xtask/]
   ├─ E2E flow (9 tests)
   ├─ Performance benchmarks (4 tests)
   └─ Fault injection (5 tests)

Target location:
xtask/src/integration/
└─ telemetry_tests.rs                  [Move here in Phase 4]
```

---

## 🚨 CRITICAL ISSUES QUICK REFERENCE

| # | Issue | Risk | Location | Test |
|---|-------|------|----------|------|
| 1 | nvidia-smi no timeout | HIGH | monitor.rs:363 | test_nvidia_smi_timeout |
| 2 | CPU% always 0.0 | MED | monitor.rs:339 | test_collect_reads_cgroup_stats |
| 3 | I/O rates always 0.0 | LOW | monitor.rs:354 | test_collect_reads_cgroup_stats |
| 4 | No auto stale cleanup | MED | registry.rs:161 | test_worker_dies_removed |
| 5 | Hardcoded 24GB VRAM | MED | registry.rs:151 | test_find_workers_with_capacity |
| 6 | Heartbeat no timeout | MED | heartbeat.rs:31 | test_heartbeat_timeout |
| 7 | Collection fails on error | MED | monitor.rs:256 | test_collect_partial_failure |

**Fix Priority:** #1 (HIGH) → #7 (MED) → #2 (MED) → #4 (MED) → #5 (MED) → #6 (MED) → #3 (LOW)

---

## 📊 TEST COVERAGE MATRIX

| Component | Unit Tests | Integration | Performance | Total |
|-----------|------------|-------------|-------------|-------|
| ProcessMonitor | 14 | - | - | 14 |
| Telemetry Collection | 8 | - | - | 8 |
| HiveRegistry | 15 | - | - | 15 |
| Heartbeat Sending | 10 | - | - | 10 |
| SSE Streaming | 11 | - | - | 11 |
| End-to-End | - | 9 | - | 9 |
| Performance | - | - | 4 | 4 |
| Fault Injection | - | 5 | - | 5 |
| **TOTAL** | **58** | **14** | **4** | **76** |

---

## ⏱️ IMPLEMENTATION TIMELINE

```
Week 1: Foundation
├─ Day 1: Phase 1 (Enable modules) ✓
├─ Day 2-3: Phase 2 (Helper functions)
└─ Day 4-5: Phase 3 (Unit tests) - Start

Week 2: Testing
├─ Day 6-7: Phase 3 (Unit tests) - Complete
├─ Day 8-9: Phase 4 (Integration tests) - Start
└─ Day 10: Phase 4 (Integration tests) - Continue

Week 3: Polish
├─ Day 11-12: Phase 4 (Integration tests) - Complete
├─ Day 13-14: Phase 5 (Performance tests)
└─ Day 15: Phase 6 (CI/CD)
```

**Parallel Work:**
- Unit tests for different components (independent)
- Performance tests while integration tests run
- Documentation updates throughout

---

## 🎯 SUCCESS CRITERIA

### **Phase 1: Modules Enabled**
```bash
cd bin/25_rbee_hive_crates/monitor
cargo test --no-run  # Should compile
```

### **Phase 2: Helpers Implemented**
```bash
cargo test process_monitor_tests::test_spawn_creates_cgroup
# Should run (may fail, but no unimplemented!)
```

### **Phase 3: Unit Tests Pass**
```bash
cargo test -p rbee-hive-monitor
cargo test -p queen-rbee-hive-registry
cargo test -p rbee-hive -- heartbeat
cargo test -p queen-rbee -- heartbeat_stream
# All should pass ✓
```

### **Phase 4: Integration Tests Pass**
```bash
cargo test -p xtask --lib integration::telemetry_tests
# All E2E tests pass ✓
```

### **Phase 5: Performance Baseline**
```bash
cargo test -p xtask --lib integration::telemetry_tests -- bench --nocapture
# Benchmarks establish baselines ✓
```

### **Phase 6: CI Green**
```bash
gh workflow run telemetry-tests.yml
gh run list --workflow=telemetry-tests.yml
# All checks pass ✓
```

---

## 🔧 COMMON COMMANDS

### **Run All Unit Tests**
```bash
cargo test -p rbee-hive-monitor
cargo test -p queen-rbee-hive-registry
cargo test -p rbee-hive -- heartbeat
cargo test -p queen-rbee -- heartbeat_stream
```

### **Run Specific Test**
```bash
cargo test -p rbee-hive-monitor test_spawn_creates_cgroup
```

### **Run With Output**
```bash
cargo test -p rbee-hive-monitor -- --nocapture
```

### **Run Integration Tests**
```bash
cargo test -p xtask --lib integration::telemetry_tests
```

### **Run Benchmarks**
```bash
cargo test -p xtask --lib integration::telemetry_tests -- bench --nocapture
```

### **Check Test Compilation**
```bash
cargo test --no-run -p rbee-hive-monitor
```

---

## 📖 KEY CONCEPTS

### **Test Categories**

1. **Unit Tests**
   - Test single component in isolation
   - Fast, no external dependencies
   - Can mock interfaces
   - Run on every commit

2. **Integration Tests**
   - Test multiple components together
   - Slower, requires binaries
   - Real cgroups, real processes
   - Run before merge

3. **Performance Tests**
   - Measure speed/throughput
   - Track regressions
   - Establish baselines
   - Run weekly

4. **Fault Injection**
   - Test error handling
   - Simulate failures
   - Verify recovery
   - Run before release

### **Platform Considerations**

- **Linux:** Full cgroup v2, nvidia-smi support
- **macOS:** Fallback mode, no cgroups
- **Windows:** Not supported (yet)

### **Test Isolation**

Each test must:
1. Create unique cgroup paths (different instance IDs)
2. Clean up workers (kill PIDs)
3. Remove cgroup directories
4. Not interfere with other tests

---

## 🆘 TROUBLESHOOTING

### **Test Hangs**
- Check for missing `kill_worker()` cleanup
- Verify `#[tokio::test]` on async tests
- Add timeout: `#[tokio::test(timeout = "10s")]`

### **cgroup Permission Denied**
- Tests require root OR
- User in cgroup writable group OR
- Run with `sudo -E cargo test`

### **nvidia-smi Not Found**
- Tests degrade gracefully (GPU stats = 0)
- Expected on non-GPU systems
- CI runners may not have GPUs

### **SSE Connection Fails**
- Check Queen/Hive started
- Verify ports not in use
- Check firewall rules

### **Race Conditions**
- Add `sleep()` after spawn
- Increase timeout windows
- Use `assert_eq!` with ranges

---

## 📝 NOTES

### **Historical Context**

- Teams 359-363 implemented telemetry pipeline (Oct 2025)
- No tests written during implementation
- This investigation found 7 critical issues
- Test stubs document expected behavior

### **Design Decisions**

- **1s collection interval** - Real-time scheduling
- **0.0 idle threshold** - Exact match, not approximate
- **90s stale timeout** - 3 missed heartbeats
- **Graceful degradation** - Continue without GPU

### **Future Work**

- CPU% calculation (time-delta needed)
- I/O rate calculation (rate tracking)
- Multi-GPU support (sum across GPUs)
- Auto stale cleanup (background task)
- Dynamic VRAM detection (query GPU)

---

## 🎓 LEARNING RESOURCES

### **cgroups v2**
- Kernel docs: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
- Resource limits: cpu.max, memory.max
- Stats: cpu.stat, memory.current, io.stat

### **nvidia-smi**
- Query format: `--query-compute-apps=pid,used_memory,sm`
- Output parsing: CSV with noheader, nounits
- Timeout handling: Required!

### **SSE (Server-Sent Events)**
- Protocol: text/event-stream
- Event format: `event: type\ndata: json\n\n`
- Keep-alive: Empty comment lines

### **Testing in Rust**
- `#[tokio::test]` for async tests
- `#[ignore]` for expensive tests
- `#[cfg(target_os = "linux")]` for platform-specific

---

**Ready to start?** → Begin with [TELEMETRY_TESTING_SUMMARY.md](./TELEMETRY_TESTING_SUMMARY.md) Phase 1

**Need details?** → See [TELEMETRY_INVESTIGATION.md](./TELEMETRY_INVESTIGATION.md)

**Production reference?** → Read [TELEMETRY_PIPELINE_COMPLETE.md](./TELEMETRY_PIPELINE_COMPLETE.md)
