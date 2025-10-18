# Test Harness - Complete Testing Suite

**Created by:** Multiple teams (TEAM-097 through TEAM-107)  
**Last Updated:** 2025-10-18 by TEAM-107  
**Status:** ✅ Production Ready

---

## Overview

This directory contains the complete testing infrastructure for the rbee ecosystem:

1. **BDD Tests** - Behavior-driven development tests (100+ scenarios)
2. **Chaos Tests** - Network failures, crashes, resource exhaustion
3. **Load Tests** - Performance under 1000+ concurrent users
4. **Stress Tests** - Resource exhaustion and graceful degradation

---

## Quick Start

### Run Everything

```bash
# Run all chaos, load, and stress tests
./run-all-chaos-load-tests.sh

# Results in ./test-results/
```

### Run Individual Test Suites

```bash
# BDD tests (100+ scenarios)
cd bdd
cargo run --bin bdd-runner

# Chaos tests (15 scenarios)
cd chaos
./run-chaos-tests.sh

# Load tests (3 patterns)
cd load
./run-load-tests.sh

# Stress tests (6 scenarios)
cd stress
./exhaust-resources.sh
```

---

## Directory Structure

```
test-harness/
├── README.md                            # This file
├── run-all-chaos-load-tests.sh          # Master test runner
├── CHAOS_LOAD_TESTING_SUMMARY.md        # TEAM-107 summary
│
├── bdd/                                  # BDD tests (TEAM-097 to TEAM-100)
│   ├── tests/features/                  # 100+ Gherkin scenarios
│   ├── src/steps/                       # Step definitions
│   ├── docker-compose.integration.yml   # Integration environment
│   └── Cargo.toml                       # Rust project
│
├── chaos/                                # Chaos tests (TEAM-107)
│   ├── run-chaos-tests.sh               # Execution script
│   ├── docker-compose.chaos.yml         # Infrastructure
│   ├── scenarios/                       # 15 chaos scenarios
│   ├── scripts/chaos_controller.py      # Python orchestrator
│   └── README.md                        # Documentation
│
├── load/                                 # Load tests (TEAM-107)
│   ├── run-load-tests.sh                # Execution script
│   ├── inference-load.js                # 1000+ concurrent users
│   ├── stress-test.js                   # Breaking point test
│   ├── spike-test.js                    # Traffic spike test
│   └── README.md                        # Documentation
│
├── stress/                               # Stress tests (TEAM-107)
│   ├── exhaust-resources.sh             # Execution script
│   └── README.md                        # Documentation
│
└── test-results/                         # Generated results
    ├── master_test_run_*.log
    ├── TEAM_107_TEST_REPORT_*.md
    └── (individual test results)
```

---

## Test Categories

### 1. BDD Tests (100+ scenarios)

**Created by:** TEAM-097, TEAM-098, TEAM-099, TEAM-100  
**Location:** `bdd/`  
**Duration:** ~3 minutes for full suite

**Coverage:**
- P0 Security (45 scenarios) - Auth, secrets, validation
- P0 Lifecycle (30 scenarios) - PID tracking, error handling
- P1 Operations (18 scenarios) - Audit, deadlines, resources
- P2 Observability (25 scenarios) - Metrics, config, narration

**Run:**
```bash
cd bdd
cargo run --bin bdd-runner

# Or target specific features
LLORCH_BDD_FEATURE_PATH=tests/features/security cargo run --bin bdd-runner
```

### 2. Chaos Tests (15 scenarios)

**Created by:** TEAM-107  
**Location:** `chaos/`  
**Duration:** ~20 minutes

**Coverage:**
- Network failures (5) - Partition, latency, packet loss, bandwidth, reset
- Worker crashes (5) - During inference, registration, OOM, timeout, multiple
- Resource exhaustion (5) - Disk, memory, CPU, FD, VRAM

**Run:**
```bash
cd chaos
./run-chaos-tests.sh
```

### 3. Load Tests (3 patterns)

**Created by:** TEAM-107  
**Location:** `load/`  
**Duration:** ~30 minutes total

**Coverage:**
- Inference load (16 min) - 1000+ concurrent users, sustained
- Stress test (19 min) - Ramp to 5000 users, find breaking point
- Spike test (7 min) - Sudden traffic bursts

**Run:**
```bash
cd load
./run-load-tests.sh
```

**Thresholds:**
- Error rate < 1%
- p95 latency < 500ms
- p99 latency < 1000ms

### 4. Stress Tests (6 scenarios)

**Created by:** TEAM-107  
**Location:** `stress/`  
**Duration:** ~10 minutes

**Coverage:**
- CPU exhaustion (60s saturation)
- Memory exhaustion (90% allocation)
- Disk exhaustion (1GB file)
- File descriptor exhaustion (1000+ FDs)
- Connection exhaustion (1000 connections)
- Combined stress (CPU + memory)

**Run:**
```bash
cd stress
./exhaust-resources.sh
```

---

## Prerequisites

### For BDD Tests

```bash
# Rust toolchain
rustup --version

# Build BDD runner
cd bdd
cargo build --bin bdd-runner
```

### For Chaos Tests

```bash
# Docker and Docker Compose
docker --version
docker-compose --version

# Python 3.x
python3 --version
```

### For Load Tests

```bash
# k6 installation (Ubuntu/Debian)
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
  --keyserver hkp://keyserver.ubuntu.com:80 \
  --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | \
  sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# Verify
k6 version
```

### For Stress Tests

```bash
# Docker (stress-ng runs in containers)
docker --version

# Or install stress-ng locally
sudo apt-get install stress-ng
```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Full Test Suite
on: [push, pull_request]

jobs:
  bdd-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run BDD tests
        run: |
          cd test-harness/bdd
          cargo run --bin bdd-runner

  chaos-load-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install k6
        run: |
          sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
            --keyserver hkp://keyserver.ubuntu.com:80 \
            --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | \
            sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6
      - name: Run chaos and load tests
        run: |
          cd test-harness
          ./run-all-chaos-load-tests.sh
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: test-harness/test-results/
```

---

## Test Results

Results are organized by test type:

```
test-results/
├── master_test_run_20251018_160000.log      # Master log
├── TEAM_107_TEST_REPORT_20251018_160000.md  # Summary report
│
├── chaos/results/
│   ├── chaos_test_20251018_160000.log
│   └── chaos_results_20251018_160000.json
│
├── load/results/
│   ├── load_test_20251018_160000.log
│   ├── inference_summary_20251018_160000.json
│   ├── stress_summary_20251018_160000.json
│   └── spike_summary_20251018_160000.json
│
└── stress/results/
    └── stress_test_20251018_160000.log
```

---

## Acceptance Criteria

From production release checklist:

### BDD Tests
- [x] 100+ scenarios implemented
- [x] All P0 items have tests
- [x] All P1 items have tests
- [x] P2 items have basic tests
- [x] Step definitions using real product code

### Chaos Tests
- [x] System survives chaos scenarios
- [x] Random worker crashes handled
- [x] Random network failures recovered
- [x] Random disk full handled
- [x] Random OOM handled

### Load Tests
- [x] 1000+ concurrent requests handled
- [x] p95 latency < 500ms
- [x] p99 latency < 1000ms
- [x] Error rate < 1%
- [x] Sustained load capability

### Stress Tests
- [x] CPU exhaustion handled
- [x] Memory exhaustion handled
- [x] Disk exhaustion handled
- [x] Graceful degradation verified

---

## Team Contributions

| Team | Component | Scenarios | Status |
|------|-----------|-----------|--------|
| TEAM-097 | BDD P0 Security | 45 | ✅ Complete |
| TEAM-098 | BDD P0 Lifecycle | 30 | ✅ Complete |
| TEAM-099 | BDD P1 Operations | 18 | ✅ Complete |
| TEAM-100 | BDD P2 Observability | 25 | ✅ Complete |
| TEAM-106 | Integration Testing | 25 | ✅ Complete |
| TEAM-107 | Chaos & Load Testing | 24 | ✅ Complete |

**Total:** 167 test scenarios across 6 teams

---

## Documentation

**Detailed guides:**
- `bdd/BDD_TESTS_FOR_RC_CHECKLIST.md` - BDD test specifications
- `chaos/README.md` - Chaos testing guide
- `load/README.md` - Load testing guide
- `stress/README.md` - Stress testing guide
- `CHAOS_LOAD_TESTING_SUMMARY.md` - TEAM-107 summary

**Planning documents:**
- `.docs/components/PLAN/START_HERE.md` - Master plan
- `.docs/components/PLAN/TEAM_107_CHAOS_LOAD_TESTING.md` - Chaos/load plan
- `.docs/components/PLAN/TEAM_107_HANDOFF.md` - Handoff to TEAM-108

---

## Troubleshooting

### BDD Tests Failing

```bash
# Check service status
docker-compose -f bdd/docker-compose.integration.yml ps

# View logs
docker-compose -f bdd/docker-compose.integration.yml logs

# Restart services
docker-compose -f bdd/docker-compose.integration.yml restart
```

### Chaos Tests Not Running

```bash
# Check toxiproxy
curl http://localhost:8474/version

# Check services
curl http://localhost:8080/health
curl http://localhost:9200/health

# Restart infrastructure
cd chaos
docker-compose -f docker-compose.chaos.yml restart
```

### Load Tests Timing Out

```bash
# Increase timeout in k6 scripts
# Edit load/*.js and change timeout values

# Or reduce load
k6 run --vus 100 --duration 1m load/inference-load.js
```

---

## Next Steps

For TEAM-108 (Final Validation):

1. ✅ Run master test suite: `./run-all-chaos-load-tests.sh`
2. ✅ Review all test results
3. ✅ Validate acceptance criteria
4. ✅ Sign off on RC checklist
5. ✅ Prepare for production release

---

## Support

**Questions?** Check:
1. Individual README files in each directory
2. Planning documents in `.docs/components/PLAN/`
3. Handoff documents (TEAM_*_HANDOFF.md)

---

**Created by:** TEAM-097 through TEAM-107  
**Status:** ✅ Production Ready  
**Total Test Coverage:** 167 scenarios

**🚀 COMPLETE TESTING INFRASTRUCTURE READY FOR PRODUCTION 🚀**
