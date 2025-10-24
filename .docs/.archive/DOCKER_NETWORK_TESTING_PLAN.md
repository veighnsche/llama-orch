# Docker Network Testing Plan for Queen-Rbee → Rbee-Hive Communication

**Date:** Oct 24, 2025  
**Purpose:** Comprehensive Docker-based testing foundation for all queen → hive communication scenarios  
**Status:** 🎯 READY FOR IMPLEMENTATION

---

## Executive Summary

This plan establishes a **production-grade Docker testing infrastructure** to validate all queen-rbee → rbee-hive communication patterns in isolated network environments.

**Coverage:**
- 3 Network Topologies (localhost, single remote, multi-remote)
- 7 Communication Patterns (HTTP, SSH, heartbeats, job streaming)
- 4 Failure Scenarios (network partitions, timeouts, crashes, restarts)
- Full E2E Workflows (hive lifecycle, worker spawning, model management)

---

## Architecture Overview

```text
Test Host (Your Machine)
  │
  ├─ rbee-keeper (CLI) → localhost:8500
  │
  └─ Docker Network: rbee-test-net (172.20.0.0/16)
      │
      ├─ queen-rbee (172.20.0.10:8500)
      │   ├─ HTTP API
      │   ├─ SSE streams
      │   └─ Job registry
      │
      ├─ rbee-hive-1 (172.20.0.20)
      │   ├─ HTTP :9000
      │   ├─ SSH :22
      │   └─ Workers
      │
      └─ rbee-hive-2 (172.20.0.21)
          ├─ HTTP :9000
          ├─ SSH :22
          └─ Workers
```

---

## Test Infrastructure

### Directory Structure
```
tests/docker/
├── Dockerfile.base          # Base image (Rust + SSH)
├── Dockerfile.queen         # Queen image
├── Dockerfile.hive          # Hive image
├── Dockerfile.worker        # Worker image
├── docker-compose.localhost.yml
├── docker-compose.multi-hive.yml
├── docker-compose.full-stack.yml
├── keys/
│   ├── test_id_rsa
│   └── test_id_rsa.pub
├── configs/
│   ├── queen.toml
│   └── hives.conf
└── scripts/
    ├── setup.sh
    ├── run-tests.sh
    └── cleanup.sh
```

---

## Test Categories

### 1. HTTP Communication Tests
**File:** `xtask/tests/docker/http_communication_tests.rs`

- Queen → Hive health checks
- Queen → Hive capabilities discovery
- Queen → Hive job submission
- SSE streaming (job results)
- HTTP timeout handling
- Connection refused scenarios
- Retry on failure

### 2. SSH Communication Tests
**File:** `xtask/tests/docker/ssh_communication_tests.rs`

- SSH connection establishment
- Command execution via SSH
- Binary installation via SCP
- Authentication failures
- SSH timeouts
- Concurrent SSH connections

### 3. Heartbeat Tests
**File:** `xtask/tests/docker/heartbeat_tests.rs`

- Worker → Queen heartbeat registration
- Heartbeat updates (last_seen)
- Staleness detection (30s timeout)
- Heartbeat after Queen restart
- Multiple workers heartbeats

### 4. Lifecycle Tests
**File:** `xtask/tests/docker/lifecycle_tests.rs`

- Hive start via Queen
- Hive stop via Queen
- Hive restart
- Hive status checks
- Capabilities refresh

### 5. Worker Lifecycle Tests
**File:** `xtask/tests/docker/worker_lifecycle_tests.rs`

- Worker spawn via Queen
- Worker list
- Worker delete
- Worker spawn on specific device

### 6. Failure Scenario Tests
**File:** `xtask/tests/docker/failure_tests.rs`

- Network partition (Queen ↔ Hive)
- Hive crash during operation
- Queen restart with active hives
- Worker crash (heartbeat stops)
- Concurrent operations on same hive
- SSH connection lost during install

### 7. End-to-End Workflows
**File:** `xtask/tests/docker/e2e_tests.rs`

- Fresh install → inference
- Multi-hive load balancing
- Model download → worker spawn
- Hive failure recovery

---

## Docker Compose Configurations

### Topology 1: Localhost Only
```yaml
# docker-compose.localhost.yml
version: '3.8'
services:
  queen:
    build:
      context: ../..
      dockerfile: tests/docker/Dockerfile.queen
    ports:
      - "8500:8500"
    networks:
      rbee-test:
        ipv4_address: 172.20.0.10

  hive-localhost:
    build:
      context: ../..
      dockerfile: tests/docker/Dockerfile.hive
    ports:
      - "9000:9000"
      - "2222:22"
    networks:
      rbee-test:
        ipv4_address: 172.20.0.20

networks:
  rbee-test:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Topology 2: Multi-Hive
```yaml
# docker-compose.multi-hive.yml
version: '3.8'
services:
  queen:
    build:
      context: ../..
      dockerfile: tests/docker/Dockerfile.queen
    ports:
      - "8500:8500"
    networks:
      rbee-test:
        ipv4_address: 172.20.0.10

  hive-1:
    build:
      context: ../..
      dockerfile: tests/docker/Dockerfile.hive
    ports:
      - "9001:9000"
      - "2221:22"
    networks:
      rbee-test:
        ipv4_address: 172.20.0.20

  hive-2:
    build:
      context: ../..
      dockerfile: tests/docker/Dockerfile.hive
    ports:
      - "9002:9000"
      - "2222:22"
    networks:
      rbee-test:
        ipv4_address: 172.20.0.21

networks:
  rbee-test:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

---

## Enhanced TestHarness

```rust
// xtask/src/integration/docker_harness.rs

pub struct DockerTestHarness {
    compose_file: PathBuf,
    containers: Vec<String>,
    test_id: String,
}

impl DockerTestHarness {
    pub async fn new(topology: Topology) -> Result<Self>;
    pub async fn exec(&self, container: &str, cmd: &[&str]) -> Result<CommandResult>;
    pub async fn logs(&self, container: &str) -> Result<String>;
    pub async fn restart(&self, container: &str) -> Result<()>;
    pub async fn kill(&self, container: &str) -> Result<()>;
    pub async fn block_network(&self, from: &str, to: &str) -> Result<()>;
    pub async fn restore_network(&self, from: &str, to: &str) -> Result<()>;
    pub async fn wait_for_healthy(&self, container: &str, timeout: Duration) -> Result<()>;
}
```

---

## Running Tests

### Quick Start
```bash
# 1. Setup (one-time)
./tests/docker/scripts/setup.sh

# 2. Run localhost tests
./tests/docker/scripts/run-tests.sh localhost

# 3. Run multi-hive tests
./tests/docker/scripts/run-tests.sh multi-hive

# 4. Cleanup
./tests/docker/scripts/cleanup.sh
```

### Individual Categories
```bash
cargo test --package xtask --test docker_http_communication_tests --ignored
cargo test --package xtask --test docker_ssh_communication_tests --ignored
cargo test --package xtask --test docker_heartbeat_tests --ignored
cargo test --package xtask --test docker_lifecycle_tests --ignored
cargo test --package xtask --test docker_worker_lifecycle_tests --ignored
cargo test --package xtask --test docker_failure_tests --ignored
cargo test --package xtask --test docker_e2e_tests --ignored
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)
- Create Dockerfiles (base, queen, hive, worker)
- Setup docker-compose files (3 topologies)
- Generate SSH keys
- Create DockerTestHarness

### Phase 2: Basic Tests (Week 2)
- HTTP communication tests (7 tests)
- SSH communication tests (6 tests)
- Heartbeat tests (5 tests)

### Phase 3: Advanced Tests (Week 3)
- Lifecycle tests (5 tests)
- Worker lifecycle tests (4 tests)
- Failure scenario tests (6 tests)

### Phase 4: E2E Tests (Week 4)
- End-to-end workflows (4 tests)
- CI/CD integration
- Documentation

---

## Expected Outcomes

1. **100% network communication coverage** - All queen ↔ hive patterns tested
2. **Isolated test environments** - No interference between tests
3. **Reproducible failures** - Network partitions, crashes, timeouts
4. **CI/CD ready** - Automated testing in GitHub Actions
5. **Production confidence** - Real SSH, real HTTP, real Docker networks

---

## Next Steps

1. Review and approve this plan
2. Create `tests/docker/` directory structure
3. Implement Phase 1 (Foundation)
4. Run first smoke test
5. Iterate through Phases 2-4
