# Daemon-Sync Docker Testing Plan

**Date:** Oct 24, 2025  
**Crate:** `bin/99_shared_crates/daemon-sync`  
**Purpose:** Docker-based testing for SSH installation operations

---

## Context

The `daemon-sync` crate has **3 critical TODOs** that block testing:

```rust
// src/sync.rs:119
// TODO: Implement actual state query (for now, assume nothing installed)
let actual_hives: Vec<String> = Vec::new();
let actual_workers: Vec<(String, Vec<String>)> = Vec::new();
```

**The Problem:** daemon-sync can't query what's actually installed on remote hosts via SSH.

**What Docker Testing Solves:**
1. Provides SSH-accessible test hosts
2. Allows testing real git clone + cargo build
3. Enables testing actual state queries
4. Validates the full sync workflow

---

## Existing Infrastructure to Leverage

### 1. SSH Client Tests (Already Exists!)
- `bin/15_queen_rbee_crates/ssh-client/tests/ssh_connection_tests.rs` (348 LOC)
- Tests SSH connectivity, authentication, command execution
- Uses `russh` library (not system `ssh` binary)

### 2. Integration Test Harness (Already Exists!)
- `xtask/src/integration/harness.rs` - TestHarness for spawning binaries
- `xtask/src/integration/assertions.rs` - Helper assertions
- Pattern: Spawn binaries, capture output, validate state

### 3. Existing Test Pattern
From `TESTING_PACKAGE_MANAGER.md` (lines 139-258):
- Dockerfile with SSH + Git + Rust
- docker-compose with multiple hosts
- Test against localhost:2222, localhost:2223, etc.

---

## Docker Infrastructure for daemon-sync

### 1. Minimal Dockerfile (Based on TESTING_PACKAGE_MANAGER.md)

```dockerfile
# bin/99_shared_crates/daemon-sync/tests/docker/Dockerfile.test-host

FROM rust:1.75

# Install SSH server and git
RUN apt-get update && apt-get install -y openssh-server git

# Setup SSH
RUN mkdir /var/run/sshd
RUN echo 'root:testpassword' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Create test user
RUN useradd -m -s /bin/bash testuser
RUN echo 'testuser:testpassword' | chpasswd

# Setup SSH keys
RUN mkdir -p /home/testuser/.ssh
COPY test_id_rsa.pub /home/testuser/.ssh/authorized_keys
RUN chown -R testuser:testuser /home/testuser/.ssh
RUN chmod 700 /home/testuser/.ssh
RUN chmod 600 /home/testuser/.ssh/authorized_keys

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
```

**Key Points:**
- Uses official `rust:1.75` image (includes Rust + Cargo)
- Minimal setup - only SSH + Git
- Matches pattern from `TESTING_PACKAGE_MANAGER.md`

### 2. Docker Compose (Simplified from TESTING_PACKAGE_MANAGER.md)

```yaml
# bin/99_shared_crates/daemon-sync/tests/docker/docker-compose.yml

version: '3.8'

services:
  test-host-1:
    build:
      context: .
      dockerfile: Dockerfile.test-host
    container_name: rbee-test-host-1
    ports:
      - "2222:22"
    networks:
      - rbee-test

  test-host-2:
    build:
      context: .
      dockerfile: Dockerfile.test-host
    container_name: rbee-test-host-2
    ports:
      - "2223:22"
    networks:
      - rbee-test

networks:
  rbee-test:
    driver: bridge
```

**Simplified from original** - Removed unnecessary complexity:
- No custom IP addresses
- No health checks (tests will wait/retry)
- No volumes (not needed for basic testing)
- Matches `TESTING_PACKAGE_MANAGER.md` pattern

### 3. SSH Key Setup (One-time)

```bash
#!/bin/bash
# bin/99_shared_crates/daemon-sync/tests/docker/setup-keys.sh

cd "$(dirname "$0")"
ssh-keygen -t ed25519 -f test_id_rsa -N "" -C "daemon-sync-tests"
chmod 600 test_id_rsa
chmod 644 test_id_rsa.pub
echo "✅ SSH keys generated"
```

### 4. Start/Stop Scripts

```bash
#!/bin/bash
# bin/99_shared_crates/daemon-sync/tests/docker/start.sh

cd "$(dirname "$0")"
[ ! -f "test_id_rsa" ] && ./setup-keys.sh
docker-compose up -d
sleep 5  # Wait for SSH
echo "✅ Containers ready: localhost:2222, localhost:2223"
```

```bash
#!/bin/bash
# bin/99_shared_crates/daemon-sync/tests/docker/stop.sh

cd "$(dirname "$0")"
docker-compose down -v
echo "✅ Containers stopped"
```

---

## Integration Tests

**Pattern:** Follow `TESTING_PACKAGE_MANAGER.md` lines 206-257

```rust
// bin/99_shared_crates/daemon-sync/tests/docker_tests.rs

use daemon_sync::*;
use rbee_config::declarative::*;
use queen_rbee_ssh_client::RbeeSSHClient;
use std::process::Command;
use tokio::time::{sleep, Duration};

#[tokio::test]
#[ignore] // Run with: cargo test --ignored docker_single_hive
async fn docker_single_hive_installation() {
    // Start containers (assumes start.sh already run)
    
    let config = HivesConfig {
        hives: vec![HiveConfig {
            alias: "test-host-1".to_string(),
            hostname: "localhost".to_string(),
            ssh_user: "testuser".to_string(),
            ssh_port: 2222,
            install_method: InstallMethod::Git {
                repo: "https://github.com/veighnsche/llama-orch".to_string(),
                branch: "main".to_string(),
            },
            workers: vec![],
            ..Default::default()
        }]
    };
    
    // Run sync
    let result = sync_all_hives(config, SyncOptions {
        dry_run: false,
        remove_extra: false,
        force: true,
    }, "test-job").await;
    
    assert!(result.is_ok());
    let report = result.unwrap();
    assert_eq!(report.hives_installed, 1);
    
    // Verify installation via SSH
    let mut client = RbeeSSHClient::connect("localhost", 2222, "testuser").await.unwrap();
    let (stdout, _, exit_code) = client.exec("~/.local/bin/rbee-hive --version").await.unwrap();
    assert_eq!(exit_code, 0);
    assert!(stdout.contains("rbee-hive"));
    client.close().await.unwrap();
}

#[tokio::test]
#[ignore]
async fn docker_multi_hive_concurrent() {
    // Test concurrent installation on 2 hosts
    let config = HivesConfig {
        hives: vec![
            HiveConfig {
                alias: "test-host-1".to_string(),
                hostname: "localhost".to_string(),
                ssh_port: 2222,
                ssh_user: "testuser".to_string(),
                ..Default::default()
            },
            HiveConfig {
                alias: "test-host-2".to_string(),
                hostname: "localhost".to_string(),
                ssh_port: 2223,
                ssh_user: "testuser".to_string(),
                ..Default::default()
            },
        ]
    };
    
    let start = std::time::Instant::now();
    let result = sync_all_hives(config, SyncOptions::default(), "test-job").await;
    let duration = start.elapsed();
    
    assert!(result.is_ok());
    let report = result.unwrap();
    assert_eq!(report.hives_installed, 2);
    
    println!("Installation took: {:?}", duration);
    // Should be faster than sequential (verify concurrency works)
}
```

---

## Running Tests

```bash
# 1. Setup (one-time)
cd bin/99_shared_crates/daemon-sync/tests/docker
./setup-keys.sh
./start.sh

# 2. Run tests
cargo test --package daemon-sync --test docker_tests --ignored

# 3. Cleanup
./stop.sh
```

---

## What This Tests

1. **Real SSH operations** - Connects to Docker containers via SSH
2. **Real git clone** - Clones actual repository
3. **Real cargo build** - Builds binaries from source
4. **Concurrent installation** - Tests tokio::spawn parallelism
5. **Actual state queries** - Once TODO is implemented

---

## Next Steps

1. **Implement TODO** in `src/sync.rs:119` - Query actual state via SSH
2. **Create test file** at `bin/99_shared_crates/daemon-sync/tests/docker_tests.rs`
3. **Create Docker files** in `bin/99_shared_crates/daemon-sync/tests/docker/`
4. **Run tests** to validate SSH installation workflow
