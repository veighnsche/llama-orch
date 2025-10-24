# Docker SSH Integration Tests

**Status:** ✅ ARCHITECTURE FIXED  
**Created:** Oct 24, 2025  
**Team:** TEAM-282

---

## Overview

These tests verify that `queen-rbee` can deploy `rbee-hive` to a remote system via SSH, using `daemon-sync` for git clone + cargo build + installation.

**Correct Architecture:**
```
┌─────────────────┐
│  HOST MACHINE   │  ← You are here
│                 │
│  queen-rbee     │ ──SSH──> Docker Container
│  (bare metal)   │          (empty Arch Linux)
└─────────────────┘
```

**NOT this (old, wrong architecture):**
```
Container #1 (queen) ──SSH──> Container #2 (hive)
❌ This was deleted. It tested nothing useful.
```

---

## Files

### Container Infrastructure
- `Dockerfile.target` - Empty Arch Linux with SSH + Rust + Git (NO rbee binaries)
- `docker-compose.yml` - Single target container with SSH on port 2222
- `keys/test_id_rsa` - SSH private key for authentication
- `keys/test_id_rsa.pub` - SSH public key (installed in container)

### Configuration
- `hives.conf` - Configuration for HOST queen-rbee to connect to container

### Documentation
- `ARCHITECTURE_FIX.md` - Full explanation of why the old architecture was wrong
- `README.md` - This file

---

## Quick Start

### 1. Build and Start Target Container

```bash
cd tests/docker
docker-compose up -d
```

**Verifies:** Container with SSH is running on localhost:2222

### 2. Test SSH Connection from Host

```bash
ssh -i keys/test_id_rsa -p 2222 rbee@localhost
```

**Expected:** You should be able to SSH into the container  
**If it fails:** Check SSH keys are correct, container is running

### 3. Build queen-rbee on Host

```bash
cd /home/vince/Projects/llama-orch
cargo build --bin queen-rbee
```

**Why:** queen-rbee runs on HOST, not in container

### 4. Run queen-rbee on Host

```bash
export HIVES_CONF=tests/docker/hives.conf
cargo run --bin queen-rbee
```

**Verifies:** queen-rbee starts and reads hives.conf

### 5. Send Install Command

```bash
curl -X POST http://localhost:8500/v1/hives/install \
  -H "Content-Type: application/json" \
  -d '{"alias": "docker-test"}'
```

**Expected workflow:**
1. queen-rbee receives install request
2. daemon-sync SSHs to localhost:2222 (container)
3. daemon-sync runs `git clone` on container
4. daemon-sync runs `cargo build --bin rbee-hive` on container
5. daemon-sync installs binary to `~/.local/bin/rbee-hive`
6. daemon-sync starts rbee-hive daemon
7. Response: Installation successful

### 6. Verify Installation in Container

```bash
docker exec rbee-test-target test -f /home/rbee/.local/bin/rbee-hive && echo "✅ Binary installed"
docker exec rbee-test-target pgrep -f rbee-hive && echo "✅ Daemon running"
```

---

## What This Tests

### ✅ Real Product Workflow
- SSH from host to remote system
- Git clone on remote system
- Cargo build on remote system
- Binary installation via daemon-sync
- Daemon lifecycle management

### ✅ Real Deployment Scenarios
- SSH key authentication
- Build environment setup
- Dependency resolution
- Binary deployment
- Daemon startup

### ❌ Does NOT Test
- Container-to-container networking (irrelevant)
- Pre-copied binaries (not how it works)
- Docker Compose orchestration (not the product)

---

## Test Development

### Create New Test

1. Create test file: `xtask/tests/daemon_sync_integration.rs`
2. Test runs on HOST (not in container)
3. Test starts target container
4. Test runs queen-rbee on HOST
5. Test verifies installation in container

**Example:**
```rust
#[tokio::test]
#[ignore] // Run with: cargo test --ignored
async fn test_daemon_sync_install() {
    // 1. Start target container
    start_target_container().await;
    
    // 2. Build queen-rbee on HOST
    build_queen_rbee().await;
    
    // 3. Run queen-rbee on HOST
    let queen = start_queen_rbee().await;
    
    // 4. Send install command
    send_install_command("docker-test").await;
    
    // 5. Verify installation in container
    assert!(container_has_binary("/home/rbee/.local/bin/rbee-hive"));
    assert!(container_daemon_running("rbee-hive"));
}
```

---

## Troubleshooting

### SSH connection refused
```bash
# Check container is running
docker ps | grep rbee-test-target

# Check SSH is running in container
docker exec rbee-test-target pgrep sshd

# Check port mapping
docker port rbee-test-target
```

### Permission denied (publickey)
```bash
# Check SSH key permissions
ls -la tests/docker/keys/

# Should be:
# -rw------- test_id_rsa (600)
# -rw-r--r-- test_id_rsa.pub (644)

chmod 600 tests/docker/keys/test_id_rsa
chmod 644 tests/docker/keys/test_id_rsa.pub
```

### Git clone fails in container
```bash
# SSH into container
ssh -i tests/docker/keys/test_id_rsa -p 2222 rbee@localhost

# Check git is installed
which git

# Check network connectivity
ping -c 3 github.com
```

### Cargo build fails in container
```bash
# SSH into container
ssh -i tests/docker/keys/test_id_rsa -p 2222 rbee@localhost

# Check Rust toolchain
rustc --version
cargo --version

# Check disk space
df -h
```

---

## Architecture Decision

See `ARCHITECTURE_FIX.md` for full explanation of why the old architecture was wrong and how this fixes it.

**TL;DR:**
- **Old:** Container → SSH → Container (tested Docker networking, useless)
- **New:** Host → SSH → Container (tests actual deployment, valuable)
