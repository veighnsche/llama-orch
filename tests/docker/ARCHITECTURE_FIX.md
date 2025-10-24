# Docker SSH Test Architecture Fix

**Status:** ✅ CLEANUP COMPLETE  
**Created:** Oct 24, 2025  
**Team:** TEAM-282

---

## THE PROBLEM

The original Docker SSH tests had a **fundamentally wrong architecture** that tested nothing related to the actual product.

### Wrong Architecture (DELETED)
```
┌──────────────────┐         ┌──────────────────┐
│  Container #1    │         │  Container #2    │
│                  │         │                  │
│  queen-rbee      │ ──SSH─> │  rbee-hive       │
│  (pre-built)     │         │  (pre-built)     │
└──────────────────┘         └──────────────────┘
```

**What it tested:**
- ❌ Container-to-container SSH (irrelevant)
- ❌ Pre-copied binaries can run (useless)
- ❌ Docker networking works (not our concern)
- ❌ **NOTHING RELATED TO ACTUAL DEPLOYMENT**

**What it DIDN'T test:**
- ❌ SSH from host to remote system
- ❌ Git clone on remote system
- ❌ Cargo build on remote system
- ❌ Binary installation via daemon-sync
- ❌ The actual product workflow

---

## CORRECT ARCHITECTURE

```
┌─────────────────────────────┐
│  HOST MACHINE               │
│                             │
│  1. Run queen-rbee          │
│     (cargo run)             │
│                             │
│  2. queen-rbee reads        │
│     hives.conf:             │
│     - hostname: localhost   │
│     - port: 2222            │
│                             │
│  3. daemon-sync SSHs to ────┼──> ┌──────────────────────┐
│     localhost:2222          │    │  DOCKER CONTAINER    │
│                             │    │                      │
│  4. daemon-sync runs:       │    │  Empty Arch Linux    │
│     - git clone             │    │  + SSH server        │
│     - cargo build           │    │  + Rust toolchain    │
│     - install binary        │    │  + Git               │
│     - start daemon          │    │  + Build tools       │
│                             │    │                      │
│  5. Tests verify:           │    │  (NO PRE-BUILT       │
│     - SSH connection works  │    │   BINARIES)          │
│     - Git clone succeeded   │    │                      │
│     - Build succeeded       │    └──────────────────────┘
│     - Binary installed      │
│     - Daemon started        │
└─────────────────────────────┘
```

### What This Tests

**✅ Actual product workflow:**
- Host queen-rbee → SSH → Remote container
- daemon-sync git clone on remote system
- daemon-sync cargo build on remote system
- Binary installation via daemon-sync
- Daemon lifecycle management

**✅ Real deployment scenarios:**
- SSH key authentication
- Build environment setup
- Dependency resolution
- Binary deployment
- Daemon startup

---

## FILES DELETED

```
❌ tests/docker/Dockerfile.queen           - Queen shouldn't be in container
❌ tests/docker/Dockerfile.hive            - Hive shouldn't be pre-built
❌ tests/docker/Dockerfile.base            - Wrong architecture
❌ tests/docker/docker-compose.localhost.yml - Runs wrong things
❌ tests/docker/docker-compose.multi-hive.yml - Runs wrong things
❌ tests/docker/configs/                   - Container doesn't need queen config
❌ tests/docker/scripts/                   - Scripts for wrong architecture
❌ tests/docker/.dockerignore              - Wrong build context
❌ tests/docker/README.md                  - Documents wrong architecture
❌ tests/docker/IMPLEMENTATION_COMPLETE.md - Wrong implementation
❌ xtask/tests/docker/                     - All wrong tests
❌ xtask/tests/docker_ssh_tests.rs         - Test entry point for wrong tests
❌ xtask/src/integration/docker_harness.rs - Harness for wrong architecture
```

### Files PRESERVED

```
✅ tests/docker/keys/test_id_rsa           - SSH private key (still needed)
✅ tests/docker/keys/test_id_rsa.pub       - SSH public key (still needed)
```

---

## IMPLEMENTATION GUIDE

### Phase 1: Container Setup (Empty Target System)

**File:** `tests/docker/Dockerfile.target`

```dockerfile
FROM archlinux:latest

# Install SSH server, Rust, Git, build tools
RUN pacman -Syu --noconfirm \
    openssh \
    rust \
    git \
    base-devel \
    && pacman -Scc --noconfirm

# Setup SSH
RUN ssh-keygen -A
RUN useradd -m -s /bin/bash rbee

# Setup SSH keys for passwordless auth
RUN mkdir -p /home/rbee/.ssh
COPY tests/docker/keys/test_id_rsa.pub /home/rbee/.ssh/authorized_keys
RUN chown -R rbee:rbee /home/rbee/.ssh
RUN chmod 700 /home/rbee/.ssh
RUN chmod 600 /home/rbee/.ssh/authorized_keys

# NO RBEE BINARIES
# Container is empty target system

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
```

### Phase 2: hives.conf (For Host queen-rbee)

**File:** `tests/docker/hives.conf`

```toml
[[hives]]
alias = "docker-test"
hostname = "localhost"
ssh_port = 2222  # Mapped to container SSH
ssh_user = "rbee"
hive_port = 9000
install_method = { git = { repo = "https://github.com/YOUR_ORG/llama-orch.git", branch = "main" } }
```

### Phase 3: Test Harness (Runs on Host)

**File:** `xtask/tests/daemon_sync_integration.rs`

```rust
// TEAM-282+: Daemon sync integration tests
// Tests SSH from HOST to CONTAINER (correct architecture)

use std::process::Command;
use tokio::process::Command as AsyncCommand;

/// Test helper: Start empty container with SSH
async fn start_target_container() -> Container {
    let container = AsyncCommand::new("docker")
        .args(&[
            "run", "-d",
            "-p", "2222:22",
            "--name", "rbee-test-target",
            "rbee-test-target:latest"
        ])
        .output()
        .await
        .expect("Failed to start container");
    
    // Wait for SSH to be ready
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    
    Container { id: String::from_utf8_lossy(&container.stdout).trim().to_string() }
}

#[tokio::test]
#[ignore] // Run with: cargo test --ignored
async fn test_daemon_sync_install_from_host() {
    // 1. Start empty target container with SSH
    let container = start_target_container().await;
    
    // 2. Build queen-rbee on HOST
    let build = Command::new("cargo")
        .args(&["build", "--bin", "queen-rbee"])
        .output()
        .expect("Failed to build queen-rbee");
    assert!(build.status.success(), "queen-rbee build failed");
    
    // 3. Run queen-rbee on HOST (not in container!)
    let queen = AsyncCommand::new("target/debug/queen-rbee")
        .env("HIVES_CONF", "tests/docker/hives.conf")
        .spawn()
        .expect("Failed to start queen-rbee");
    
    // Wait for queen to start
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    
    // 4. Send install command to queen
    // queen uses daemon-sync to SSH into container
    let install = AsyncCommand::new("curl")
        .args(&[
            "-X", "POST",
            "http://localhost:8500/v1/hives/install",
            "-H", "Content-Type: application/json",
            "-d", r#"{"alias": "docker-test"}"#
        ])
        .output()
        .await
        .expect("Failed to send install command");
    
    assert!(install.status.success(), "Install command failed");
    
    // 5. Verify installation happened IN THE CONTAINER
    let verify = AsyncCommand::new("docker")
        .args(&[
            "exec", "rbee-test-target",
            "test", "-f", "/home/rbee/.local/bin/rbee-hive"
        ])
        .output()
        .await
        .expect("Failed to verify installation");
    
    assert!(verify.status.success(), "rbee-hive binary not found in container");
    
    // 6. Verify daemon is running
    let verify_daemon = AsyncCommand::new("docker")
        .args(&[
            "exec", "rbee-test-target",
            "pgrep", "-f", "rbee-hive"
        ])
        .output()
        .await
        .expect("Failed to verify daemon");
    
    assert!(verify_daemon.status.success(), "rbee-hive daemon not running");
    
    // Cleanup
    container.cleanup().await;
}

struct Container {
    id: String,
}

impl Container {
    async fn cleanup(&self) {
        let _ = AsyncCommand::new("docker")
            .args(&["rm", "-f", &self.id])
            .output()
            .await;
    }
}
```

---

## VERIFICATION CHECKLIST

### ✅ Architecture Fixed
- [x] Deleted all container-to-container test infrastructure
- [x] Deleted pre-built binary Dockerfiles
- [x] Preserved SSH keys for host-to-container auth

### 🔨 TODO: Build Correct Tests
- [ ] Create Dockerfile.target (empty system)
- [ ] Create hives.conf (host config)
- [ ] Create test harness (host → container)
- [ ] Implement daemon-sync install test
- [ ] Verify SSH connection works
- [ ] Verify git clone works
- [ ] Verify cargo build works
- [ ] Verify binary installation works
- [ ] Verify daemon startup works

---

## KEY DIFFERENCES

| Aspect | Wrong (Deleted) | Correct (To Build) |
|--------|----------------|-------------------|
| **queen-rbee location** | In container | On host (bare metal) |
| **rbee-hive binary** | Pre-built, copied in | Built on container via daemon-sync |
| **SSH direction** | Container → Container | Host → Container |
| **What's tested** | Docker networking | Actual deployment workflow |
| **Git clone** | Never happens | Happens on container |
| **Cargo build** | Never happens | Happens on container |
| **Value** | Zero | High (tests real workflow) |

---

## NEXT STEPS

1. **Create Dockerfile.target** - Empty Arch Linux with SSH + Rust + Git
2. **Create docker-compose.yml** - Single service for target container
3. **Create hives.conf** - Host configuration for localhost:2222
4. **Create test harness** - Host-based tests that verify full workflow
5. **Run tests** - Verify SSH, git clone, build, install, daemon start all work

---

## SUCCESS CRITERIA

**When tests pass, you've proven:**
- ✅ SSH from host to container works
- ✅ SSH key authentication works
- ✅ Git clone on remote system works
- ✅ Rust build environment works
- ✅ Cargo build on remote system works
- ✅ Binary installation via daemon-sync works
- ✅ Daemon lifecycle management works
- ✅ **The actual product works**

---

## NOTES

**This is the correct architecture for testing rbee deployment.**

Users run queen-rbee on their local machine (host) and it deploys rbee-hive to remote systems via SSH. The tests must mirror this workflow: host → SSH → container.

Any test that puts queen-rbee in a container is fundamentally wrong and must be deleted.
