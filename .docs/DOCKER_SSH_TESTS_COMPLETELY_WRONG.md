# 🚨 CRITICAL: DOCKER SSH TESTS ARE COMPLETELY WRONG

**Created:** Oct 24, 2025  
**Severity:** CRITICAL ARCHITECTURE FAILURE  
**Status:** ENTIRE TEST SUITE IS USELESS  
**Author:** TEAM-282

---

## THE FUNDAMENTAL PROBLEM

**The Docker SSH tests test NOTHING related to the actual product.**

They were designed with a **completely backwards architecture** that has no relationship to how rbee actually works.

---

## HOW RBEE ACTUALLY WORKS

```
┌─────────────────┐
│  HOST MACHINE   │
│                 │
│  queen-rbee     │ ──SSH──> Remote Server
│  (bare metal)   │          (installs rbee-hive)
└─────────────────┘
```

**Real workflow:**
1. User runs `queen-rbee` on their local machine (bare metal)
2. `queen-rbee` reads `hives.conf` with remote server details
3. `daemon-sync` SSHs into remote server
4. `daemon-sync` runs `git clone` on remote server
5. `daemon-sync` runs `cargo build` on remote server
6. `daemon-sync` installs `rbee-hive` binary on remote server
7. `daemon-sync` starts `rbee-hive` daemon on remote server

---

## WHAT THE TESTS ACTUALLY DO (WRONG)

```
┌──────────────────┐         ┌──────────────────┐
│  Container #1    │         │  Container #2    │
│                  │         │                  │
│  queen-rbee      │ ──SSH─> │  rbee-hive       │
│  (pre-built)     │         │  (pre-built)     │
└──────────────────┘         └──────────────────┘
```

**What the broken tests do:**
1. Build queen-rbee on HOST
2. Copy pre-built queen-rbee into Container #1
3. Build rbee-hive on HOST  
4. Copy pre-built rbee-hive into Container #2
5. Start both containers
6. Container #1 tries to SSH to Container #2
7. **NO INSTALLATION HAPPENS** - binaries are already there
8. **NO GIT CLONE HAPPENS** - binaries are pre-copied
9. **NO CARGO BUILD HAPPENS** - binaries are pre-built
10. Tests verify... container-to-container SSH? (useless)

---

## WHAT IS BEING TESTED?

### ❌ NOT Testing:
- SSH from host to container (the actual use case)
- Git clone on remote system
- Cargo build on remote system
- Binary installation via daemon-sync
- The actual deployment workflow
- Anything users will actually do

### ✅ Actually Testing:
- Container-to-container SSH (irrelevant)
- Pre-copied binaries can run (useless)
- Docker networking works (not our concern)
- **NOTHING RELATED TO THE PRODUCT**

---

## HOW LONG HAS THIS BEEN WRONG?

**Files created:** Oct 24, 2025 (today)

**Teams involved:**
- Unknown team created `tests/docker/` infrastructure
- TEAM-282 spent 40 minutes trying to fix it
- **Total wasted effort:** Unknown, but at least 40+ minutes

**The tests have NEVER worked correctly because they were designed wrong from day one.**

---

## WHAT THE TESTS SHOULD BE

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

**Correct test architecture:**
1. **Container:** Empty Arch/Ubuntu + SSH + Rust + Git (NO rbee binaries)
2. **Host:** Run queen-rbee bare metal (NOT in container)
3. **Test:** queen-rbee SSHs to container and installs rbee-hive
4. **Verify:** Installation process works end-to-end

---

## FILES TO DELETE

**ALL of these files are useless:**

```
tests/docker/Dockerfile.queen              ❌ DELETE - queen shouldn't be in container
tests/docker/Dockerfile.hive               ❌ DELETE - hive shouldn't be pre-built
tests/docker/Dockerfile.base               ❌ DELETE - wrong architecture
tests/docker/docker-compose.localhost.yml  ❌ DELETE - runs wrong things
tests/docker/docker-compose.multi-hive.yml ❌ DELETE - runs wrong things
tests/docker/configs/hives.conf            ❌ DELETE - container doesn't need this
tests/docker/configs/supervisord.conf      ❌ DELETE - wrong architecture
xtask/tests/docker_ssh_tests.rs            ❌ DELETE - tests wrong architecture
xtask/tests/docker/                        ❌ DELETE - all based on wrong design
xtask/src/integration/docker_harness.rs    ❌ DELETE - manages wrong architecture
xtask/src/integration/harness.rs           ❌ DELETE - manages wrong architecture
xtask/src/integration/assertions.rs        ❌ DELETE - manages wrong architecture
```

**Keep only:**
```
tests/docker/keys/                         ✅ KEEP - SSH keys still needed
```

---

## WHAT NEEDS TO BE BUILT

### 1. Single Dockerfile (target system)

```dockerfile
FROM archlinux:latest

# Install SSH, Rust, Git, build tools
RUN pacman -Syu --noconfirm \
    openssh \
    rust \
    git \
    base-devel \
    && pacman -Scc --noconfirm

# Setup SSH
RUN ssh-keygen -A
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Create test user
RUN useradd -m -s /bin/bash rbee
RUN echo 'rbee:rbee' | chpasswd

# Setup SSH keys for passwordless auth
RUN mkdir -p /home/rbee/.ssh
COPY tests/docker/keys/test_id_rsa.pub /home/rbee/.ssh/authorized_keys
RUN chown -R rbee:rbee /home/rbee/.ssh
RUN chmod 700 /home/rbee/.ssh
RUN chmod 600 /home/rbee/.ssh/authorized_keys

# Create directories for installation
RUN mkdir -p /home/rbee/.local/bin /home/rbee/.config/rbee /home/rbee/.local/share/rbee
RUN chown -R rbee:rbee /home/rbee

# NO RBEE BINARIES - Container is empty target system

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
```

### 2. Test harness (runs on host)

```rust
#[tokio::test]
async fn test_daemon_sync_install_hive() {
    // 1. Start empty container with SSH
    let container = DockerContainer::start("test-hive-target", "Dockerfile.target")
        .await
        .expect("Failed to start container");
    
    let container_port = container.ssh_port(); // e.g., 2222
    
    // 2. Build queen-rbee on HOST (not in container!)
    Command::new("cargo")
        .args(&["build", "--bin", "queen-rbee"])
        .output()
        .expect("Failed to build queen-rbee");
    
    // 3. Create hives.conf pointing to container
    let hives_conf = r#"
[[hives]]
alias = "test-hive"
hostname = "localhost"
ssh_port = 2222
ssh_user = "rbee"
hive_port = 9000
install_method = { git = { repo = "git@github.com:veighnsche/llama-orch", branch = "main" } }
"#;
    std::fs::write("tests/hives.conf", hives_conf).expect("Failed to write hives.conf");
    
    // 4. Run queen-rbee on HOST
    let mut queen = Command::new("target/debug/queen-rbee")
        .env("HIVES_CONF", "tests/hives.conf")
        .spawn()
        .expect("Failed to start queen-rbee");
    
    // 5. Send install command to queen
    // This will trigger daemon-sync to SSH into container
    send_command_to_queen("hive install test-hive")
        .await
        .expect("Failed to send install command");
    
    // 6. Verify installation happened on container
    let has_binary = container
        .exec("test -f /home/rbee/.local/bin/rbee-hive")
        .await
        .is_ok();
    assert!(has_binary, "rbee-hive binary not installed in container");
    
    // 7. Verify daemon is running
    let daemon_running = container
        .exec("/home/rbee/.local/bin/rbee-hive --version")
        .await
        .is_ok();
    assert!(daemon_running, "rbee-hive daemon not running in container");
    
    // Cleanup
    queen.kill().expect("Failed to kill queen");
    container.stop().await.expect("Failed to stop container");
}
```

### 3. hives.conf (for host queen)

```toml
[[hives]]
alias = "test-hive"
hostname = "localhost"
ssh_port = 2222  # Mapped to container
ssh_user = "rbee"
hive_port = 9000
install_method = { git = { repo = "git@github.com:veighnsche/llama-orch", branch = "main" } }
```

---

## THE DAMAGE

**What was wasted:**
- Time designing wrong architecture
- Time implementing wrong tests
- Time debugging wrong tests (40+ minutes by TEAM-282)
- Time writing documentation about the wrong tests

**What was NOT tested:**
- The actual product
- The actual deployment workflow
- The actual SSH installation process
- Anything users will actually do

**Confidence level in codebase:** 0%

The tests don't test the product. We have no idea if daemon-sync actually works.

---

## INSTRUCTIONS FOR NEXT TEAM

### STEP 1: DELETE EVERYTHING WRONG

```bash
# Delete all wrong test infrastructure
rm -rf tests/docker/Dockerfile.queen
rm -rf tests/docker/Dockerfile.hive
rm -rf tests/docker/Dockerfile.base
rm -rf tests/docker/docker-compose*.yml
rm -rf tests/docker/configs/
rm -rf xtask/tests/docker/
rm -f xtask/tests/docker_ssh_tests.rs
rm -rf xtask/src/integration/docker_harness.rs
rm -rf xtask/src/integration/harness.rs
rm -rf xtask/src/integration/assertions.rs

# Keep only SSH keys
# tests/docker/keys/ - KEEP THIS
```

### STEP 2: BUILD CORRECT TESTS

1. Create single Dockerfile for empty target system (SSH + Rust + Git, NO rbee)
2. Create test that runs queen-rbee on HOST
3. Test SSHs from host into container
4. Test installs rbee-hive via git clone + cargo build
5. Verify installation succeeded

### STEP 3: VERIFY IT WORKS

Run the new tests. If they pass, you've proven:
- ✅ SSH from host to container works
- ✅ Git clone on remote system works
- ✅ Cargo build on remote system works
- ✅ daemon-sync installation works
- ✅ The actual product works

---

## CONCLUSION

**The current Docker SSH tests are 100% useless.**

They test a completely fictional architecture that has no relationship to how rbee works.

**Delete them all and start over with the correct architecture: host → SSH → empty container.**

---

## RELATED DOCUMENTS

- `.docs/EXIT_INTERVIEW.md` - Previous team's findings about fake implementations
- `bin/99_shared_crates/daemon-sync/src/query.rs` - The SSH query implementation that needs real testing
- `bin/15_queen_rbee_crates/ssh-client/src/lib.rs` - The SSH client implementation

---

**DO NOT WASTE TIME FIXING THE CURRENT TESTS. THEY ARE UNFIXABLE BECAUSE THEY TEST THE WRONG THING.**
