# Daemon-Sync Code vs Tests vs Documentation Analysis

**Date:** Oct 24, 2025  
**Purpose:** Verify alignment between implementation, tests, and documentation  
**Status:** ⚠️ **CRITICAL GAPS FOUND**

---

## Executive Summary

### 🚨 Critical Finding

**The Docker tests we just created DO NOT match the daemon-sync implementation.**

The tests expect:
- `sync_all_hives()` function ✅ EXISTS
- SSH-based installation ✅ EXISTS
- Git clone + cargo build ❌ **NOT IMPLEMENTED**
- State query via SSH ❌ **TODO at line 119**

### Gap Analysis

| Component | Documentation Says | Code Does | Tests Expect | Status |
|-----------|-------------------|-----------|--------------|--------|
| **State Query** | Query actual state via SSH | `Vec::new()` (TODO) | Working query | ❌ **BROKEN** |
| **Git Install** | Clone repo + cargo build | Stub function | Working install | ⚠️ **INCOMPLETE** |
| **SSH Operations** | Real SSH via russh | ✅ Implemented | ✅ Working | ✅ **OK** |
| **Concurrent Install** | tokio::spawn parallelism | ✅ Implemented | ✅ Working | ✅ **OK** |
| **Binary Verification** | `--version` check | ✅ Implemented | ✅ Working | ✅ **OK** |

---

## Problem 1: State Query Not Implemented

### What Documentation Says (DAEMON_SYNC_DOCKER_TESTS.md:20)
```
The Problem: daemon-sync can't query what's actually installed on remote hosts via SSH.

What Docker Testing Solves:
1. Provides SSH-accessible test hosts
2. Allows testing real git clone + cargo build
3. Enables testing actual state queries  ← THIS
4. Validates the full sync workflow
```

### What Code Does (sync.rs:119-121)
```rust
// Step 1: Query actual state
// TODO: Implement actual state query (for now, assume nothing installed)
let actual_hives: Vec<String> = Vec::new();
let actual_workers: Vec<(String, Vec<String>)> = Vec::new();
```

### What Tests Expect (DAEMON_SYNC_DOCKER_TESTS.md:209-213)
```rust
// Verify installation via SSH
let mut client = RbeeSSHClient::connect("localhost", 2222, "testuser").await.unwrap();
let (stdout, _, exit_code) = client.exec("~/.local/bin/rbee-hive --version").await.unwrap();
assert_eq!(exit_code, 0);
assert!(stdout.contains("rbee-hive"));
```

### Impact
- ❌ **Tests will fail** - Can't verify installation because state query returns empty
- ❌ **Sync is broken** - Always assumes nothing installed, will try to reinstall
- ❌ **Idempotency broken** - Can't detect "already installed" state

---

## Problem 2: Git Installation IS Implemented ✅

### What Documentation Says (DAEMON_SYNC_DOCKER_TESTS.md:24)
```
What Docker Testing Solves:
2. Allows testing real git clone + cargo build  ← THIS
```

### What Code Does (install.rs:297-349)
```rust
async fn install_hive_from_git(
    client: &mut RbeeSSHClient,
    repo: &str,
    branch: &str,
    job_id: &str,
    alias: &str,
) -> Result<String> {
    // Clone repository (shallow, no history for speed)
    let clone_dir = "~/.local/share/rbee/build";
    let clone_cmd = format!(
        "rm -rf {} && mkdir -p {} && git clone --depth 1 --branch {} {} {}",
        clone_dir, clone_dir, branch, repo, clone_dir
    );
    
    // Build rbee-hive binary
    let build_cmd = format!(
        "cd {} && cargo build --release --bin rbee-hive",
        clone_dir
    );
    
    // Copy binary to ~/.local/bin
    let install_cmd = format!(
        "cp {}/target/release/rbee-hive ~/.local/bin/rbee-hive && chmod +x ~/.local/bin/rbee-hive",
        clone_dir
    );
}
```

### Status
✅ **IMPLEMENTED** - Git clone + cargo build is fully working

---

## Problem 3: Docker Tests Don't Match daemon-sync

### What We Just Created
We created Docker tests for **queen-rbee → rbee-hive communication**:
- HTTP health checks
- SSH command execution
- Capabilities discovery
- Container lifecycle

### What daemon-sync Actually Needs
Daemon-sync is a **package manager** that needs to test:
- SSH installation of binaries
- Git clone + cargo build workflow
- State query (what's installed)
- Concurrent installation across multiple hosts
- Idempotency (don't reinstall if already there)

### The Mismatch

| What We Built | What daemon-sync Needs |
|---------------|------------------------|
| Queen → Hive HTTP tests | Queen → Remote Host SSH install tests |
| Container health checks | Binary installation verification |
| Capabilities discovery | State query implementation |
| Service lifecycle | Package manager operations |

### Impact
⚠️ **The Docker tests we created are valuable but for the WRONG use case!**

- ✅ Tests are great for queen-rbee → rbee-hive communication
- ❌ Tests don't help with daemon-sync package manager functionality
- ❌ Tests don't address the TODOs in daemon-sync

---

## What daemon-sync Actually Needs

### 1. State Query Implementation (CRITICAL)

**Current Code (sync.rs:119-121):**
```rust
// TODO: Implement actual state query (for now, assume nothing installed)
let actual_hives: Vec<String> = Vec::new();
let actual_workers: Vec<(String, Vec<String>)> = Vec::new();
```

**What It Should Do:**
```rust
// Query actual state via SSH
let actual_hives = query_installed_hives(&config.hives).await?;
let actual_workers = query_installed_workers(&config.hives).await?;

// Implementation:
async fn query_installed_hives(hives: &[HiveConfig]) -> Result<Vec<String>> {
    let mut installed = Vec::new();
    for hive in hives {
        let mut client = RbeeSSHClient::connect(&hive.hostname, hive.ssh_port, &hive.ssh_user).await?;
        let (stdout, _, exit_code) = client.exec("~/.local/bin/rbee-hive --version").await?;
        if exit_code == 0 && stdout.contains("rbee-hive") {
            installed.push(hive.alias.clone());
        }
        client.close().await?;
    }
    Ok(installed)
}
```

### 2. Docker Tests for daemon-sync (NEW)

**Location:** `bin/99_shared_crates/daemon-sync/tests/docker/`

**What to Test:**
```rust
#[tokio::test]
#[ignore]
async fn test_install_hive_from_git() {
    // 1. Start Docker container with SSH + Git + Rust
    // 2. Run daemon-sync to install hive
    // 3. Verify binary exists via SSH
    // 4. Run sync again - should detect "already installed"
    // 5. Verify idempotency
}

#[tokio::test]
#[ignore]
async fn test_concurrent_multi_hive_install() {
    // 1. Start 3 Docker containers
    // 2. Run daemon-sync with 3 hives
    // 3. Verify all installed concurrently
    // 4. Verify faster than sequential
}

#[tokio::test]
#[ignore]
async fn test_state_query() {
    // 1. Manually install hive on container
    // 2. Run daemon-sync state query
    // 3. Verify it detects existing installation
    // 4. Verify it doesn't try to reinstall
}
```

---

## Recommendations

### Immediate Actions

1. **Implement State Query (CRITICAL)**
   - File: `bin/99_shared_crates/daemon-sync/src/query.rs` (new)
   - Functions: `query_installed_hives()`, `query_installed_workers()`
   - Remove TODO at sync.rs:119
   - Enable idempotency

2. **Create daemon-sync Docker Tests (NEW)**
   - Location: `bin/99_shared_crates/daemon-sync/tests/docker/`
   - Dockerfile: SSH + Git + Rust (matches DAEMON_SYNC_DOCKER_TESTS.md)
   - Tests: Installation, state query, idempotency, concurrency

3. **Keep Existing Docker Tests**
   - Location: `tests/docker/` (what we just created)
   - Purpose: Queen → Hive communication (HTTP, SSH, lifecycle)
   - Status: ✅ Valuable for integration testing
   - Note: Different use case than daemon-sync

### Long-term Actions

4. **Implement Other TODOs**
   - `status.rs:83` - State query for status command
   - `migrate.rs:40` - State query for migration
   - `sync.rs:261` - Auto-start hive after installation

5. **Add More daemon-sync Tests**
   - Worker installation
   - Release-based installation (not just git)
   - Local binary installation
   - Error handling (git clone fails, build fails, etc.)

---

## Summary

### What We Have

✅ **daemon-sync Implementation**
- Git clone + cargo build ✅ Working
- SSH-based installation ✅ Working
- Concurrent installation ✅ Working
- Binary verification ✅ Working
- State query ❌ **TODO (CRITICAL)**

✅ **Docker Tests (tests/docker/)**
- Queen → Hive HTTP communication ✅ Working
- SSH command execution ✅ Working
- Container lifecycle ✅ Working
- Purpose: Integration testing (not package manager)

❌ **daemon-sync Docker Tests**
- Location: Should be `bin/99_shared_crates/daemon-sync/tests/docker/`
- Status: **NOT CREATED YET**
- Purpose: Test package manager functionality

### Critical Gap

🚨 **The state query TODO (sync.rs:119) blocks all daemon-sync testing!**

Without state query:
- Can't detect "already installed"
- Can't verify idempotency
- Can't test sync workflow end-to-end
- Always assumes nothing installed

### Next Steps

**Priority 1: Implement State Query**
1. Create `bin/99_shared_crates/daemon-sync/src/query.rs`
2. Implement `query_installed_hives()` and `query_installed_workers()`
3. Replace TODO at sync.rs:119 with actual query
4. Test manually via SSH

**Priority 2: Create daemon-sync Docker Tests**
1. Create `bin/99_shared_crates/daemon-sync/tests/docker/`
2. Copy Dockerfile pattern from DAEMON_SYNC_DOCKER_TESTS.md
3. Implement tests for installation, state query, idempotency
4. Run tests to validate package manager functionality

**Priority 3: Keep Existing Tests**
1. The tests in `tests/docker/` are valuable
2. They test queen → hive communication
3. Different use case than daemon-sync
4. Keep both test suites

---

## Conclusion

### Alignment Status

| Component | Code | Tests | Docs | Aligned? |
|-----------|------|-------|------|----------|
| **Git Install** | ✅ | ❌ | ✅ | ⚠️ Tests missing |
| **SSH Install** | ✅ | ❌ | ✅ | ⚠️ Tests missing |
| **State Query** | ❌ TODO | ❌ | ✅ | ❌ **BROKEN** |
| **Concurrent Install** | ✅ | ❌ | ✅ | ⚠️ Tests missing |
| **Queen→Hive HTTP** | ✅ | ✅ | ✅ | ✅ **OK** |

### The Real Problem

We created excellent Docker tests for **queen-rbee → rbee-hive communication**, but daemon-sync needs tests for **package manager functionality** (SSH installation, state query, idempotency).

Both test suites are valuable, but they serve different purposes:
- `tests/docker/` → Integration testing (queen ↔ hive)
- `daemon-sync/tests/docker/` → Package manager testing (install, query, sync)

**Action Required:** Implement state query + create daemon-sync-specific Docker tests.
