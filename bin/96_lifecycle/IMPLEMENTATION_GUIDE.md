# 96_lifecycle Implementation Guide

**For:** Next implementation team  
**Date:** Oct 30, 2025  
**Status:** Ready for implementation  
**Estimated Time:** 2-3 days

---

## 📋 Overview

You need to implement and refactor the `96_lifecycle` crate bundle, which manages daemon lifecycle operations across the rbee system.

**What's Done:**
- ✅ Crate structure created
- ✅ Cargo.toml files configured
- ✅ Root workspace updated
- ✅ `health-poll` crate implemented
- ✅ Stub files created for lifecycle-local, lifecycle-ssh, lifecycle-monitored
- ✅ Documentation written

**What You Need to Do:**
- 🔧 Refactor lifecycle-local (remove SSH code)
- 🔧 Refactor lifecycle-ssh (remove poll.rs duplication)
- 🔧 Implement lifecycle-monitored (start, stop, status)
- 🔧 Implement rbee-hive-monitor (process monitoring backend)

---

## 🎯 Priority Order

### **Phase 1: Clean Up Duplication (CRITICAL - 2 hours)**
Remove RULE ZERO violations before implementing new features.

### **Phase 2: Refactor lifecycle-local (4 hours)**
Remove SSH code, integrate health-poll crate.

### **Phase 3: Refactor lifecycle-ssh (2 hours)**
Remove poll.rs duplication, integrate health-poll crate.

### **Phase 4: Implement lifecycle-monitored (8 hours)**
Implement start/stop functions (status already done).

### **Phase 5: Implement rbee-hive-monitor (16+ hours)**
Cross-platform process monitoring (future work).

---

## 📚 Required Reading

**Before starting, read these documents in order:**

1. **`RULE_ZERO_FIX.md`** - Understanding what we fixed (5 min)
2. **`UTILS_ANALYSIS.md`** - Understanding utils duplication (10 min)
3. **`lifecycle-local/REFACTOR_CHECKLIST.md`** - Detailed refactor steps (15 min)
4. **`health-poll/CHECKLIST.md`** - Health-poll requirements (10 min)
5. **`MIGRATION_COMPLETE.md`** - Understanding the migration (10 min)

**Total Reading Time:** ~50 minutes

---

## 🚀 Phase 1: Clean Up Duplication (CRITICAL)

**Time:** 2 hours  
**Priority:** 🔥 CRITICAL - Do this FIRST!

### **Why This Matters:**
We have RULE ZERO violations (duplicated code). Fix these before adding new features.

### **Tasks:**

#### **1.1 Delete poll.rs from BOTH crates (30 min)**

```bash
# Delete duplicated polling logic
rm bin/96_lifecycle/lifecycle-local/src/utils/poll.rs
rm bin/96_lifecycle/lifecycle-ssh/src/utils/poll.rs
```

**Why:** We already have `health-poll` crate. Don't duplicate!

**Files to update:**
- `lifecycle-local/src/utils/mod.rs` - Remove poll module export
- `lifecycle-ssh/src/utils/mod.rs` - Remove poll module export
- `lifecycle-local/src/start.rs` - Replace poll usage (see Phase 2)
- `lifecycle-ssh/src/start.rs` - Replace poll usage (see Phase 3)

#### **1.2 Delete ssh.rs from lifecycle-local (15 min)**

```bash
# lifecycle-local never does SSH!
rm bin/96_lifecycle/lifecycle-local/src/utils/ssh.rs
```

**Why:** lifecycle-local manages LOCAL daemons only. No SSH needed!

**Files to update:**
- `lifecycle-local/src/utils/mod.rs` - Remove ssh module export

#### **1.3 Update mod.rs files (15 min)**

**lifecycle-local/src/utils/mod.rs:**
```rust
pub mod binary;
pub mod local;
pub mod serde;

// Re-export main functions
pub use binary::check_binary_installed;
pub use local::{local_copy, local_exec};
```

**lifecycle-ssh/src/utils/mod.rs:**
```rust
pub mod binary;
pub mod local;
pub mod serde;
pub mod ssh;

// Re-export main functions
pub use binary::check_binary_installed;
pub use local::{local_copy, local_exec};
pub use ssh::{scp_upload, ssh_exec};
```

#### **1.4 Verify (30 min)**

```bash
# Should fail (good - we need to fix imports)
cargo check --package lifecycle-local
cargo check --package lifecycle-ssh

# Fix all import errors (next phases)
```

**Checkpoint:** Commit with message: "RULE ZERO: Remove duplicated poll.rs and ssh.rs from lifecycle-local"

---

## 🔧 Phase 2: Refactor lifecycle-local (4 hours)

**Time:** 4 hours  
**Priority:** 🔥 HIGH  
**Reference:** `lifecycle-local/REFACTOR_CHECKLIST.md`

### **2.1 Add health-poll dependency (5 min)**

**File:** `lifecycle-local/Cargo.toml`

```toml
[dependencies]
# ... existing dependencies ...
health-poll = { path = "../health-poll" }
```

### **2.2 Update start.rs (45 min)**

**Find and replace polling logic:**

```rust
// OLD (delete this):
use crate::utils::poll::{poll_daemon_health, HealthPollConfig};

let poll_config = HealthPollConfig {
    base_url: format!("http://localhost:{}", port),
    health_endpoint: Some("/health".to_string()),
    max_attempts: 30,
    initial_delay_ms: 200,
    backoff_multiplier: 1.5,
    job_id: job_id.clone(),
    daemon_name: Some(daemon_name.clone()),
    daemon_binary_name: daemon_name.clone(),
    ssh_config: ssh_config.clone(),
};
poll_daemon_health(poll_config).await?;

// NEW (replace with this):
let health_url = format!("http://localhost:{}/health", port);
health_poll::poll_health(
    &health_url,
    30,    // max_attempts
    200,   // initial_delay_ms
    1.5,   // backoff_multiplier
).await
.context("Daemon failed to become healthy")?;
```

**Remove SSH logic:**
- Delete any `if ssh_config.is_localhost()` checks
- Delete any SSH-specific error messages
- Use only `local_exec()` for process spawning

### **2.3 Update stop.rs (30 min)**

**Remove SSH logic:**
- Delete SSH-specific stop mechanisms
- Keep only: HTTP shutdown → SIGTERM → SIGKILL
- Use only local process termination

### **2.4 Update install.rs (30 min)**

**Remove SSH/SCP logic:**
- Delete `scp_upload()` calls
- Use only `local_copy()` for binary installation
- Simplify to direct file copy

### **2.5 Update uninstall.rs (20 min)**

**Remove SSH logic:**
- Delete SSH file deletion
- Use only local file removal

### **2.6 Update rebuild.rs (30 min)**

**Remove SSH logic:**
- Use only local operations
- Flow: build → stop → install → start (all local)

### **2.7 Update status.rs (20 min)**

**Remove SshConfig parameter:**
```rust
// OLD:
pub async fn check_daemon_health(
    health_url: &str,
    daemon_name: &str,
    ssh_config: &SshConfig,  // ← Remove this
) -> DaemonStatus

// NEW:
pub async fn check_daemon_health(
    health_url: &str,
    daemon_name: &str,
) -> DaemonStatus
```

### **2.8 Update shutdown.rs (15 min)**

**Remove SSH logic:**
- Use only local HTTP shutdown

### **2.9 Remove SshConfig from types (30 min)**

**Check all config structs:**
- `StartConfig` - Remove ssh_config field
- `StopConfig` - Remove ssh_config field
- `InstallConfig` - Remove ssh_config field
- `UninstallConfig` - Remove ssh_config field
- `RebuildConfig` - Remove ssh_config field

**Update lib.rs:**
- Remove `pub use SshConfig;` (if present)

### **2.10 Verify (30 min)**

```bash
# Should compile cleanly
cargo check --package lifecycle-local

# Run tests
cargo test --package lifecycle-local

# Check for remaining SSH references
grep -r "ssh_config" bin/96_lifecycle/lifecycle-local/src/
grep -r "SshConfig" bin/96_lifecycle/lifecycle-local/src/
grep -r "ssh_exec" bin/96_lifecycle/lifecycle-local/src/
grep -r "scp_upload" bin/96_lifecycle/lifecycle-local/src/

# Should return nothing!
```

**Checkpoint:** Commit with message: "Refactor lifecycle-local: Remove SSH code, integrate health-poll"

---

## 🔧 Phase 3: Refactor lifecycle-ssh (2 hours)

**Time:** 2 hours  
**Priority:** 🔥 HIGH

### **3.1 Add health-poll dependency (5 min)**

**File:** `lifecycle-ssh/Cargo.toml`

```toml
[dependencies]
# ... existing dependencies ...
health-poll = { path = "../health-poll" }
```

### **3.2 Update start.rs (45 min)**

**Replace polling logic (same as Phase 2.2):**

```rust
// OLD: poll_daemon_health(poll_config).await?;

// NEW:
let health_url = format!("http://{}:{}/health", hostname, port);
health_poll::poll_health(
    &health_url,
    30,    // max_attempts
    200,   // initial_delay_ms
    1.5,   // backoff_multiplier
).await
.context("Remote daemon failed to become healthy")?;
```

**Keep SSH logic:**
- Keep `ssh_exec()` for remote process spawning
- Keep `scp_upload()` for binary upload
- Keep SSH error handling

### **3.3 Update other files (30 min)**

**Files to check:**
- `stop.rs` - May use polling, replace if needed
- `status.rs` - May use polling, replace if needed
- `rebuild.rs` - May use polling, replace if needed

### **3.4 Verify (30 min)**

```bash
# Should compile cleanly
cargo check --package lifecycle-ssh

# Run tests
cargo test --package lifecycle-ssh

# Check for poll.rs references
grep -r "poll_daemon_health" bin/96_lifecycle/lifecycle-ssh/src/
grep -r "HealthPollConfig" bin/96_lifecycle/lifecycle-ssh/src/

# Should return nothing!
```

**Checkpoint:** Commit with message: "Refactor lifecycle-ssh: Integrate health-poll crate"

---

## 🚀 Phase 4: Implement lifecycle-monitored (8 hours)

**Time:** 8 hours  
**Priority:** ⚠️ MEDIUM (blocked by rbee-hive-monitor)

### **4.1 Understand Current Status (30 min)**

**What's done:**
- ✅ `status.rs` - Fully implemented (HTTP health check)
- ✅ `lib.rs` - Module structure defined
- ✅ `MonitoredConfig` type defined

**What's stubbed:**
- 📋 `start.rs` - Returns error "not yet implemented"
- 📋 `stop.rs` - Returns error "not yet implemented"

**Blocker:**
- ⚠️ Depends on `rbee-hive-monitor` crate (not implemented yet)

### **4.2 Implement start.rs (3 hours)**

**File:** `lifecycle-monitored/src/start.rs`

**Requirements:**
1. Find binary in common locations
2. Start process with `rbee-hive-monitor`
3. Apply resource limits (CPU, memory)
4. Poll health endpoint
5. Return PID

**Implementation outline:**
```rust
pub async fn start_daemon(config: MonitoredConfig) -> Result<u32> {
    // 1. Find binary
    let binary_path = find_binary(&config.daemon_name)?;
    
    // 2. Start with monitoring
    let monitor = rbee_hive_monitor::ProcessMonitor::new()?;
    let pid = monitor.start_process(
        &binary_path,
        &config.args,
        &config.monitor_config,
    ).await?;
    
    // 3. Poll health
    health_poll::poll_health(
        &config.health_url,
        30,
        200,
        1.5,
    ).await?;
    
    // 4. Return PID
    Ok(pid)
}

fn find_binary(name: &str) -> Result<PathBuf> {
    // Check: ~/.local/bin/, /usr/local/bin/, /usr/bin/
    // Return first found
}
```

### **4.3 Implement stop.rs (2 hours)**

**File:** `lifecycle-monitored/src/stop.rs`

**Requirements:**
1. Try HTTP shutdown first
2. If HTTP fails, use process-monitor to stop
3. SIGTERM → wait → SIGKILL
4. Verify process stopped

**Implementation outline:**
```rust
pub async fn stop_daemon(config: MonitoredConfig) -> Result<()> {
    // 1. Try HTTP shutdown
    let shutdown_url = format!("{}/v1/shutdown", config.health_url);
    if let Ok(_) = reqwest::Client::new()
        .post(&shutdown_url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
    {
        // Wait for graceful shutdown
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    
    // 2. Use process-monitor to force stop
    let monitor = rbee_hive_monitor::ProcessMonitor::new()?;
    monitor.stop_process(&config.daemon_name).await?;
    
    Ok(())
}
```

### **4.4 Add tests (2 hours)**

**File:** `lifecycle-monitored/tests/integration_tests.rs`

**Test cases:**
1. Start daemon successfully
2. Stop daemon gracefully (HTTP)
3. Stop daemon forcefully (SIGTERM)
4. Check status (running vs stopped)
5. Handle missing binary
6. Handle health check timeout

### **4.5 Verify (30 min)**

```bash
# Should compile
cargo check --package lifecycle-monitored

# Run tests (will fail until rbee-hive-monitor is implemented)
cargo test --package lifecycle-monitored

# Check API usage
cargo doc --package lifecycle-monitored --open
```

**Checkpoint:** Commit with message: "Implement lifecycle-monitored: start, stop functions"

---

## 🔨 Phase 5: Implement rbee-hive-monitor (16+ hours)

**Time:** 16+ hours (2+ days)  
**Priority:** ⚠️ MEDIUM (future work)  
**Complexity:** HIGH (cross-platform, cgroups, etc.)

### **5.1 Understand Requirements (1 hour)**

**Read these specs:**
- `/bin/.specs/CGROUP_INTEGRATION_PLAN.md`
- `/bin/.specs/DAEMON_LIFECYCLE_ARCHITECTURE.md`
- `/bin/99_shared_crates/process-monitor/README.md`

**Key requirements:**
- Cross-platform (Linux, macOS, Windows)
- Resource limits (CPU, memory)
- Process monitoring (CPU%, memory, I/O)
- Grouping (by service type)
- Backend abstraction (cgroup, launchd, Job Objects, sysinfo)

### **5.2 Design API (2 hours)**

**File:** `rbee-hive-monitor/src/lib.rs`

**Core types:**
```rust
pub struct ProcessMonitor {
    backend: Box<dyn MonitorBackend>,
}

pub trait MonitorBackend {
    async fn start_process(&self, binary: &Path, args: &[String], config: &MonitorConfig) -> Result<u32>;
    async fn stop_process(&self, group: &str, instance: &str) -> Result<()>;
    async fn get_stats(&self, group: &str, instance: &str) -> Result<ProcessStats>;
}

pub struct MonitorConfig {
    pub group: String,
    pub instance: String,
    pub cpu_limit: Option<String>,
    pub memory_limit: Option<String>,
}

pub struct ProcessStats {
    pub pid: u32,
    pub cpu_pct: f64,
    pub rss_mb: u64,
    pub vram_mb: Option<u64>,
    pub io_r_mb_s: f64,
    pub io_w_mb_s: f64,
    pub uptime_s: u64,
}
```

### **5.3 Implement Linux Backend (6 hours)**

**File:** `rbee-hive-monitor/src/backends/linux.rs`

**Implementation:**
1. Detect if systemd available
2. If systemd: Use `systemd-run` with cgroup v2
3. If no systemd: Use manual cgroup v2 setup
4. Read stats from `/sys/fs/cgroup/`

**Reference:** CGROUP_INTEGRATION_PLAN.md

### **5.4 Implement macOS Backend (4 hours)**

**File:** `rbee-hive-monitor/src/backends/macos.rs`

**Implementation:**
1. Try launchd for resource limits (if available)
2. Fallback to sysinfo crate for monitoring
3. No hard limits (best-effort)

### **5.5 Implement Windows Backend (4 hours)**

**File:** `rbee-hive-monitor/src/backends/windows.rs`

**Implementation:**
1. Try Job Objects for resource limits
2. Fallback to sysinfo crate for monitoring
3. Limited functionality

### **5.6 Add tests (3 hours)**

**Test each backend:**
- Start process with limits
- Monitor process stats
- Stop process gracefully
- Stop process forcefully
- Handle errors

### **5.7 Verify (1 hour)**

```bash
# Compile
cargo check --package rbee-hive-monitor

# Test on current platform
cargo test --package rbee-hive-monitor

# Test lifecycle-monitored integration
cargo test --package lifecycle-monitored
```

**Checkpoint:** Commit with message: "Implement rbee-hive-monitor: Cross-platform process monitoring"

---

## ✅ Final Verification

### **Compile All Crates**

```bash
cargo check --package health-poll
cargo check --package lifecycle-local
cargo check --package lifecycle-ssh
cargo check --package lifecycle-monitored
cargo check --package rbee-hive-monitor
```

### **Run All Tests**

```bash
cargo test --package health-poll
cargo test --package lifecycle-local
cargo test --package lifecycle-ssh
cargo test --package lifecycle-monitored
cargo test --package rbee-hive-monitor
```

### **Check Integration**

```bash
# Test rbee-keeper (uses lifecycle-local + lifecycle-ssh)
cargo check --package rbee-keeper

# Test rbee-hive (uses lifecycle-local + lifecycle-monitored)
cargo check --package rbee-hive
```

### **Verify No Duplication**

```bash
# Should return nothing:
find bin/96_lifecycle -name "poll.rs" -path "*/utils/*"

# Should only be in lifecycle-ssh:
find bin/96_lifecycle -name "ssh.rs" -path "*/utils/*"
```

---

## 📊 Success Metrics

### **Code Quality:**
- [ ] All crates compile without warnings
- [ ] All tests pass
- [ ] No duplicated code (poll.rs, ssh.rs)
- [ ] No RULE ZERO violations

### **Functionality:**
- [ ] health-poll works for all lifecycle crates
- [ ] lifecycle-local manages local daemons (no SSH code)
- [ ] lifecycle-ssh manages remote daemons (with SSH)
- [ ] lifecycle-monitored manages monitored processes
- [ ] rbee-hive-monitor provides cross-platform monitoring

### **Integration:**
- [ ] rbee-keeper uses lifecycle-local + lifecycle-ssh
- [ ] rbee-hive uses lifecycle-local + lifecycle-monitored
- [ ] All health checks use health-poll crate

---

## 🚨 Common Pitfalls

### **1. Don't Create Duplicates**
❌ **WRONG:** Create `start_local()` alongside `start()`  
✅ **RIGHT:** Update `start()` directly

### **2. Don't Keep "For Compatibility"**
❌ **WRONG:** Keep poll.rs "just in case"  
✅ **RIGHT:** Delete immediately, use health-poll

### **3. Don't Skip Reading**
❌ **WRONG:** Jump straight to coding  
✅ **RIGHT:** Read all docs first (50 min investment saves hours)

### **4. Don't Implement Out of Order**
❌ **WRONG:** Start with Phase 5 (rbee-hive-monitor)  
✅ **RIGHT:** Follow phases 1→2→3→4→5

### **5. Don't Ignore RULE ZERO**
❌ **WRONG:** "Let's keep both APIs for now"  
✅ **RIGHT:** Delete old code, update existing functions

---

## 📞 Getting Help

### **If You're Stuck:**

1. **Re-read the relevant checklist:**
   - Phase 1-3: `UTILS_ANALYSIS.md`
   - Phase 2: `lifecycle-local/REFACTOR_CHECKLIST.md`
   - Phase 4: `lifecycle-monitored/src/lib.rs` (comments)
   - Phase 5: `CGROUP_INTEGRATION_PLAN.md`

2. **Check existing code:**
   - Look at lifecycle-ssh for SSH patterns
   - Look at health-poll for polling patterns
   - Look at existing monitor crate for types

3. **Ask specific questions:**
   - "How do I replace poll_daemon_health in start.rs?"
   - "What's the correct health-poll API?"
   - "How do I remove SshConfig from StartConfig?"

---

## 🎯 Timeline Summary

| Phase | Time | Priority | Blocker |
|-------|------|----------|---------|
| **Phase 1: Clean Up** | 2 hours | 🔥 CRITICAL | None |
| **Phase 2: lifecycle-local** | 4 hours | 🔥 HIGH | Phase 1 |
| **Phase 3: lifecycle-ssh** | 2 hours | 🔥 HIGH | Phase 1 |
| **Phase 4: lifecycle-monitored** | 8 hours | ⚠️ MEDIUM | Phase 5 |
| **Phase 5: rbee-hive-monitor** | 16+ hours | ⚠️ MEDIUM | None |
| **Total** | **32+ hours** | | |

**Realistic Schedule:** 4-5 days (accounting for testing, debugging, breaks)

---

## ✅ Handoff Checklist

When you're done, verify:

- [ ] All phases completed
- [ ] All tests passing
- [ ] No compilation warnings
- [ ] No duplicated code
- [ ] Documentation updated
- [ ] Integration tests pass
- [ ] Committed with clear messages
- [ ] Created handoff document (like this one!)

**Final commit message:**
```
Complete 96_lifecycle implementation

- Removed poll.rs duplication (RULE ZERO)
- Removed ssh.rs from lifecycle-local
- Refactored lifecycle-local (no SSH code)
- Refactored lifecycle-ssh (uses health-poll)
- Implemented lifecycle-monitored (start, stop, status)
- Implemented rbee-hive-monitor (cross-platform)

All crates compile, all tests pass.
```

---

## 🎉 You're Ready!

Read the docs, follow the phases, and you'll be done in 4-5 days.

**Good luck! 🚀**
