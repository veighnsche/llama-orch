# Investigation Report: daemon-sync SSH Installation Failure

**Team:** TEAM-260  
**Date:** October 24, 2025  
**Status:** 🔴 CRITICAL BUG IDENTIFIED - Product broken, test correctly identifies the issue  
**Test:** `xtask/tests/daemon_sync_integration.rs::test_queen_installs_hive_in_docker`

---

## Executive Summary

The integration test for SSH-based remote installation is **correctly designed** and **properly tests the product**. The test is **FAILING** because the product has a critical bug: SSH-based installation fails silently without proper error reporting.

**Key Finding:** The SSH `exec()` call fails when executing the git clone command, but the error is swallowed in async task execution and never reaches the user via SSE narration.

---

## Test Quality Assessment ✅

### What the Test Does RIGHT

1. **Real Product Testing** - No shortcuts or mocking:
   - Builds actual `queen-rbee` binary on HOST
   - Starts real `queen-rbee` HTTP server
   - Sends real HTTP POST to `/v1/jobs`
   - SSHs to real Docker container (localhost:2222)
   - Verifies binary exists in container filesystem

2. **Proper Architecture**:
   - HOST (bare metal) → queen-rbee → SSH → Container (empty target)
   - Tests the actual deployment workflow
   - No test harness SSH (which would mask product bugs)

3. **Good Verification**:
   - Checks binary exists at correct path
   - Runs `--version` to verify binary works
   - Checks daemon status

### Test Findings

The test **correctly identifies** that:
- ❌ Binary is NOT installed in container
- ❌ Installation completes in ~8 seconds (too fast)
- ❌ No error messages in SSE stream
- ❌ Job reports success even though it failed

**This is EXACTLY what integration tests should do** - find bugs that unit tests miss!

---

## Bug Analysis

### Symptom Timeline

```
[pkg-inst  ] git_clone_hive : 📥 Cloning repository for hive 'docker-test': docker-test
[pkg-inst  ] git_clone_exec : 🔧 Executing clone command for 'docker-test': docker-test
------------------------------------------------------------
✅ Installation complete        ← WRONG! Installation did NOT complete

🔍 STEP 7: Verifying binary installation...
❌ PANIC: rbee-hive binary not found in container
```

### What's Missing from Narration

Expected but NOT seen:
- `git_clone_result` - Should show after SSH exec returns
- `git_clone_complete` - Should show after successful clone  
- `build_hive` - Should show when cargo build starts
- `build_complete` - Should show after build finishes
- `hive_result_error` - Should show if there's an error
- `sync_complete` - Should show at end of sync

### The Smoking Gun

**File:** `bin/99_shared_crates/daemon-sync/src/install.rs:326`

```rust
// This line FAILS but error is swallowed
let (stdout, stderr, exit_code) = client.exec(&clone_cmd).await?;
```

The `?` operator propagates the error up through:
1. `install_hive_from_git()` → returns `Err`
2. `install_hive_binary()` → propagates with `?`
3. `install_all()` → propagates with `?`
4. `sync_single_hive()` → propagates with `?`
5. `sync_all_hives()` → **spawns task**, catches error

**File:** `bin/99_shared_crates/daemon-sync/src/sync.rs:161-193`

```rust
let task = tokio::spawn(async move { 
    sync_single_hive(hive_arc, job_id_clone).await 
});

// Later...
match result {
    Ok(Err(e)) => {
        // Error IS caught here
        // Error narration IS added (TEAM-260)
        // But narration NEVER appears in SSE stream!
        NARRATE.action("hive_install_failed")...  // ← Not reaching client
        
        hive_results.push(HiveResult {
            error: Some(e.to_string()),  // ← Error stored but not visible
            ...
        });
    }
}

// Function returns Ok even with errors!
Ok(SyncReport { ... })  // ← Job completes "successfully"
```

### Why Error Narration Doesn't Appear

**Theory:** The spawned task is being cancelled or the job completes before narration is emitted.

**Evidence:**
1. Narration shows `git_clone_exec` (before SSH call)
2. Narration does NOT show `git_clone_result` (after SSH call)
3. Job sends `[DONE]` marker immediately
4. Test sees `[DONE]` and stops reading SSE stream
5. Any narration after `[DONE]` is lost

---

## Manual Testing Results

### SSH Command Works Manually ✅

```bash
# This WORKS perfectly when run manually
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=30 \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=120 -p 2222 rbee@localhost \
    "rm -rf ~/.local/share/rbee/build && mkdir -p ~/.local/share/rbee/build && \
     git clone --depth 1 --branch main https://github.com/veighnsche/llama-orch.git \
     ~/.local/share/rbee/build"

# Output: Cloning into '/home/rbee/.local/share/rbee/build'...
# Exit code: 0
```

### Container State After Failed Test

```bash
$ docker exec -u rbee rbee-test-target ls -la ~/.local/share/rbee/build/
# Directory exists but repo is NOT cloned (or partially cloned)

$ docker exec -u rbee rbee-test-target ls -la ~/.local/bin/
# Directory exists but rbee-hive binary is NOT present
```

### Cargo Build Works Manually ✅

```bash
# When run manually, cargo build starts and compiles
$ docker exec -u rbee rbee-test-target bash -c \
    "cd ~/.local/share/rbee/build && cargo build --release --bin rbee-hive"
# Output: Compiling dependencies... (takes 5-10 minutes)
```

---

## Fixes Applied by TEAM-260

### 1. SSH Keepalive Options ✅

**Problem:** Long-running commands (cargo build) might timeout  
**Fix:** Added ServerAliveInterval and ServerAliveCountMax

**File:** `bin/99_shared_crates/ssh-client/src/lib.rs:95-102`

```rust
// TEAM-260: Keep SSH alive during long builds
.arg("-o").arg("ServerAliveInterval=60")      // Send keepalive every 60s
.arg("-o").arg("ServerAliveCountMax=120")     // Allow 120 missed (120min total)
```

### 2. Build Output Redirection ✅

**Problem:** SSH buffer might fill up with cargo build output  
**Fix:** Redirect output to log file

**File:** `bin/99_shared_crates/daemon-sync/src/install.rs:341-362`

```rust
// TEAM-260: Redirect to prevent SSH buffer issues
let build_log = format!("{}/.build.log", clone_dir);
let build_cmd = format!(
    "cd {} && cargo build --release --bin rbee-hive > {} 2>&1; echo $?",
    clone_dir, build_log
);

// Parse exit code from stdout
let build_exit_code: i32 = stdout.trim().parse().unwrap_or(255);
```

### 3. Error Narration Added ✅ (but not working)

**Problem:** Errors fail silently  
**Fix:** Added narration at multiple points

**Files:**
- `bin/99_shared_crates/daemon-sync/src/install.rs:320-329` - Git clone errors
- `bin/99_shared_crates/daemon-sync/src/sync.rs:176-184` - Task errors
- `bin/99_shared_crates/daemon-sync/src/sync.rs:195-203` - Task panics
- `bin/99_shared_crates/daemon-sync/src/sync.rs:232-249` - Hive results

**Status:** Narration added but NOT appearing in SSE stream (see "Why Error Narration Doesn't Appear" above)

### 4. Test Timeout Increased ✅

**Problem:** 3 minutes too short for cargo build  
**Fix:** Increased to 10 minutes

**File:** `xtask/tests/daemon_sync_integration.rs:318`

```rust
// TEAM-260: Increased from 90 to 300 attempts
for attempt in 1..=300 {  // 300 * 2s = 10 minutes max
```

### 5. Debug Narration Added ✅

**Purpose:** Track execution flow to find where it fails

**File:** `bin/99_shared_crates/daemon-sync/src/install.rs:318-335`

```rust
// TEAM-260: Debug narration
NARRATE.action("git_clone_exec")...      // ← Shows up
let (stdout, stderr, exit_code) = client.exec(&clone_cmd).await?;
NARRATE.action("git_clone_result")...    // ← Does NOT show up
```

---

## Root Cause Hypothesis

### Primary Theory: SSH Exec Fails Immediately

The SSH `client.exec()` call is returning an `Err` (not a successful result with non-zero exit code).

**Possible reasons:**
1. **Connection timeout** - SSH connection closes before command completes
2. **Permission issue** - Command can't execute in SSH context
3. **Shell expansion** - Tilde `~` not expanding in non-login shell
4. **Command too long** - SSH command line length limit
5. **Async cancellation** - Tokio task cancelled before completion

### Evidence Supporting This Theory

1. Manual SSH works → Product SSH fails
2. Narration shows `git_clone_exec` → Does NOT show `git_clone_result`
3. This means the line `client.exec().await?` never returns
4. The `?` operator propagates an error immediately
5. Error is caught in spawned task but narration doesn't reach client

### Why Manual Works But Product Fails

**Manual:**
- Runs in foreground shell
- Waits indefinitely for completion
- Direct stdout/stderr visibility

**Product:**
- Runs in async tokio task
- May have implicit timeouts
- Buffered stdout/stderr
- Possible task cancellation

---

## What We DON'T Know Yet

### Critical Unknowns

1. **What is the actual error message?**
   - The `Err` returned by `client.exec()` contains the error
   - But we never see it because narration doesn't reach the stream
   - Need synchronous logging (not just narration) to capture it

2. **Why doesn't error narration appear?**
   - Is the task being cancelled?
   - Is the job completing before narration is emitted?
   - Is there a race condition in SSE stream closing?

3. **Does the SSH connection even establish?**
   - We see `git_clone_exec` so the function starts
   - But does the SSH connection succeed?
   - Or does it fail during connection?

4. **Is this a tokio::spawn issue?**
   - Does spawning the task cause problems?
   - Would running synchronously work?
   - Is there a timeout on spawned tasks?

---

## Recommended Investigation Steps

### PRIORITY 1: Capture the Actual Error

**Add synchronous logging** (not narration) to see the error:

```rust
// In install.rs:326
match client.exec(&clone_cmd).await {
    Ok((stdout, stderr, exit_code)) => {
        eprintln!("SSH exec SUCCESS: exit={}", exit_code);  // ← Add this
        // Continue...
    }
    Err(e) => {
        eprintln!("SSH exec FAILED: {}", e);  // ← Add this
        eprintln!("Error details: {:?}", e);
        return Err(e);
    }
}
```

**Check queen-rbee stderr** to see the error message.

### PRIORITY 2: Test SSH Connection Separately

**Add a simple test before git clone:**

```rust
// Test if SSH works at all
let test_cmd = "echo 'SSH OK'";
let (stdout, stderr, exit_code) = client.exec(test_cmd).await?;
NARRATE.action("ssh_test_result")
    .context(&format!("exit={}, stdout={}", exit_code, stdout))
    .emit();
```

If this fails, SSH connection is broken.  
If this works, the problem is specific to the git clone command.

### PRIORITY 3: Simplify the Git Clone Command

**Test each part separately:**

```rust
// Step 1: Test rm
client.exec("rm -rf ~/.local/share/rbee/build").await?;

// Step 2: Test mkdir
client.exec("mkdir -p ~/.local/share/rbee/build").await?;

// Step 3: Test git clone
client.exec("git clone --depth 1 --branch main https://... ~/.local/share/rbee/build").await?;
```

Find which command fails.

### PRIORITY 4: Fix Job Completion Logic

**Ensure tasks complete before job finishes:**

```rust
// In sync.rs
let results = futures::future::join_all(tasks).await;

// BEFORE returning, ensure all narration is emitted
tokio::time::sleep(Duration::from_millis(100)).await;  // Give narration time to flush

// Emit final results
for result in &hive_results {
    // ... emit narration ...
}

// THEN return
Ok(SyncReport { ... })
```

### PRIORITY 5: Add Timeout Handling

**Make timeouts explicit:**

```rust
// In install.rs
let clone_future = client.exec(&clone_cmd);
let result = tokio::time::timeout(Duration::from_secs(300), clone_future).await;

match result {
    Ok(Ok((stdout, stderr, exit_code))) => { /* success */ }
    Ok(Err(e)) => { /* SSH error */ }
    Err(_) => { /* timeout after 5 minutes */ }
}
```

---

## Quick Wins for Next Team

### Option A: Pre-built Binary (Workaround)

**Fastest way to get test passing:**

1. Build `rbee-hive` on HOST
2. Use SCP to copy to container
3. Skip git clone + cargo build
4. Verify rest of flow works

**Code:**
```rust
// Instead of install_hive_from_git()
let local_binary = workspace_root.join("target/debug/rbee-hive");
client.copy_file(local_binary.to_str().unwrap(), "~/.local/bin/rbee-hive").await?;
client.exec("chmod +x ~/.local/bin/rbee-hive").await?;
```

**Pros:** Test passes, verifies deployment flow  
**Cons:** Doesn't test git clone (which is broken)

### Option B: Synchronous Execution (Debug)

**Run without tokio::spawn to see errors:**

```rust
// In sync.rs, instead of spawning
// let task = tokio::spawn(async move { ... });

// Run directly
let result = sync_single_hive(hive_arc, job_id_clone).await;
```

**Pros:** Errors propagate normally  
**Cons:** No concurrency (but only 1 hive in test anyway)

### Option C: Enhanced Logging

**Add eprintln! everywhere:**

```rust
eprintln!("TEAM-260 DEBUG: Starting git clone");
eprintln!("TEAM-260 DEBUG: Command: {}", clone_cmd);
let result = client.exec(&clone_cmd).await;
eprintln!("TEAM-260 DEBUG: Result: {:?}", result);
```

**Pros:** See exactly where it fails  
**Cons:** Clutters code with debug statements

---

## Architecture Issues Identified

### Issue 1: Silent Failures

**Problem:** Errors are caught but not visible to users

**Current:**
```rust
Ok(Err(e)) => {
    hive_results.push(HiveResult { error: Some(e.to_string()) });
}
// Returns Ok(SyncReport { ... })  ← Job succeeds even with errors!
```

**Should be:**
```rust
Ok(Err(e)) => {
    // Emit error narration (already added by TEAM-260)
    // But ALSO fail the job if ANY hive fails
    return Err(e);  // ← Propagate error to job
}
```

### Issue 2: Async Task Errors Lost

**Problem:** Spawned tasks can fail without visibility

**Solution:** Use `tokio::task::JoinSet` with better error handling:

```rust
let mut set = tokio::task::JoinSet::new();
for hive in hives {
    set.spawn(async move { sync_single_hive(hive, job_id).await });
}

while let Some(result) = set.join_next().await {
    match result {
        Ok(Ok(hive_result)) => { /* success */ }
        Ok(Err(e)) => { /* task returned error - VISIBLE */ }
        Err(e) => { /* task panicked - VISIBLE */ }
    }
}
```

### Issue 3: Narration Timing

**Problem:** Narration may be emitted after SSE stream closes

**Solution:** Ensure narration is flushed before `[DONE]`:

```rust
// Emit all narration
for result in &hive_results {
    NARRATE.action(...)...emit();
}

// Wait for narration to flush
tokio::time::sleep(Duration::from_millis(50)).await;

// THEN complete job (which sends [DONE])
```

---

## Files Modified by TEAM-260

### SSH Client
- `bin/99_shared_crates/ssh-client/src/lib.rs`
  - Added ServerAliveInterval/CountMax (lines 95-102)
  - Improved connection keepalive for long commands

### Daemon Sync
- `bin/99_shared_crates/daemon-sync/src/install.rs`
  - Added git clone debug narration (lines 318-335)
  - Added git clone error narration (lines 320-329)
  - Redirected cargo build output to log file (lines 341-362)

- `bin/99_shared_crates/daemon-sync/src/sync.rs`
  - Added task error narration (lines 176-184)
  - Added task panic narration (lines 195-203)
  - Added hive result narration (lines 232-249)

### Integration Test
- `xtask/tests/daemon_sync_integration.rs`
  - Increased timeout from 3min to 10min (line 318)
  - Updated error messages (line 353)

### Documentation
- `bin/99_shared_crates/ssh-client/TEAM_260_RUSSH_TO_COMMAND_MIGRATION.md`
- `bin/99_shared_crates/ssh-client/USAGE_ANALYSIS.md`
- `bin/99_shared_crates/ssh-client/TEAM_260_RELOCATION_SUMMARY.md`
- `TEAM_260_TEST_DEBUG_SUMMARY.md`

---

## Test Environment Details

### Container Setup

**Image:** `rbee-test-target:latest` (Arch Linux)  
**Dockerfile:** `tests/docker/Dockerfile.target`

**Installed:**
- OpenSSH server
- Rust toolchain (cargo, rustc)
- Git
- Base development tools

**User:** `rbee` (non-root)  
**SSH Port:** 2222 (mapped from container port 22)  
**SSH Key:** `tests/docker/keys/test_id_rsa`

**Directory Structure:**
```
/home/rbee/
├── .ssh/
│   └── authorized_keys (test_id_rsa.pub)
├── .local/
│   ├── bin/              ← rbee-hive should be installed here
│   └── share/rbee/
│       ├── build/        ← git clone target
│       └── hives/
└── .config/rbee/
```

### Test Configuration

**File:** `tests/docker/hives.conf`

```toml
[[hive]]
alias = "docker-test"
hostname = "localhost"
ssh_port = 2222
ssh_user = "rbee"
hive_port = 9000
auto_start = true

[hive.install_method]
git = { repo = "https://github.com/veighnsche/llama-orch.git", branch = "main" }
```

---

## Success Criteria

The test will PASS when:

1. ✅ Git clone completes successfully
2. ✅ Cargo build completes (5-10 minutes)
3. ✅ Binary installed at `/home/rbee/.local/bin/rbee-hive`
4. ✅ Binary is executable and runs `--version`
5. ✅ All narration appears in SSE stream
6. ✅ Errors (if any) are visible to user

---

## For the Next Team

### Start Here

1. **Read this document** - Understand what's been tried
2. **Run PRIORITY 1** - Add eprintln! to capture the actual error
3. **Check queen-rbee stderr** - See what the error message is
4. **Fix the root cause** - Don't work around it

### Don't Waste Time On

- ❌ Rewriting the test (it's correct)
- ❌ Mocking SSH (defeats the purpose)
- ❌ Blaming the container (manual SSH works)
- ❌ Adding more narration (we have enough, it's not reaching the stream)

### Focus On

- ✅ **Why does SSH exec fail?** (capture the error)
- ✅ **Why doesn't error narration appear?** (task/job lifecycle)
- ✅ **How to make errors visible?** (synchronous logging)

### Quick Test

```bash
# Clean slate
docker rm -f rbee-test-target
pkill -f queen-rbee

# Run test
cargo test --package xtask --test daemon_sync_integration -- --ignored --nocapture

# Check queen-rbee stderr for error messages
# (if you added eprintln! as recommended)
```

---

## Conclusion

**The test is GOOD.** It found a real bug.

**The product is BROKEN.** SSH-based installation fails silently.

**The fix is NEEDED.** Users can't deploy to remote systems.

**Next team:** Capture the actual error message, fix the root cause, ensure errors are visible.

Good luck! 🐝

---

**TEAM-260 Sign-off**  
Date: October 24, 2025  
Status: Investigation complete, bug identified, fixes attempted, root cause still unknown  
Handoff: Ready for next team to capture error and fix
