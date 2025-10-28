# TEAM-260: Integration Test Debugging Summary

**Date:** October 24, 2025  
**Test:** `daemon_sync_integration::test_queen_installs_hive_in_docker`  
**Status:** ❌ FAILING

## Problem

The integration test fails because the rbee-hive binary is not installed in the container, even though the job completes successfully.

## Root Cause Analysis

### Symptoms

1. Test completes in ~8 seconds (too fast for git clone + cargo build)
2. SSE stream shows:
   ```
   [pkg-inst  ] git_clone_hive : 📥 Cloning repository...
   [pkg-inst  ] git_clone_exec : 🔧 Executing clone command...
   [DONE]
   ```
3. Missing narration:
   - ❌ `git_clone_result` - Should show after SSH exec completes
   - ❌ `git_clone_complete` - Should show after successful clone
   - ❌ `build_hive` - Should show when build starts
   - ❌ `hive_result_error` - Should show if there's an error
   - ❌ `sync_complete` - Should show at the end

### Investigation

**File:** `bin/99_shared_crates/daemon-sync/src/install.rs:326`

The SSH exec call is failing:
```rust
let (stdout, stderr, exit_code) = client.exec(&clone_cmd).await?;  // ← Fails here
```

The `?` operator propagates the error up the call stack, but the error is being swallowed somewhere.

**Call Stack:**
1. `install_hive_from_git` - SSH exec fails, returns `Err` with `?`
2. `install_hive_binary` - Propagates error with `?`
3. `install_all` - Propagates error with `?`
4. `sync_single_hive` - Propagates error with `?`
5. `sync_all_hives` - Spawns task with `tokio::spawn`, catches error

**File:** `bin/99_shared_crates/daemon-sync/src/sync.rs:175-193`

Errors ARE being caught and converted to `HiveResult` with error field:
```rust
Ok(Err(e)) => {
    // Error narration added but NOT showing up!
    NARRATE.action("hive_install_failed")...
    
    hive_results.push(HiveResult {
        error: Some(e.to_string()),
        ...
    });
}
```

But the narration is NOT appearing in the SSE stream!

### The Mystery

**Why is the error narration not showing up?**

Possible explanations:
1. ❌ Job completes before narration is emitted
2. ❌ Narration is emitted after `[DONE]` marker
3. ❌ SSE stream closes before receiving narration
4. ✅ **Most likely:** The spawned task is being cancelled/killed before it completes

## SSH Command Analysis

**Manual Test (WORKS):**
```bash
ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=30 \
    -o ServerAliveInterval=60 -o ServerAliveCountMax=120 -p 2222 rbee@localhost \
    "rm -rf ~/.local/share/rbee/build && mkdir -p ~/.local/share/rbee/build && \
     git clone --depth 1 --branch main https://github.com/veighnsche/llama-orch.git \
     ~/.local/share/rbee/build"
# ✅ Works perfectly
```

**Product Code (FAILS):**
```rust
// Same command, same SSH options
client.exec(&clone_cmd).await?  // ❌ Fails silently
```

**Why the discrepancy?**
- Manual test runs in foreground shell
- Product code runs via tokio async
- Possible timeout/cancellation issue

## Fixes Applied

### 1. SSH Keepalive (DONE)
Added ServerAliveInterval and ServerAliveCountMax to prevent SSH timeout during long builds.

**File:** `bin/99_shared_crates/ssh-client/src/lib.rs:101-102`

### 2. Build Output Redirection (DONE)
Redirect cargo build output to log file to prevent SSH buffer issues.

**File:** `bin/99_shared_crates/daemon-sync/src/install.rs:341-345`

### 3. Error Narration (DONE but not working)
Added detailed error narration at multiple points:
- Git clone failures
- Task errors
- Hive result errors

**Files:**
- `bin/99_shared_crates/daemon-sync/src/install.rs:320-329`
- `bin/99_shared_crates/daemon-sync/src/sync.rs:176-184, 195-203, 232-249`

### 4. Test Timeout Increase (DONE)
Increased test timeout from 3 minutes to 10 minutes.

**File:** `xtask/tests/daemon_sync_integration.rs:318`

## What's Still Broken

The SSH exec call is failing, but we can't see the error because:
1. The error is being caught in the spawned task
2. The error narration is not reaching the SSE stream
3. The job completes (sends `[DONE]`) before we can see what happened

## Next Steps

### Option 1: Debug the SSH Exec Failure
1. Add logging to see the actual error message
2. Check if it's a timeout, permission, or connection issue
3. Fix the root cause

### Option 2: Simplify the Test
1. Pre-build rbee-hive on HOST
2. Use SCP to copy binary to container
3. Verify the rest of the flow works
4. Come back to fix git clone later

### Option 3: Fix the Job Execution
1. Ensure spawned tasks complete before job finishes
2. Ensure all narration is emitted before `[DONE]`
3. Add timeout handling for long-running tasks

## Recommendation

**Start with Option 1** - We need to see the actual error message. Add synchronous logging (not just narration) to capture the SSH exec error.

**Then do Option 3** - Fix the job execution to ensure tasks complete properly.

**Option 2 is a workaround** - Only use if we can't fix the root cause quickly.

## Test Quality Assessment

✅ **The test is GOOD** - It properly tests the product:
- Real queen-rbee binary
- Real SSH connection
- Real HTTP API
- Real verification

❌ **The product is BROKEN** - SSH-based installation is failing silently.

The test found a real bug! This is exactly what integration tests are for.
