# TEAM-335: Stack Overflow Comprehensive Analysis

**Date:** Oct 28, 2025  
**Status:** ❌ UNSOLVED  
**Severity:** CRITICAL - Blocks Tauri GUI install functionality

---

## Symptom

```
thread 'tokio-runtime-worker' has overflowed its stack
fatal runtime error: stack overflow, aborting
```

**Trigger:** Click "Install Queen" button in Tauri GUI  
**Works in CLI:** `./rbee queen install` ✅ No issues

---

## What We've Tried (All Failed)

### ❌ Attempt 1: Remove `#[with_timeout]` Macro

**Theory:** Timeout macro creates nested async wrappers causing stack overflow

**Changes:**
- `daemon-lifecycle/src/install.rs:127` - Commented out `#[with_timeout(secs = 300)]`
- Removed `timeout_enforcer` import

**Result:** ❌ **STILL STACK OVERFLOW**

**Learning:** Timeout macro was not the cause

---

### ❌ Attempt 2: Remove `#[with_job_id]` Macro

**Theory:** Job ID macro creates additional async context layers

**Changes:**
- `daemon-lifecycle/src/install.rs:132` - Commented out `#[with_job_id]`
- `daemon-lifecycle/src/build.rs:98` - Commented out `#[with_job_id]`
- Removed imports

**Result:** ❌ **STILL STACK OVERFLOW**

**Learning:** Job ID macro was not the cause

---

### ❌ Attempt 3: Remove ALL Narration

**Theory:** `n!()` macro or ProcessNarrationCapture creating async depth

**Changes:**
- `daemon-lifecycle/src/build.rs` - Commented out all `n!()` calls (5 locations)
- `daemon-lifecycle/src/build.rs:111` - Disabled ProcessNarrationCapture
- `daemon-lifecycle/src/build.rs:131` - Changed to plain `command.spawn()`
- `daemon-lifecycle/src/install.rs` - Commented out all `n!()` calls (7 locations)
- `rbee-keeper/src/tauri_commands.rs:93` - Commented out narration

**Result:** ❌ **STILL STACK OVERFLOW**

**Learning:** Narration was not the cause

---

### ❌ Attempt 4: Use `spawn_blocking`

**Theory:** Tauri workers have 2MB stack, OS threads have 8MB

**Changes:**
- `rbee-keeper/src/tauri_commands.rs:133` - Wrapped in `tokio::task::spawn_blocking`
- Created fresh Tokio runtime inside blocking context

**Result:** ❌ **STILL STACK OVERFLOW**

**Learning:** Either not a stack size issue, or spawn_blocking isn't providing the expected stack

---

## What This Tells Us

### Definitely NOT the Cause

1. ✅ **NOT #[with_timeout] macro** - Removed, still crashes
2. ✅ **NOT #[with_job_id] macro** - Removed, still crashes  
3. ✅ **NOT n!() narration** - Removed all, still crashes
4. ✅ **NOT ProcessNarrationCapture** - Disabled, still crashes
5. ✅ **NOT simple stack size** - spawn_blocking didn't help

### Still Possible Causes

#### Possibility 1: Recursive Function Call

**Likelihood:** 🟡 Medium

**Theory:** There's a hidden recursive call somewhere in the call stack

**Evidence:**
- Stack overflow is consistent
- Happens at same point every time
- Not affected by removing async wrappers

**How to Test:**
```bash
# Use Rust's backtrace
RUST_BACKTRACE=full ./target/debug/rbee-keeper
# Click install, capture stack trace
```

**Next Steps:**
- Add explicit stack trace logging
- Look for repeated function names in stack
- Check ssh_exec, scp_upload for recursion

---

#### Possibility 2: Tokio Runtime Nested Creation

**Likelihood:** 🔴 High

**Theory:** Creating a Tokio runtime INSIDE a Tokio context causes issues

**Evidence:**
```rust
// In spawn_blocking, we do:
let rt = tokio::runtime::Runtime::new()?;
rt.block_on(async move { ... });
```

This might be creating nested runtimes which share stack space incorrectly.

**How to Test:**
- Don't create new runtime, just call handle_queen directly
- Use `Handle::current()` instead of new runtime

**Next Steps:**
```rust
tokio::task::spawn_blocking(move || {
    // DON'T create new runtime
    let handle = tokio::runtime::Handle::current();
    handle.block_on(async move {
        // ... 
    })
})
```

---

#### Possibility 3: SSH Operations Deep Call Stack

**Likelihood:** 🟡 Medium

**Theory:** `ssh_exec()` or `scp_upload()` have deep internal call stacks

**Evidence:**
- Uses tokio::process::Command
- Might have many internal async layers
- Install does 4 SSH calls + 1 SCP

**How to Test:**
- Skip SSH operations entirely (mock them)
- Test install with pre-built binary (skips cargo build)

**Next Steps:**
```rust
// In install_daemon, add early return for testing:
if ssh_config.hostname == "localhost" {
    // Skip all SSH, just return success
    return Ok(());
}
```

---

#### Possibility 4: Cargo Build Process

**Likelihood:** 🟢 Low

**Theory:** Spawning cargo build in Tauri context causes issues

**Evidence:**
- Build is long-running
- Uses tokio::process::Command
- Lots of output streaming

**How to Test:**
- Install with pre-built binary (set `binary: Some("/path/to/queen-rbee")`)
- This skips build entirely

**Next Steps:**
```typescript
// In UI, pass explicit binary path:
await invoke("queen_install", { 
    binary: "/path/to/pre-built/queen-rbee" 
});
```

---

#### Possibility 5: Tauri invoke_handler Limit

**Likelihood:** 🟡 Medium

**Theory:** Tauri has undocumented limit on async depth in command handlers

**Evidence:**
- Only happens in Tauri, not CLI
- Tauri v2 is relatively new
- Might be a Tauri bug

**How to Test:**
- Check Tauri GitHub issues for "stack overflow"
- Try simpler operation (just return "OK" without any logic)

**Next Steps:**
```rust
#[tauri::command]
pub async fn queen_install_test() -> Result<String, String> {
    Ok("Test passed".to_string())
}
```

If this works, incrementally add back logic to find breaking point.

---

#### Possibility 6: Actual Stack Size Configuration

**Likelihood:** 🔴 High

**Theory:** spawn_blocking might not give 8MB, or we need explicit stack size

**Evidence:**
- Tokio's spawn_blocking uses thread pool
- Thread pool threads might have custom stack size
- Tauri might configure smaller stack for all threads

**How to Test:**
```rust
// Create thread with explicit large stack
std::thread::Builder::new()
    .stack_size(16 * 1024 * 1024)  // 16MB explicit
    .spawn(move || {
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async { ... })
    })?
    .join()
```

**Next Steps:**
- Try explicit stack size in thread builder
- Check Tauri's thread pool configuration

---

#### Possibility 7: anyhow::Error Deep Stack

**Likelihood:** 🟢 Low  

**Theory:** anyhow::Error with deep context chains causes stack issues

**Evidence:**
- Every function uses `.context("...")?`
- Error chains can be deep
- Stack overflow happens on error path?

**How to Test:**
- Remove all `.context()` calls temporarily
- See if it still crashes

**Next Steps:**
- Add logging before each major operation
- See which operation causes crash

---

#### Possibility 8: Config::load() Issues

**Likelihood:** 🟢 Low

**Theory:** Config loading has deep call stack

**Evidence:**
- Called at start of queen_install
- Might do file I/O with deep async

**How to Test:**
```rust
#[tauri::command]
pub async fn queen_install(binary: Option<String>) -> Result<String, String> {
    // Skip config loading entirely
    let queen_url = "http://localhost:7833";
    // ... rest
}
```

---

## Recommended Testing Order

### Priority 1: Eliminate Build Step (Fastest Test)

```bash
# Pre-build queen
cargo build --release --bin queen-rbee

# In UI, install with explicit binary path
await invoke("queen_install", { 
    binary: "/path/to/llama-orch/target/release/queen-rbee" 
});
```

**If this works:** Problem is in `build_daemon()`  
**If this fails:** Problem is in `install_daemon()` or Tauri layer

---

### Priority 2: Explicit Stack Size Thread

```rust
std::thread::Builder::new()
    .stack_size(16 * 1024 * 1024)
    .spawn(move || { ... })
```

**If this works:** It's a stack size issue, spawn_blocking isn't enough  
**If this fails:** It's something else

---

### Priority 3: Mock All Operations

```rust
#[tauri::command]
pub async fn queen_install(binary: Option<String>) -> Result<String, String> {
    // Just return success, do nothing
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    Ok("Fake install complete".to_string())
}
```

**If this works:** Problem is in business logic  
**If this fails:** Problem is in Tauri itself (very unlikely)

---

### Priority 4: Incremental Complexity

Start with empty function, add back logic piece by piece:

1. Empty function → Works?
2. + Config::load() → Works?
3. + handle_queen call → Works?
4. + install_daemon call → Works?
5. + build_daemon call → Works?

Find the exact breaking point.

---

## Stack Trace Analysis Needed

```bash
# Get full stack trace
RUST_BACKTRACE=full RUST_LOG=trace ./target/debug/rbee-keeper

# Or use gdb
gdb ./target/debug/rbee-keeper
(gdb) run
# Click install, let it crash
(gdb) backtrace full
```

**Look for:**
- Repeated function names (indicates recursion)
- Very deep async frames (>50 depth)
- Tokio internal functions repeated
- Any function appearing >10 times

---

## Potential Root Causes Ranked

| Rank | Cause | Likelihood | Test Priority |
|------|-------|------------|---------------|
| 1 | Tokio runtime nested creation | 🔴 High | P2 |
| 2 | spawn_blocking stack not enough | 🔴 High | P2 |
| 3 | Tauri async depth limit | 🟡 Medium | P3 |
| 4 | SSH operations deep stack | 🟡 Medium | P1 |
| 5 | Cargo build process | 🟡 Medium | P1 |
| 6 | Hidden recursion | 🟡 Medium | P4 |
| 7 | anyhow::Error chains | 🟢 Low | P4 |
| 8 | Config loading | 🟢 Low | P4 |

---

## Files Changed (Current State)

### Narration Disabled
- `bin/99_shared_crates/daemon-lifecycle/src/build.rs`
- `bin/99_shared_crates/daemon-lifecycle/src/install.rs`
- `bin/00_rbee_keeper/src/tauri_commands.rs`

### Macros Commented Out
- `#[with_timeout]` in install.rs
- `#[with_job_id]` in install.rs and build.rs

### Current Attempt
- `spawn_blocking` with fresh runtime in tauri_commands.rs

**All changes documented in code comments per debugging-rules.md**

---

## Next Team TODO

1. **Read this document fully** - Don't repeat our failed attempts
2. **Try Priority 1 test first** - Skip build step
3. **Get stack trace** - Use RUST_BACKTRACE=full
4. **Try explicit stack size** - Priority 2
5. **Document your attempts** - Add to investigation comment in tauri_commands.rs

---

## Open Questions

1. Why does CLI work but Tauri doesn't?
   - Same code, same functions, different context
   - CLI uses main thread (8MB), Tauri uses worker thread (2MB?)

2. Why did spawn_blocking not help?
   - Should give OS thread with 8MB stack
   - Maybe Tauri configures different thread pool?
   - Maybe we need explicit stack_size()?

3. Is there a Tauri configuration we're missing?
   - Check tauri.conf.json for stack size settings
   - Check Tauri docs for async depth limits

4. Could this be a Tauri v2 bug?
   - Tauri v2 is relatively new
   - Check GitHub issues
   - Try downgrade to Tauri v1?

---

**STATUS: Unsolved - 4 attempts failed**

**DO NOT:**
- ❌ Try removing macros again (already tried)
- ❌ Try removing narration again (already tried)
- ❌ Assume spawn_blocking will work (already tried, didn't work)
- ❌ Be confident about any fix without testing

**DO:**
- ✅ Get stack trace first
- ✅ Test incrementally
- ✅ Document all attempts
- ✅ Consider it might be a Tauri bug

---

**END OF ANALYSIS**
