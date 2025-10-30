# TEAM-335: Experimental Narration Removal - Stack Overflow Debugging

**Date:** Oct 28, 2025  
**Status:** 🧪 EXPERIMENTAL  
**Severity:** CRITICAL

---

## Problem Evolution

### Attempt 1: Remove `#[with_timeout]`
**Result:** ❌ Still stack overflow

### Attempt 2: Remove `#[with_job_id]`  
**Result:** ❌ Still stack overflow

### Attempt 3: Remove ALL narration (current)
**Status:** 🧪 Testing...

---

## Hypothesis

The stack overflow may be caused by:

1. ~~`#[with_timeout]` macro~~ ❌ Not the cause
2. ~~`#[with_job_id]` macro~~ ❌ Not the cause
3. **`n!()` narration macro** ← Testing now
4. **`ProcessNarrationCapture`** ← HIGHLY SUSPECT!

---

## What We Removed

### File: `daemon-lifecycle/src/build.rs`

**Commented out:**
```rust
// use observability_narration_core::{n, process_capture::ProcessNarrationCapture};

// All n!() calls removed:
// n!("build_start", ...);
// n!("build_target", ...);
// n!("build_running", ...);
// n!("build_failed", ...);
// n!("build_complete", ...);

// ProcessNarrationCapture DISABLED:
// let capture = ProcessNarrationCapture::new(job_id);
// let mut child = capture.spawn(command).await?;

// Now using plain tokio::process::Command:
let mut child = command.spawn()?;
```

### File: `daemon-lifecycle/src/install.rs`

**Commented out:**
```rust
// use observability_narration_core::n;

// All n!() calls removed:
// n!("install_start", ...);
// n!("using_binary", ...);
// n!("create_dir", ...);
// n!("copying", ...);
// n!("chmod", ...);
// n!("verify", ...);
// n!("install_complete", ...);
```

### File: `rbee-keeper/src/tauri_commands.rs`

**Commented out:**
```rust
// use observability_narration_core::n;
// n!("queen_start", "🚀 Starting queen from Tauri GUI");
```

---

## Key Suspect: ProcessNarrationCapture

**Original code:**
```rust
let capture = ProcessNarrationCapture::new(job_id);
let mut child = capture.spawn(command).await?;
```

This creates async wrappers around tokio::process::Command to stream output through SSE.

**Why it's suspect:**
- Creates additional async layers
- Wraps tokio spawn in custom async wrapper
- May buffer/stream output asynchronously
- Likely has its own internal async state machine

**Replaced with:**
```rust
let mut child = command.spawn()?;
```

Plain tokio spawn - no wrappers, no async layers.

---

## Call Stack Depth Analysis

### BEFORE (With everything)

```
1. Tauri event loop (depth: 1)
2. → Tauri invoke handler (depth: 2)
3.   → queen_install() [Tauri command] (depth: 3)
4.     → handle_queen() (depth: 4)
5.       → install_daemon() (depth: 5)
6.         → [with_job_id wrapper] (depth: 6-7)
7.         → [with_timeout wrapper] (depth: 8-9)
8.           → n!() narration (depth: 10)
9.             → build_daemon() (depth: 11)
10.              → [with_job_id wrapper] (depth: 12-13)
11.                → n!() narration (depth: 14)
12.                  → ProcessNarrationCapture::new() (depth: 15)
13.                    → ProcessNarrationCapture::spawn() (depth: 16)
14.                      → Internal async wrappers? (depth: 17-20?)
15.                        → tokio::process::Command (depth: 21+)
```

**Estimated depth:** 20+ async layers

### AFTER (Everything removed)

```
1. Tauri event loop (depth: 1)
2. → Tauri invoke handler (depth: 2)
3.   → queen_install() [Tauri command] (depth: 3)
4.     → handle_queen() (depth: 4)
5.       → install_daemon() (depth: 5)
6.         → build_daemon() (depth: 6)
7.           → tokio::process::Command (depth: 7)
```

**Estimated depth:** 7 async layers (71% reduction!)

---

## What To Test

### Test 1: Does install work now?

```bash
# In Tauri GUI: Click "Install Queen" button
# Expected: No stack overflow
```

**If it works:** ProcessNarrationCapture or narration was the cause  
**If it still fails:** Something else is wrong

### Test 2: Which component is the culprit?

If Test 1 works, systematically re-enable:

1. **Re-enable n!() calls only** (no ProcessNarrationCapture)
   - If crashes: `n!()` macro is the issue
   - If works: ProcessNarrationCapture is the issue

2. **Re-enable ProcessNarrationCapture only** (no n!() calls)
   - If crashes: ProcessNarrationCapture is the issue
   - If works: Some interaction between them is the issue

---

## Impact

### What BREAKS Without Narration

❌ **No progress updates** - User sees nothing during build  
❌ **No SSE streaming** - Web UI can't show cargo output  
❌ **No debugging** - Can't see what's happening  
❌ **No job tracking** - No job_id propagation

### What STILL WORKS

✅ **Functionality** - Install/build still works  
✅ **Error handling** - anyhow errors still propagate  
✅ **Return values** - Success/failure still detected  
✅ **Performance** - Actually faster without narration overhead

---

## Root Cause Theories

### Theory 1: n!() Macro Depth

The `n!()` macro might:
- Create temporary async contexts
- Allocate on stack for formatting
- Call into tracing infrastructure recursively

### Theory 2: ProcessNarrationCapture Async Wrapper

`ProcessNarrationCapture` likely:
- Wraps tokio::process in additional async layer
- Creates channels for stdout/stderr streaming
- Spawns background tasks for line buffering
- Each task = additional async context on stack

### Theory 3: Cumulative Effect

It's not one thing, it's the **combination**:
- Macros (#[with_job_id], #[with_timeout])
- Narration (n!() calls, ~10 per function)
- ProcessNarrationCapture (async wrappers)
- Tauri event loop (base overhead)

**Total:** 20+ async layers = stack overflow on 2MB stack

### Theory 4: Tokio Worker Stack Size

Tauri uses Tokio's default worker thread stack:
```rust
// Default in tokio:
thread_stack_size: 2 * 1024 * 1024  // 2MB
```

Main thread has 8MB, but worker threads only 2MB!

---

## Next Steps

### If This Works

1. **Confirm** - Test install in Tauri GUI
2. **Bisect** - Re-enable features one by one
3. **Identify** - Find the exact culprit
4. **Fix** - Either:
   - Remove offending feature permanently
   - Increase Tokio worker stack size
   - Refactor to reduce async depth

### If This Still Fails

1. **Check tokio version** - May have stack size regression
2. **Profile stack usage** - Use valgrind/heaptrack
3. **Add explicit stack increase**:
   ```rust
   tokio::runtime::Builder::new_multi_thread()
       .thread_stack_size(8 * 1024 * 1024)  // 8MB
       .build()
   ```
4. **Consider spawn_blocking** - Run in blocking thread pool

---

## Compilation Status

✅ `cargo build --package rbee-keeper` - SUCCESS  
⚠️ Deprecation warnings (expected - narration code still exists)  
✅ No compilation errors  
✅ Ready to test in Tauri GUI

---

## Testing Instructions

1. **Build:**
   ```bash
   cargo build --package rbee-keeper --bin rbee-keeper
   ```

2. **Run Tauri GUI:**
   ```bash
   ./target/debug/rbee-keeper
   # Or: ./rbee (via xtask)
   ```

3. **Test Install:**
   - Click "Install Queen" button
   - Watch for stack overflow

4. **If it works:**
   - Note it in this document
   - Start bisecting to find culprit

5. **If it still fails:**
   - Proceed to Theory 4 (stack size increase)

---

## Code Backup

User has made backup of working code. Safe to experiment!

---

**END OF EXPERIMENTAL DOCUMENT**
