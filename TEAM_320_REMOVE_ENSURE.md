# TEAM-320: Remove Ensure Pattern

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025  
**Impact:** Removed 163 LOC of ensure pattern code

---

## What Was Removed

**Two ensure.rs files deleted:**

1. **queen-lifecycle/src/ensure.rs** (163 LOC)
   - `ensure_queen_running()` function
   - `spawn_queen_with_preflight()` helper
   - Auto-start logic

2. **daemon-lifecycle/src/ensure.rs** (200+ LOC)
   - `ensure_daemon_running()` generic function
   - `ensure_daemon_with_handle()` generic function
   - Complex handle management logic

---

## Why Remove It?

### 1. Auto-Start is Unwanted Behavior

**The ensure pattern:**
```rust
let handle = ensure_queen_running(queen_url).await?;
// ↑ Checks if running, auto-starts if not
```

**Problem:** User doesn't want auto-start. They want explicit control.

### 2. Wrong Semantics

**"ensure" means:** Check if running, start if not  
**"start" means:** Start (fail if already running)

These are different operations. We removed "ensure" because we want explicit "start".

### 3. No Longer Used

After TEAM-318 removed auto-start from `job_client.rs`, nothing uses the ensure pattern anymore.

**Before TEAM-318:**
```rust
// job_client.rs
let queen_handle = ensure_queen_running(queen_url).await?;
```

**After TEAM-318:**
```rust
// job_client.rs
// No auto-start - just submit job
```

**Result:** The ensure pattern has zero callers.

---

## Files Changed

1. **Deleted:** `queen-lifecycle/src/ensure.rs` (163 LOC)
2. **Deleted:** `daemon-lifecycle/src/ensure.rs` (200+ LOC)

3. **Updated:** `queen-lifecycle/src/lib.rs`
   - Removed `pub mod ensure;`
   - Removed `pub use ensure::ensure_queen_running;`
   - Removed ensure documentation

4. **Updated:** `daemon-lifecycle/src/lib.rs`
   - Removed `pub mod ensure;`
   - Removed `pub use ensure::{ensure_daemon_running, ensure_daemon_with_handle};`
   - Removed ensure documentation

---

## What Users Should Do Instead

### Before (ensure pattern)

```rust
use queen_lifecycle::ensure_queen_running;

// Auto-starts if not running
let handle = ensure_queen_running("http://localhost:7833").await?;
// ... use queen ...
handle.shutdown().await?;
```

### After (explicit start)

```rust
use queen_lifecycle::{start_queen, stop_queen};

// Explicit start (fails if already running)
start_queen("http://localhost:7833").await?;
// ... use queen ...
stop_queen("http://localhost:7833").await?;
```

**Benefits:**
- ✅ Explicit control
- ✅ Clear error if queen already running
- ✅ User knows when queen starts
- ✅ Easier to debug

---

## Why This is Better

### 1. Explicit > Implicit

**Ensure (implicit):**
- Hides auto-start behavior
- User doesn't know if daemon was started
- Confusing when debugging

**Start (explicit):**
- Clear what happens
- User controls lifecycle
- Easy to debug

### 2. Simpler API

**Before:** 3 ways to start
- `start_queen()` - explicit start
- `ensure_queen_running()` - auto-start
- `spawn_queen_with_preflight()` - internal

**After:** 1 way to start
- `start_queen()` - explicit start

### 3. Less Code to Maintain

**Removed:**
- 163 LOC from queen-lifecycle
- 200+ LOC from daemon-lifecycle
- Complex handle management
- Auto-start logic

**Kept:**
- Simple start/stop functions
- Clear semantics
- Easy to understand

---

## Migration Guide

If you have code using `ensure_queen_running()`:

### Old Code
```rust
let handle = ensure_queen_running(queen_url).await?;
// ... do work ...
std::mem::forget(handle); // Keep queen alive
```

### New Code
```rust
// Check if queen is running first (optional)
if !is_queen_healthy(queen_url).await {
    start_queen(queen_url).await?;
}
// ... do work ...
// Queen stays running (no handle needed)
```

**Or just:**
```rust
// Start queen explicitly before running commands
start_queen(queen_url).await?;
// ... do work ...
```

---

## Verification

```bash
# Compilation
cargo check -p queen-lifecycle -p daemon-lifecycle
# ✅ PASS (0.47s)

# Test explicit start
./rbee queen start
# ✅ Starts queen

# Test error if already running
./rbee queen start
# ✅ Error: "Failed to start" (expected)

# Test stop
./rbee queen stop
# ✅ Stops queen
```

---

## Combined Impact (TEAM-317 through TEAM-320)

| Team | Description | LOC Impact |
|------|-------------|------------|
| TEAM-317 | Lifecycle parity | -245 |
| TEAM-318 | Remove auto-start | -27 |
| TEAM-319 | SSH duplication + mem::forget | -73 |
| TEAM-320 | Binary resolution | -10 |
| TEAM-320 | Remove ensure | -363 |
| **TOTAL** | | **-718** |

---

**Key Insight:** When a pattern has zero callers, delete it. Don't keep it "just in case."
