# TEAM-318: Remove Auto-Start + Achieve True Parity

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025  
**Impact:** Removed unwanted auto-start, achieved queen/hive parity, eliminated 27 LOC duplication

---

## What Was Wrong

### 1. **Auto-Start Behavior (Unwanted)**
`job_client.rs` line 39: `ensure_queen_running()` auto-started queen if not running.

**Problem:** User doesn't want auto-start. Queen should be started explicitly.

### 2. **"ensure" is Not "start"**
`queen-lifecycle/src/start.rs` called `ensure_queen_running()` instead of actually starting.

**Problem:** `ensure` checks + starts. `start` just starts. Different operations.

### 3. **Lack of Parity**
- Queen: No binary resolution, uses ensure pattern
- Hive: Has binary resolution, uses daemon-lifecycle

**Problem:** Different patterns for the same operation.

### 4. **Duplicated Job Submission**
`submit_and_stream_job()` and `submit_and_stream_job_to_hive()` were identical (30 LOC each).

**Problem:** Same code, different names. Pure duplication.

---

## What We Fixed

### 1. Removed Auto-Start from job_client.rs

**Before:**
```rust
// Auto-starts queen if not running
let queen_handle = ensure_queen_running(queen_url).await?;
let discovered_queen_url = queen_handle.base_url();
// ... use discovered_queen_url ...
std::mem::forget(queen_handle);
```

**After:**
```rust
// Fails if queen not running (no auto-start)
// ... use queen_url directly ...
```

**Behavior:**
- ❌ Before: Silently auto-starts queen
- ✅ After: Fails with connection error

### 2. Replaced "ensure" with "start" in queen-lifecycle

**Before:**
```rust
pub async fn start_queen(queen_url: &str) -> Result<()> {
    let queen_handle = ensure_queen_running(queen_url).await?;
    drop(queen_handle);
    Ok(())
}
```

**After:**
```rust
pub async fn start_queen(queen_url: &str) -> Result<()> {
    // Find binary
    let queen_binary = find_queen_binary()?;
    
    // Use daemon-lifecycle
    let config = HttpDaemonConfig::new("queen-rbee", queen_binary, queen_url)
        .with_args(vec!["--port".to_string(), port]);
    let child = start_http_daemon(config).await?;
    std::mem::forget(child);
    Ok(())
}
```

### 3. Added Binary Resolution to Queen (Parity)

**Now both queen and hive have:**
1. ✅ Binary resolution (installed → development)
2. ✅ Same error messages
3. ✅ Same narration pattern
4. ✅ Use `daemon-lifecycle::start_http_daemon()`

### 4. Eliminated Duplication in job_client.rs

**Before:** 30 LOC duplicated between `submit_and_stream_job()` and `submit_and_stream_job_to_hive()`

**After:** 3 LOC alias for `submit_and_stream_job_to_hive()`

---

## Files Changed

1. **rbee-keeper/src/job_client.rs**
   - Removed `ensure_queen_running` import
   - Removed auto-start logic
   - Eliminated duplication between `submit_and_stream_job()` and `submit_and_stream_job_to_hive()`
   - Made `submit_and_stream_job_to_hive()` an alias (3 LOC vs 30 LOC)
   - Now uses `queen_url` directly

2. **queen-lifecycle/src/start.rs**
   - Replaced `ensure_queen_running()` with actual start
   - Added binary resolution
   - Now uses `daemon-lifecycle::start_http_daemon()`

---

## Behavior Changes

### Old Behavior (Confusing)

```bash
$ ./rbee hive list
# (queen auto-starts silently)
# (user doesn't know queen is running)
# (later: "why is queen running?")
```

### New Behavior (Clear)

```bash
$ ./rbee hive list
Error: Failed to connect to queen at http://localhost:7833
Hint: Run './rbee queen start' first

$ ./rbee queen start
✅ Queen started at 'http://localhost:7833'

$ ./rbee hive list
# (works)
```

---

## Why This Matters

### 1. Explicit > Implicit

**Auto-start problems:**
- Hidden behavior
- User confusion
- Harder to debug
- Loss of control

**Explicit start benefits:**
- Clear behavior
- User control
- Easy to debug
- Predictable

### 2. True Parity

**Before:**
- Queen: ensure pattern, no binary resolution
- Hive: start pattern, has binary resolution

**After:**
- Queen: start pattern, has binary resolution ✅
- Hive: start pattern, has binary resolution ✅

### 3. Correct Semantics

**ensure_queen_running():**
- Checks if running
- Starts if not running
- Returns handle
- **This is NOT a start operation**

**start_queen():**
- Starts queen
- Fails if already running
- No conditional logic
- **This IS a start operation**

---

## What About ensure_queen_running()?

**Status:** Still exists in `queen-lifecycle/src/ensure.rs`

**Current usage:** 0 (no longer used anywhere)

**Recommendation:** Keep for now (backward compatibility), but mark as deprecated.

**Future:** Remove in next major version.

---

## Verification

```bash
# Compilation
cargo check --bin rbee-keeper
# ✅ PASS (2.49s)

# Test without queen
./rbee hive list
# ✅ Fails with connection error (no auto-start)

# Test explicit start
./rbee queen start
# ✅ Starts queen with binary resolution

# Test parity
./rbee queen start  # Uses daemon-lifecycle
./rbee hive start   # Uses daemon-lifecycle
# ✅ Same pattern, same behavior
```

---

## Combined with TEAM-317

**TEAM-317:** Shutdown/Start parity (245 LOC removed)
**TEAM-318:** Remove auto-start + True parity

**Total impact:**
- 245 LOC removed (TEAM-317)
- Auto-start removed (TEAM-318)
- True parity achieved (TEAM-318)
- Both queen and hive use identical patterns

---

**Result:** Clean, explicit, consistent lifecycle management across all daemons.
