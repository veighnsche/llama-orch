# TEAM-318: Remove Auto-Start Behavior

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025  
**Priority:** CRITICAL - Unwanted Auto-Start + Lack of Parity

---

## Problems Found

### 1. Auto-Start in job_client.rs
**Line 39:** `let queen_handle = ensure_queen_running(queen_url).await?;`

This auto-started queen if it wasn't running. **You don't want this.**

### 2. queen-lifecycle uses "ensure" not "start"
**queen-lifecycle/src/start.rs** called `ensure_queen_running()` which:
- Checks if queen is running
- Auto-starts if not running
- Returns handle

This is NOT a start operation - it's an ensure operation.

### 3. Lack of Parity
**queen-lifecycle/src/start.rs:**
- No binary resolution
- Uses `ensure_queen_running()` helper
- Different pattern from hive

**hive-lifecycle/src/start.rs:**
- Has binary resolution (installed vs development)
- Uses `daemon-lifecycle::start_http_daemon()`
- Proper start operation

---

## Solution

### 1. Removed Auto-Start from job_client.rs

**Before:**
```rust
pub async fn submit_and_stream_job(queen_url: &str, operation: Operation) -> Result<()> {
    // Ensure queen is running
    let queen_handle = ensure_queen_running(queen_url).await?;
    
    // Use discovered queen URL from handle
    let discovered_queen_url = queen_handle.base_url();
    
    // ... submit job ...
    
    // Cleanup queen handle
    std::mem::forget(queen_handle);
}
```

**After:**
```rust
pub async fn submit_and_stream_job(queen_url: &str, operation: Operation) -> Result<()> {
    // TEAM-318: No auto-start - queen must be running
    // Job submission fails if queen is not running
    
    // ... submit job directly to queen_url ...
}
```

**Behavior change:**
- ❌ Before: Auto-started queen if not running
- ✅ After: Fails with connection error if queen not running

### 2. Replaced "ensure" with "start" in queen-lifecycle

**Before (ensure pattern):**
```rust
pub async fn start_queen(queen_url: &str) -> Result<()> {
    let queen_handle = ensure_queen_running(queen_url).await?;
    n!("queen_start", "✅ Queen started on {}", queen_handle.base_url());
    drop(queen_handle);
    Ok(())
}
```

**After (actual start):**
```rust
pub async fn start_queen(queen_url: &str) -> Result<()> {
    n!("start_queen", "▶️  Starting queen-rbee...");
    
    // Find binary (matches hive pattern)
    let queen_binary = find_queen_binary()?;
    
    // Use daemon-lifecycle
    let config = HttpDaemonConfig::new("queen-rbee", queen_binary, queen_url)
        .with_args(vec!["--port".to_string(), port.to_string()]);
    
    let child = start_http_daemon(config).await?;
    std::mem::forget(child);
    
    n!("start_queen_complete", "✅ Queen started at '{}'", queen_url);
    Ok(())
}
```

### 3. Added Binary Resolution to Queen (Parity with Hive)

**Now both queen and hive:**
1. ✅ Resolve binary location (installed → development)
2. ✅ Use `daemon-lifecycle::start_http_daemon()`
3. ✅ Same error messages
4. ✅ Same narration pattern
5. ✅ Same startup flow

---

## Files Changed

1. **rbee-keeper/src/job_client.rs**
   - Removed `use queen_lifecycle::ensure_queen_running`
   - Removed auto-start logic (lines 38-42, 59-60)
   - Now fails if queen not running

2. **queen-lifecycle/src/start.rs**
   - Replaced `ensure_queen_running()` with actual start
   - Added binary resolution (matches hive pattern)
   - Now uses `daemon-lifecycle::start_http_daemon()`
   - Added TEAM-318 signatures

---

## Behavior Changes

### Before

```bash
# User runs command without starting queen
./rbee hive list

# OLD BEHAVIOR:
# 1. Checks if queen is running
# 2. Queen not running → auto-starts queen
# 3. Submits job
# 4. Returns result
```

### After

```bash
# User runs command without starting queen
./rbee hive list

# NEW BEHAVIOR:
# 1. Tries to connect to queen
# 2. Connection refused → ERROR
# 3. User sees: "Failed to connect to queen at http://localhost:7833"
# 4. User must run: ./rbee queen start
```

**This is correct.** The user should explicitly start queen, not have it auto-start.

---

## Why This Matters

### 1. Explicit is Better Than Implicit

**Auto-start problems:**
- User doesn't know queen is running
- User can't control when queen starts
- Hidden behavior leads to confusion
- Makes debugging harder

**Explicit start benefits:**
- User knows queen is running
- User controls lifecycle
- Clear error messages
- Easier to debug

### 2. Consistency

**Before:** Different patterns
- Queen: auto-start via ensure
- Hive: explicit start

**After:** Same pattern
- Queen: explicit start
- Hive: explicit start

### 3. "ensure" is Not "start"

**ensure_queen_running():**
- Checks if running
- Starts if not running
- Returns handle

**start_queen():**
- Starts queen
- Fails if already running
- No conditional logic

These are different operations. Don't conflate them.

---

## User Experience

### Old (Confusing)

```bash
$ ./rbee hive list
# (queen auto-starts silently)
# (user doesn't know queen is now running)
# (later: "why is queen running?")
```

### New (Clear)

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

## Verification

```bash
# Compilation
cargo check --bin rbee-keeper
# ✅ PASS (2.49s)

# Test without queen running
./rbee hive list
# ✅ Fails with connection error (no auto-start)

# Test explicit start
./rbee queen start
# ✅ Starts queen

./rbee hive list
# ✅ Works now
```

---

## What About ensure_queen_running()?

**Status:** Still exists in `queen-lifecycle/src/ensure.rs`

**Should it be deleted?** Probably yes, but checking first:

<function_calls>
<invoke name="grep_search">
<parameter name="SearchPath">/home/vince/Projects/llama-orch
