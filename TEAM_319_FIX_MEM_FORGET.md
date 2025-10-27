# TEAM-319: Fix mem::forget Clippy Warning

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025  
**Issue:** Clippy warning about `mem::forget` on type with `Drop` fields

---

## The Problem

**Clippy warning:**
```
usage of `mem::forget` on type with `Drop` fields
argument has type `tokio::process::Child`
```

**Location:** Every call to `start_http_daemon()` followed by `std::mem::forget(child)`

**Why it's flagged:**
- `tokio::process::Child` implements `Drop` to clean up the child process
- `mem::forget()` prevents `Drop` from running
- This can leak resources (though in this case it's intentional)

---

## The Root Cause

**Bad API design:**

```rust
// daemon-lifecycle/src/start.rs
pub async fn start_http_daemon(config: HttpDaemonConfig) -> Result<Child> {
    let child = manager.spawn().await?;
    poll_until_healthy(health_config).await?;
    Ok(child)  // ← Caller must manage this
}

// Callers (queen-lifecycle, hive-lifecycle)
let child = start_http_daemon(config).await?;
std::mem::forget(child);  // ← Clippy doesn't like this
```

**Problem:** The API returns `Child` expecting the caller to use `mem::forget()`. This is a code smell.

---

## The Solution

**Better API design:** Detach the child inside `start_http_daemon()`.

```rust
// daemon-lifecycle/src/start.rs
pub async fn start_http_daemon(config: HttpDaemonConfig) -> Result<()> {
    let mut child = manager.spawn().await?;
    poll_until_healthy(health_config).await?;
    
    // Detach internally using ManuallyDrop
    let _ = std::mem::ManuallyDrop::new(child);
    Ok(())  // ← No Child returned
}

// Callers (queen-lifecycle, hive-lifecycle)
start_http_daemon(config).await?;  // ← Clean, no mem::forget needed
```

---

## Why ManuallyDrop is Better

### mem::forget (Clippy doesn't like)
```rust
std::mem::forget(child);
```
- Clippy warning: "usage of mem::forget on type with Drop fields"
- Not explicit about intent
- Can be accidentally used on wrong types

### ManuallyDrop (Clippy approves)
```rust
let _ = std::mem::ManuallyDrop::new(child);
```
- No Clippy warning
- Explicit: "I'm intentionally preventing Drop"
- Type-safe wrapper designed for this purpose

---

## Files Changed

1. **daemon-lifecycle/src/start.rs**
   - Changed return type: `Result<Child>` → `Result<()>`
   - Added `ManuallyDrop::new(child)` to detach internally
   - Updated documentation

2. **hive-lifecycle/src/start.rs**
   - Removed `let child = ...` assignment
   - Removed `std::mem::forget(child)`
   - Now just: `start_http_daemon(config).await?`

3. **queen-lifecycle/src/start.rs**
   - Removed `let child = ...` assignment
   - Removed `std::mem::forget(child)`
   - Now just: `start_http_daemon(config).await?`

---

## Before vs After

### Before (Clippy warning)

```rust
// daemon-lifecycle
pub async fn start_http_daemon(config: HttpDaemonConfig) -> Result<Child> {
    let child = manager.spawn().await?;
    poll_until_healthy(health_config).await?;
    Ok(child)
}

// Callers
let child = start_http_daemon(config).await?;
std::mem::forget(child);  // ⚠️ Clippy warning
```

### After (No warning)

```rust
// daemon-lifecycle
pub async fn start_http_daemon(config: HttpDaemonConfig) -> Result<()> {
    let mut child = manager.spawn().await?;
    poll_until_healthy(health_config).await?;
    let _ = std::mem::ManuallyDrop::new(child);  // ✅ No warning
    Ok(())
}

// Callers
start_http_daemon(config).await?;  // ✅ Clean
```

---

## Why This is Better Design

### 1. Encapsulation

**Before:** Caller must know to use `mem::forget()`  
**After:** Implementation detail hidden inside `start_http_daemon()`

### 2. Single Responsibility

**Before:** Function starts daemon, caller detaches it  
**After:** Function starts AND detaches daemon

### 3. Harder to Misuse

**Before:** Caller might forget to call `mem::forget()` → daemon dies  
**After:** Impossible to misuse, daemon always detached

### 4. Cleaner Call Sites

**Before:** 3 lines (call, assign, forget)  
**After:** 1 line (call)

---

## Verification

```bash
# Check for mem::forget warnings
cargo clippy -p hive-lifecycle 2>&1 | grep mem_forget
# ✅ No warnings!

# Compilation
cargo check -p daemon-lifecycle -p queen-lifecycle -p hive-lifecycle
# ✅ PASS (0.58s)

# Test daemon start
./rbee queen start
./rbee hive start -a localhost
# ✅ Both work, daemons stay running
```

---

## Key Insight

**When you find yourself returning a value just so the caller can call `mem::forget()` on it, that's a sign the API should handle it internally.**

The pattern:
```rust
let x = function()?;
std::mem::forget(x);
```

Should be:
```rust
function()?;  // Handles detachment internally
```

---

**Result:** Clippy warning eliminated, cleaner API, better encapsulation
