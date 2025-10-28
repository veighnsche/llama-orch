# TEAM-320: Consolidate Binary Resolution

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025  
**Impact:** Eliminated 22 LOC of duplicated binary resolution logic

---

## The Problem

**Both queen and hive had duplicate binary resolution functions:**

```rust
// queen-lifecycle/src/start.rs
fn find_queen_binary() -> Result<PathBuf> {
    let home = std::env::var("HOME")?;
    let installed_path = PathBuf::from(format!("{}/.local/bin/queen-rbee", home));
    if installed_path.exists() {
        Ok(installed_path)
    } else {
        DaemonManager::find_in_target("queen-rbee")
    }
}

// hive-lifecycle/src/start.rs
fn find_hive_binary(install_dir: &str) -> Result<PathBuf> {
    if install_dir.contains(".local/bin") {
        let path = PathBuf::from(install_dir).join("rbee-hive");
        if path.exists() {
            Ok(path)
        } else {
            anyhow::bail!("Hive not installed at {}", path.display())
        }
    } else {
        DaemonManager::find_in_target("rbee-hive")
    }
}
```

**Same pattern, different names, pure duplication.**

---

## The Solution

**Add `DaemonManager::find_binary()` that does this for ALL daemons:**

```rust
// daemon-lifecycle/src/manager.rs
impl DaemonManager {
    /// Find a binary (installed or development)
    ///
    /// Search order:
    /// 1. `~/.local/bin/{name}` (installed)
    /// 2. `target/debug/{name}` (development)
    /// 3. `target/release/{name}` (development)
    pub fn find_binary(name: &str) -> Result<PathBuf> {
        // Try installed location first
        if let Ok(home) = std::env::var("HOME") {
            let installed_path = PathBuf::from(format!("{}/.local/bin/{}", home, name));
            if installed_path.exists() {
                return Ok(installed_path);
            }
        }
        
        // Fall back to development builds
        Self::find_in_target(name)
    }
}
```

**Now both queen and hive just call:**
```rust
let binary = DaemonManager::find_binary("queen-rbee")?;
let binary = DaemonManager::find_binary("rbee-hive")?;
```

---

## Files Changed

1. **daemon-lifecycle/src/manager.rs**
   - Added `find_binary()` method (12 LOC)
   - Checks installed location first, then development builds

2. **queen-lifecycle/src/start.rs**
   - Removed `find_queen_binary()` function (10 LOC)
   - Now uses `DaemonManager::find_binary("queen-rbee")`

3. **hive-lifecycle/src/start.rs**
   - Removed `find_hive_binary()` function (12 LOC)
   - Now uses `DaemonManager::find_binary("rbee-hive")`
   - Removed unused `install_dir` parameter from `start_hive_local()`

---

## Code Reduction

| Component | Before | After | Saved |
|-----------|--------|-------|-------|
| find_queen_binary | 10 LOC | 0 | 10 |
| find_hive_binary | 12 LOC | 0 | 12 |
| DaemonManager::find_binary | 0 | 12 LOC | - |
| **Net savings** | **22** | **12** | **10** |

---

## Why This is Better

### 1. Single Source of Truth

**Before:** 2 implementations of binary resolution  
**After:** 1 implementation in `DaemonManager`

### 2. Consistent Behavior

**Before:** Queen and hive had slightly different logic  
**After:** Both use identical logic

### 3. Easier to Extend

**Before:** Adding a new daemon requires copying the pattern  
**After:** Just call `DaemonManager::find_binary("new-daemon")`

### 4. Easier to Modify

**Before:** Change search order → update 2 functions  
**After:** Change search order → update 1 function

---

## Search Order

**All daemons now use this search order:**

1. **Installed:** `~/.local/bin/{name}`
2. **Debug build:** `target/debug/{name}`
3. **Release build:** `target/release/{name}`

This makes sense:
- Installed binaries are preferred (production)
- Debug builds are tried first (development)
- Release builds are fallback (development)

---

## Before vs After

### Before (Duplication)

```rust
// queen-lifecycle
fn find_queen_binary() -> Result<PathBuf> {
    let home = std::env::var("HOME")?;
    let installed = PathBuf::from(format!("{}/.local/bin/queen-rbee", home));
    if installed.exists() { Ok(installed) }
    else { DaemonManager::find_in_target("queen-rbee") }
}

pub async fn start_queen(queen_url: &str) -> Result<()> {
    let binary = find_queen_binary()?;
    // ...
}

// hive-lifecycle
fn find_hive_binary(install_dir: &str) -> Result<PathBuf> {
    if install_dir.contains(".local/bin") {
        let path = PathBuf::from(install_dir).join("rbee-hive");
        if path.exists() { Ok(path) }
        else { anyhow::bail!("Not installed") }
    } else {
        DaemonManager::find_in_target("rbee-hive")
    }
}

pub async fn start_hive(..., install_dir: &str, ...) -> Result<()> {
    let binary = find_hive_binary(install_dir)?;
    // ...
}
```

### After (Consolidated)

```rust
// daemon-lifecycle (shared)
impl DaemonManager {
    pub fn find_binary(name: &str) -> Result<PathBuf> {
        // Check installed, then development
    }
}

// queen-lifecycle
pub async fn start_queen(queen_url: &str) -> Result<()> {
    let binary = DaemonManager::find_binary("queen-rbee")?;
    // ...
}

// hive-lifecycle
pub async fn start_hive(...) -> Result<()> {  // No install_dir needed!
    let binary = DaemonManager::find_binary("rbee-hive")?;
    // ...
}
```

---

## Bonus: Simplified hive-lifecycle API

**Before:** `start_hive_local(install_dir, port, queen_url)`  
**After:** `start_hive_local(port, queen_url)`

The `install_dir` parameter was only used for binary resolution. Now that's handled by `DaemonManager::find_binary()`, we don't need it!

---

## Verification

```bash
# Compilation
cargo check -p daemon-lifecycle -p queen-lifecycle -p hive-lifecycle
# ✅ PASS (0.77s)

# Test binary resolution
./rbee queen start
# ✅ Finds queen binary (installed or development)

./rbee hive start -a localhost
# ✅ Finds hive binary (installed or development)
```

---

## Combined Impact (TEAM-317 through TEAM-320)

| Team | Description | LOC Saved |
|------|-------------|-----------|
| TEAM-317 | Lifecycle parity | 245 |
| TEAM-318 | Remove auto-start | 27 |
| TEAM-319 | SSH duplication | 73 |
| TEAM-320 | Binary resolution | 10 |
| **TOTAL** | | **355** |

---

**Key Insight:** When you see the same pattern in multiple places, it belongs in a shared utility.
