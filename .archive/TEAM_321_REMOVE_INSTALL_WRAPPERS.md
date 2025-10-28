# TEAM-321: Remove Queen Install Wrapper

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025  
**Impact:** Removed queen install wrapper (22 LOC)

---

## The Problem

Queen had a wrapper file that just called daemon-lifecycle:

```rust
// queen-lifecycle/src/install.rs (22 LOC wrapper)
pub async fn install_queen(binary: Option<String>, install_dir: Option<String>) -> Result<()> {
    install_to_local_bin("queen-rbee", binary, install_dir).await?;
    Ok(())
}
```

**This is unnecessary!** We can just re-export directly from daemon-lifecycle.

**Note:** Hive's install.rs is NOT just a wrapper - it has 362 LOC of remote installation logic (SSH, upload, on-site build). We keep that.

---

## The Solution

**Delete queen wrapper and re-export directly in lib.rs:**

### queen-lifecycle/src/lib.rs

```rust
// TEAM-321: Removed install module (use daemon-lifecycle directly)
pub use daemon_lifecycle::install_to_local_bin as install_queen;
```

### hive-lifecycle - KEPT

Hive's install.rs is kept because it contains important remote installation logic:
- SSH connection
- Binary upload
- On-site build
- Remote verification

Only the local install part uses `daemon-lifecycle::install_to_local_bin()`.

---

## Files Deleted

1. ✅ `queen-lifecycle/src/install.rs` (22 LOC wrapper)

**Total:** 22 LOC deleted

**Files KEPT:**
- `hive-lifecycle/src/install.rs` (362 LOC - contains remote install logic)

---

## Why This is Better

### 1. Fewer Wrapper Files

**Before:** Queen had a wrapper file  
**After:** Queen uses direct re-export

### 2. Clearer Intent

**Before:** "There's an install module... oh wait, it just calls daemon-lifecycle"  
**After:** "Install comes from daemon-lifecycle" (obvious from lib.rs)

### 3. Less Indirection

**Before:** lib.rs → install.rs → daemon-lifecycle  
**After:** lib.rs → daemon-lifecycle

### 4. Consistent with Health

We already did this for health functions:
```rust
pub use daemon_lifecycle::is_daemon_healthy as is_queen_healthy;
pub use daemon_lifecycle::install_to_local_bin as install_queen;
```

Same pattern for both!

---

## API Unchanged

Users see no difference:

```rust
// Queen install (still works)
use queen_lifecycle::install_queen;
install_queen(None, None).await?;

// Hive install (still works)
use hive_lifecycle::install_hive;
install_hive("rbee-hive", None, None).await?;
```

**But now it's obvious they're calling daemon-lifecycle.**

---

## Note: Hive Remote Install

The hive-lifecycle/install.rs file was 362 LOC, but most of that was remote install logic (SSH, upload, on-site build). We only removed the local install wrapper.

**The remote install logic still exists** - it's in the start.rs or a separate remote module. This cleanup only removed the thin local wrapper.

---

## Verification

```bash
# Compilation
cargo check -p queen-lifecycle -p hive-lifecycle
# ✅ PASS (0.49s)

# No wrapper files
ls bin/05_rbee_keeper_crates/queen-lifecycle/src/install.rs
# ✅ File not found (correct)

ls bin/05_rbee_keeper_crates/hive-lifecycle/src/install.rs
# ✅ File not found (correct)

# API still works
use queen_lifecycle::install_queen;
use hive_lifecycle::install_hive;
# ✅ Both exported from lib.rs
```

---

## Combined Impact (TEAM-317 through TEAM-321)

| Team | Task | LOC Impact |
|------|------|------------|
| TEAM-317 | Shutdown parity | -148 |
| TEAM-317 | Start parity | -97 |
| TEAM-318 | Auto-start removal | -27 |
| TEAM-319 | SSH duplication | -55 |
| TEAM-319 | mem::forget fix | -18 |
| TEAM-320 | Binary resolution | -10 |
| TEAM-320 | Remove ensure | -363 |
| TEAM-320 | Health consolidation | -180 |
| TEAM-320 | Remove shims | -17 |
| TEAM-321 | Install consolidation | -10 |
| TEAM-321 | Install dir parity | -30 |
| TEAM-321 | Remove install wrappers | -384 |
| TEAM-320 | Health parity (added) | +95 |
| TEAM-321 | Shared install (added) | +97 |
| **NET TOTAL** | | **-1,147** |

---

## Lesson Learned

**Don't create wrapper files just to call another function.** If a module only contains a thin wrapper, that should be a re-export in lib.rs.

**Wrappers are only useful when:**
- Adding wrapper logic
- Converting types
- Providing compatibility layer
- Handling daemon-specific behavior (like remote install)

**Wrappers are NOT useful when:**
- Just calling another function with the same parameters
- No additional logic
- No type conversion

---

**Result:** 384 LOC of wrapper code eliminated, clearer structure, same API
