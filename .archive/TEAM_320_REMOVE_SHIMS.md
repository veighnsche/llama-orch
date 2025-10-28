# TEAM-320: Remove Health Shims

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025  
**Impact:** Removed 2 unnecessary shim files

---

## The Problem

We created two shim files that just re-exported from daemon-lifecycle:

```rust
// queen-lifecycle/src/health.rs (9 LOC shim)
pub use daemon_lifecycle::is_daemon_healthy as is_queen_healthy;
pub use daemon_lifecycle::poll_until_healthy;
pub use daemon_lifecycle::HealthPollConfig;

// hive-lifecycle/src/health.rs (8 LOC shim)
pub use daemon_lifecycle::is_daemon_healthy as is_hive_healthy;
pub use daemon_lifecycle::poll_until_healthy;
pub use daemon_lifecycle::HealthPollConfig;
```

**This is unnecessary!** We can just re-export directly in lib.rs.

---

## The Solution

**Delete both shim files and re-export directly in lib.rs:**

### queen-lifecycle/src/lib.rs

```rust
// TEAM-320: Import health functions directly from daemon-lifecycle
pub use daemon_lifecycle::is_daemon_healthy as is_queen_healthy;
pub use daemon_lifecycle::poll_until_healthy;
pub use daemon_lifecycle::HealthPollConfig;
```

### hive-lifecycle/src/lib.rs

```rust
// TEAM-320: Import health functions directly from daemon-lifecycle
pub use daemon_lifecycle::is_daemon_healthy as is_hive_healthy;
pub use daemon_lifecycle::poll_until_healthy;
pub use daemon_lifecycle::HealthPollConfig;
```

---

## Files Deleted

1. ✅ `queen-lifecycle/src/health.rs` (9 LOC shim)
2. ✅ `hive-lifecycle/src/health.rs` (8 LOC shim)

**Total:** 17 LOC of unnecessary shim code removed

---

## Why This is Better

### 1. Fewer Files

**Before:** 2 shim files  
**After:** 0 shim files

### 2. Clearer Intent

**Before:** "There's a health module... oh wait, it just re-exports"  
**After:** "Health functions come from daemon-lifecycle" (obvious from lib.rs)

### 3. Less Indirection

**Before:** lib.rs → health.rs → daemon-lifecycle  
**After:** lib.rs → daemon-lifecycle

### 4. Easier to Maintain

**Before:** Update health exports → update 2 shim files + 2 lib.rs files  
**After:** Update health exports → update 2 lib.rs files

---

## API Unchanged

Users see no difference:

```rust
// Queen health check (still works)
use queen_lifecycle::is_queen_healthy;
is_queen_healthy("http://localhost:7833").await?;

// Hive health check (still works)
use hive_lifecycle::is_hive_healthy;
is_hive_healthy("http://localhost:7835").await?;
```

---

## Verification

```bash
# Compilation
cargo check -p queen-lifecycle -p hive-lifecycle
# ✅ PASS (0.40s)

# No shim files
ls bin/05_rbee_keeper_crates/queen-lifecycle/src/health.rs
# ✅ File not found (correct)

ls bin/05_rbee_keeper_crates/hive-lifecycle/src/health.rs
# ✅ File not found (correct)
```

---

## Updated Total Impact

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
| TEAM-320 | Health parity (added) | +95 |
| **NET TOTAL** | | **-820** |

---

## Lesson Learned

**Don't create shim files just to re-export.** If a module only contains `pub use` statements, those should go directly in lib.rs.

**Shims are only useful when:**
- Adding wrapper logic
- Converting types
- Providing compatibility layer

**Shims are NOT useful when:**
- Just re-exporting (use lib.rs)
- No additional logic
- No type conversion

---

**Result:** 17 LOC of shim code eliminated, clearer structure, same API
