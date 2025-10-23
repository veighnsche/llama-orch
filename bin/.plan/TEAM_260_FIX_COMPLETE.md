# TEAM-260: Auto-Update Dependency Detection Fix

**Date:** Oct 23, 2025  
**Status:** ✅ COMPLETE & TESTED

---

## Problem Summary

Auto-update system was NOT detecting when shared crates changed. This caused:
- Changes to `narration-core` not triggering rebuilds
- Changes to `timeout-enforcer` not triggering rebuilds  
- Changes to ANY dependency not triggering rebuilds
- Users running stale binaries with old code

**Evidence from TEAM-259 testing:**
```bash
# Changed narration-core/src/lib.rs
echo "// test" >> bin/99_shared_crates/narration-core/src/lib.rs

# Ran command
./rbee hive list

# Result: ❌ FAIL
# Auto-update said "✅ Binary rbee-keeper is up-to-date"
# But file timestamps showed source was NEWER than binary!
```

---

## Root Cause Analysis

### The Bug

Dependency paths from `Cargo.toml` are **relative to the crate directory**, not the workspace root.

Example from `bin/00_rbee_keeper/Cargo.toml`:
```toml
[dependencies]
narration-core = { path = "../99_shared_crates/narration-core" }
```

The old code did:
1. Read path: `"../99_shared_crates/narration-core"`
2. Store as-is in `all_deps`
3. In `needs_rebuild()`: `workspace_root.join(dep_path)`
4. This created: `workspace_root/../99_shared_crates/narration-core`
5. The `"../"` put it OUTSIDE the workspace → path not found
6. `is_dir_newer()` returned `false` → no rebuild triggered

### Why It Seemed To Work Initially

- Changes to the binary's own source directory were detected correctly
- Only dependency changes were missed
- Tests that only modified the main crate passed
- Bug only appeared when testing cross-crate dependencies

---

## The Fix

### Code Location
`bin/99_shared_crates/auto-update/src/lib.rs` - Lines 427-451

### Solution

Properly resolve dependency paths:

1. Start with crate directory: `workspace_root/bin/00_rbee_keeper`
2. Join with relative path: `../99_shared_crates/narration-core`
3. Canonicalize to resolve `".."`: `/home/user/Projects/llama-orch/bin/99_shared_crates/narration-core`
4. Strip workspace root: `bin/99_shared_crates/narration-core`
5. Store workspace-relative path

### Key Code Change

```rust
// OLD (BROKEN):
let dep_path = PathBuf::from(path);  // Just store "../99_shared_crates/narration-core"
all_deps.push(dep_path);

// NEW (FIXED):
let crate_dir = workspace_root.join(source_dir);
let dep_absolute = crate_dir.join(path);

// Canonicalize to resolve ".." components
let dep_canonical = dep_absolute.canonicalize()
    .with_context(|| format!("Failed to resolve dependency path: {}", dep_absolute.display()))?;

// Convert back to relative path from workspace_root
let dep_relative = dep_canonical
    .strip_prefix(workspace_root)
    .with_context(|| format!("Dependency {} is outside workspace", dep_canonical.display()))?
    .to_path_buf();

all_deps.push(dep_relative.clone());
```

---

## Testing Results

### Test 1: narration-core Dependency Change
```bash
# Make change
echo "// test" >> bin/99_shared_crates/narration-core/src/lib.rs

# Run command
./rbee hive list

# Result: ✅ PASS
[auto-upd  ] check_rebuild  : 🔨 Dependency bin/99_shared_crates/narration-core changed, rebuild needed
🔨 Building rbee-keeper...
```

### Test 2: timeout-enforcer Dependency Change
```bash
# Make change
echo "// test" >> bin/99_shared_crates/timeout-enforcer/src/lib.rs

# Run command
./rbee hive list

# Result: ✅ PASS
[auto-upd  ] check_rebuild  : 🔨 Dependency bin/99_shared_crates/timeout-enforcer changed, rebuild needed
🔨 Building rbee-keeper...
```

### Test 3: No Changes (Baseline)
```bash
# No changes made

# Run command
./rbee hive list

# Result: ✅ PASS
[auto-upd  ] check_rebuild  : ✅ Binary rbee-keeper is up-to-date
```

---

## Files Modified

### Primary Fix
- **bin/99_shared_crates/auto-update/src/lib.rs** (+57 LOC)
  - Added comprehensive bug fix documentation (debugging-rules.md compliant)
  - Fixed `collect_deps_recursive()` to properly resolve dependency paths
  - All path resolution now uses workspace-relative paths

---

## Documentation Standards

This fix follows `.windsurf/rules/debugging-rules.md`:

✅ Full bug fix comment block at the fix location  
✅ SUSPICION section (what we initially thought)  
✅ INVESTIGATION section (what we tested/ruled out)  
✅ ROOT CAUSE section (the actual cause)  
✅ FIX section (what we changed and why)  
✅ TESTING section (how we verified the fix)  

---

## Impact

### Before Fix
- ❌ Dependency changes not detected
- ❌ Stale binaries running old code
- ❌ Manual rebuilds required
- ❌ Developers confused by "up-to-date" messages

### After Fix
- ✅ All dependency changes detected correctly
- ✅ Automatic rebuilds when dependencies change
- ✅ Transitive dependencies tracked recursively
- ✅ Clear narration messages show which dependency changed

---

## Verification Commands

To verify the fix works on your machine:

```bash
# 1. Ensure everything is built
cargo build --bin rbee-keeper

# 2. Verify baseline (should be up-to-date)
./rbee hive list | grep check_rebuild
# Expected: "✅ Binary rbee-keeper is up-to-date"

# 3. Make a test change to a dependency
echo "// test comment" >> bin/99_shared_crates/narration-core/src/lib.rs

# 4. Run command again
./rbee hive list | grep check_rebuild
# Expected: "🔨 Dependency bin/99_shared_crates/narration-core changed, rebuild needed"
# Expected: "🔨 Building rbee-keeper..."

# 5. Clean up
git checkout bin/99_shared_crates/narration-core/src/lib.rs
```

---

## Related Issues

This fix resolves:
- TEAM-259 test failure (auto-update not detecting changes)
- Future reliability of auto-update system
- Cross-crate dependency tracking

---

## Next Steps

✅ Fix is complete and tested  
✅ All tests passing  
✅ Documentation complete  

**READY FOR:** Integration into daemon-lifecycle (as planned by TEAM-259)

---

## Summary

**Problem:** Auto-update didn't detect dependency changes  
**Root Cause:** Relative paths from Cargo.toml not resolved correctly  
**Fix:** Properly canonicalize and resolve all dependency paths  
**Testing:** Verified with multiple dependencies (narration-core, timeout-enforcer)  
**Status:** ✅ COMPLETE - Auto-update now works as designed
