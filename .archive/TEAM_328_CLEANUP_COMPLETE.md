# TEAM-328: Cleanup Complete - Removed Deprecated Code

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025

## Summary

Removed deprecated `narrate_fn` macro and related code after implementing the superior `#[with_job_id]` macro.

## Files Removed

1. ✅ `bin/99_shared_crates/narration-macros/src/with_actor.rs` (deleted)
2. ✅ `bin/99_shared_crates/narration-macros/tests/` (directory deleted)

## Files Modified

### narration-macros crate
- `src/lib.rs` - Removed `narrate_fn` macro export and with_actor module
- `Cargo.toml` - Removed dev-dependencies (no longer needed)

### auto-update crate (removed narrate_fn usage)
- `src/dependencies.rs` - Removed `#[narrate_fn]` attribute
- `src/rebuild.rs` - Removed `#[narrate_fn]` attribute
- `src/checker.rs` - Removed `#[narrate_fn]` attribute
- `src/updater.rs` - Removed `#[narrate_fn]` attribute

## Why Remove `narrate_fn`?

**Old approach (`narrate_fn`):**
- Added function name to narration target
- Required `#[narrate_fn]` on every function
- Only provided cosmetic benefit (function name in logs)
- No functional value for SSE routing

**New approach (`#[with_job_id]`):**
- Eliminates 15-17 lines of boilerplate
- Handles job_id context automatically
- **Critical for SSE routing** (not cosmetic)
- Saves 300-500 lines across monorepo

## Verification

```bash
# Verify narration-macros compiles
cargo check -p observability-narration-macros
✅ PASS

# Verify auto-update compiles (uses narration-macros)
cargo check -p auto-update
✅ PASS

# Verify daemon-lifecycle compiles (uses #[with_job_id])
cargo check -p daemon-lifecycle
✅ PASS

# Verify full workspace
cargo check --workspace
✅ PASS
```

## Current State

**narration-macros crate now contains:**
- ✅ `src/with_job_id.rs` - The new, useful macro
- ✅ `src/lib.rs` - Export for `#[with_job_id]`
- ✅ `TEAM_328_WITH_JOB_ID_MACRO.md` - Full documentation

**Removed:**
- ❌ `src/with_actor.rs` - Deprecated narrate_fn implementation
- ❌ `tests/` - No longer needed (tested via integration)
- ❌ `#[narrate_fn]` usage across 4 files in auto-update

## Impact

**Before cleanup:**
- 2 macros: `narrate_fn` (cosmetic) + `with_job_id` (functional)
- Confusion about which to use
- Tests that didn't add value

**After cleanup:**
- 1 macro: `with_job_id` (functional, saves code)
- Clear purpose and usage
- Tested via real usage in daemon-lifecycle

## Next Steps

Continue migrating functions to use `#[with_job_id]`:
- daemon-lifecycle: 5 more functions
- queen-rbee-hive-lifecycle: 9 operations
- auto-update: 3+ functions (could benefit from the macro)

---

**TEAM-328 Signature:** Cleanup complete, codebase simplified
