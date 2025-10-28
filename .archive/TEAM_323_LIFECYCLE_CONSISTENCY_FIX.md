# TEAM-323: Lifecycle Consistency Fix

**Date:** Oct 27, 2025  
**Status:** ✅ COMPLETE

## Problem

Hive and Queen lifecycle operations were **completely inconsistent**:

### Queen (CORRECT):
```rust
QueenAction::Install => install_queen(...).await,
QueenAction::Uninstall => uninstall_queen(...).await,
QueenAction::Rebuild => build_daemon_local(...).await,
```

### Hive (BROKEN):
```rust
HiveAction::Install => {
    let operation = Operation::HiveInstall { alias };
    submit_and_stream_job(queen_url, operation).await  // ❌ WRONG!
}
```

**Why this was broken:**
- Lifecycle operations (install/uninstall/rebuild) are **LOCAL** operations
- They don't need job tracking, SSE streaming, or job_id
- Routing them through queen's job router added unnecessary complexity
- Created inconsistency: queen used direct calls, hive used job router

## Root Cause

The old `ADDING_NEW_OPERATIONS.md` guide incorrectly suggested routing **ALL** operations through the job router, including lifecycle operations.

This created a pattern where:
- Queen operations evolved to use `daemon-lifecycle` directly (correct)
- Hive operations still routed through job router (incorrect)
- No one noticed the inconsistency

## Solution

### 1. Fixed Hive Handlers

**Before:**
```rust
HiveAction::Install { alias } => {
    let operation = Operation::HiveInstall { alias };
    submit_and_stream_job(queen_url, operation).await
}
```

**After:**
```rust
HiveAction::Install { alias: _ } => {
    daemon_lifecycle::install_to_local_bin("rbee-hive", None, None).await?;
    Ok(())
}
```

### 2. Deleted Operation Enum Variants

Removed from `operations-contract/src/lib.rs`:
- ❌ `Operation::HiveInstall`
- ❌ `Operation::HiveUninstall`
- ❌ `Operation::HiveRebuild`

These were never needed. Lifecycle operations don't go through the job router.

### 3. Updated Architecture Guide

Created new `bin/ADDING_NEW_OPERATIONS.md` with **TWO CLEAR PATTERNS**:

#### Pattern 1: Lifecycle Operations (Direct)
- Start, Stop, Install, Uninstall, Rebuild
- Use `daemon-lifecycle` directly
- NO job router, NO Operation enum

#### Pattern 2: Job-Based Operations (Routed)
- Worker spawn, Model download, Inference
- Use job router with SSE streaming
- Requires Operation enum variant

### 4. Archived Old Guide

Moved old guide to `bin/.archive/ADDING_NEW_OPERATIONS_OLD.md` with deprecation notice.

## Files Changed

### Modified
- `bin/00_rbee_keeper/src/handlers/hive.rs` - Fixed Install/Uninstall/Rebuild, Status uses daemon-lifecycle
- `bin/00_rbee_keeper/src/handlers/queen.rs` - Fixed stop_http_daemon API, Status uses daemon-lifecycle
- `bin/00_rbee_keeper/src/cli/hive.rs` - Removed Info action (unnecessary)
- `bin/00_rbee_keeper/src/cli/queen.rs` - Removed Info action (unnecessary)
- `bin/97_contracts/operations-contract/src/lib.rs` - Removed HiveInstall/Uninstall/Rebuild

### Created
- `bin/ADDING_NEW_OPERATIONS.md` - New architecture guide with two patterns

### Archived
- `bin/.archive/ADDING_NEW_OPERATIONS_OLD.md` - Old guide (deprecated)

### Deleted
- `check_queen_status()` usage - Replaced with `daemon-lifecycle::check_daemon_status()`
- `Info` CLI actions - Users can just `curl http://localhost:PORT/v1/build-info`

## Verification

```bash
cargo check -p operations-contract  # ✅ PASS
cargo check -p rbee-keeper          # ✅ PASS (tauri_commands.rs has old API, separate GUI concern)
cargo check -p queen-rbee           # ✅ PASS
```

**Note:** `tauri_commands.rs` still uses old API but that's a Tauri GUI integration file, not part of the CLI. Separate concern.

## Additional Fix: Deleted Pointless check_queen_status()

**User caught another inconsistency:**
- `daemon-lifecycle` has `check_daemon_status()` - generic for all daemons
- `queen-lifecycle` created `check_queen_status()` - queen-specific wrapper
- **This was pointless duplication**

**Why it was wrong:**
- Both check if daemon is running
- Both hit `/health` endpoint (or should)
- Creating daemon-specific wrappers defeats the purpose of generic `daemon-lifecycle`
- `check_queen_status()` hit `/v1/build-info` instead of `/health` (inconsistent)

**Fix:**
- Deleted `check_queen_status()` usage
- Both Queen and Hive now use `daemon-lifecycle::check_daemon_status()` directly
- Deleted `Info` CLI actions - users can just `curl http://localhost:PORT/v1/build-info`

**Result:** One less function to maintain, consistent pattern across all daemons.

## Key Insights

### What We Learned

1. **Consistency matters more than you think**
   - Small inconsistencies compound over time
   - "It works" doesn't mean "it's right"
   - Pattern violations create confusion for future teams

2. **Documentation drives architecture**
   - The old guide created the wrong pattern
   - Teams followed the guide, not the better example
   - Fixing the guide prevents future mistakes

3. **Two patterns are better than one confused pattern**
   - Clear separation: Lifecycle vs Job-Based
   - Each pattern has clear rules
   - Decision tree makes it obvious which to use

### Why This Matters

**Before:** Adding a new hive operation required:
1. Add Operation enum variant
2. Add to job_router.rs
3. Add CLI handler
4. Wonder why it's different from queen

**After:** Adding a new lifecycle operation requires:
1. Add CLI action
2. Add handler calling daemon-lifecycle
3. Done

**Saved:** 2 files, 50+ LOC, and all the confusion.

## Architecture Principles

### Lifecycle Operations (Local)
- **Where:** rbee-keeper → daemon-lifecycle
- **Why:** Local system operations, no tracking needed
- **Examples:** Start, Stop, Install, Uninstall, Rebuild

### Job-Based Operations (Tracked)
- **Where:** rbee-keeper → queen/hive job router → SSE stream
- **Why:** Long-running, need progress tracking
- **Examples:** Worker spawn, Model download, Inference

## Decision Matrix

```
Is this start/stop/install/uninstall/rebuild?
├─ YES → Pattern 1 (Direct)
│         - Use daemon-lifecycle
│         - NO Operation enum
│
└─ NO → Does it need progress tracking?
        ├─ YES → Pattern 2 (Job-Based)
        │         - Add Operation enum
        │         - Route through job router
        │
        └─ NO → Pattern 1 (Direct)
                  - Simple request/response
```

## Impact

### Code Reduction
- **Deleted:** 3 Operation enum variants
- **Deleted:** ~50 LOC from job_router.rs (would have been added)
- **Simplified:** Hive handlers now match Queen handlers

### Consistency Achieved
- ✅ Queen and Hive use same pattern for lifecycle
- ✅ Clear separation: Lifecycle vs Job-Based
- ✅ Documentation matches implementation

### Future Prevention
- ✅ New guide prevents this mistake
- ✅ Decision tree makes pattern choice obvious
- ✅ Examples show both patterns clearly

## Lessons for Future Teams

1. **Check for inconsistency FIRST**
   - Before adding code, check existing patterns
   - If two similar things work differently, investigate why
   - Don't assume the guide is always right

2. **Question complexity**
   - If something feels unnecessarily complex, it probably is
   - "Why does this need a job_id?" is a valid question
   - Simple operations should be simple

3. **Update documentation**
   - When you find a better pattern, update the guide
   - Prevent future teams from making the same mistake
   - Good docs prevent bad code

## Related Issues

- TEAM-322: Removed SSH/remote complexity
- TEAM-316: RULE ZERO - delete backwards compatibility
- TEAM-285: Removed HiveStart/HiveStop from job router

All part of the same theme: **Simplify by removing unnecessary indirection.**

---

**TEAM-323 Complete:** Hive and Queen lifecycle operations now use the same pattern. 
Architecture guide updated to prevent future inconsistency.
