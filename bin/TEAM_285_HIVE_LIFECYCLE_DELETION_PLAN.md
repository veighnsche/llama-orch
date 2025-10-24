# TEAM-285: Hive Lifecycle Operations Deletion Plan

**Date:** Oct 24, 2025  
**Status:** 🔴 IN PROGRESS

## Mission

**COMPLETELY DELETE** all hive lifecycle operations (HiveStart, HiveStop, HiveEnsure).

These operations are no longer needed because rbee is localhost-only. There's no need to start/stop remote hives.

## Operations to Delete

- ❌ `HiveStart` - Start a hive daemon
- ❌ `HiveStop` - Stop a hive daemon  
- ❌ `HiveEnsure` - Ensure hive is running (not in operations-contract, only in lifecycle crate)

## The 3-File Deletion Pattern (Reverse of Adding)

Following the guide in `ADDING_NEW_OPERATIONS.md`, we delete in reverse order:

### Step 1: Remove from CLI (rbee-keeper)
**File:** `bin/00_rbee_keeper/src/handlers/hive.rs`

Remove:
- `HiveAction::Start` enum variant
- `HiveAction::Stop` enum variant
- Match arms that create `Operation::HiveStart` and `Operation::HiveStop`

### Step 2: Remove from Router (queen-rbee)
**File:** `bin/10_queen_rbee/src/job_router.rs`

Remove:
- Import: `execute_hive_start`, `execute_hive_stop`
- Import: `HiveStartRequest`, `HiveStopRequest`
- Match arm: `Operation::HiveStart { alias } => { ... }`
- Match arm: `Operation::HiveStop { alias } => { ... }`

### Step 3: Remove from Contract (operations-contract)
**File:** `bin/97_contracts/operations-contract/src/lib.rs`

Remove:
- Enum variant: `HiveStart { alias: String }`
- Enum variant: `HiveStop { alias: String }`
- From `Operation::name()`: cases for HiveStart, HiveStop
- From `Operation::hive_id()`: cases for HiveStart, HiveStop
- Tests: `test_serialize_hive_start()`, `test_hive_start_defaults_to_localhost()`, etc.

### Step 4: Delete Lifecycle Crate Files
**Directory:** `bin/15_queen_rbee_crates/hive-lifecycle/src/`

Delete completely:
- ❌ `ensure.rs` - HiveEnsure implementation
- ❌ `start.rs` - HiveStart implementation
- ❌ `stop.rs` - HiveStop implementation

Update:
- `lib.rs` - Remove exports for these modules

## Files to Modify

1. `bin/97_contracts/operations-contract/src/lib.rs` - Remove enum variants, tests
2. `bin/10_queen_rbee/src/job_router.rs` - Remove imports, match arms
3. `bin/00_rbee_keeper/src/handlers/hive.rs` - Remove CLI commands
4. `bin/00_rbee_keeper/src/main.rs` - Remove HiveAction enum variants
5. `bin/15_queen_rbee_crates/hive-lifecycle/src/lib.rs` - Remove module exports
6. `bin/15_queen_rbee_crates/hive-lifecycle/src/ensure.rs` - DELETE FILE
7. `bin/15_queen_rbee_crates/hive-lifecycle/src/start.rs` - DELETE FILE
8. `bin/15_queen_rbee_crates/hive-lifecycle/src/stop.rs` - DELETE FILE

## Verification

After deletion:
- [ ] `cargo check -p operations-contract` passes
- [ ] `cargo check -p queen-rbee` passes
- [ ] `cargo check -p rbee-keeper` passes
- [ ] `cargo check -p queen-rbee-hive-lifecycle` passes
- [ ] `./rbee hive --help` does NOT show start/stop commands
- [ ] All tests pass

## Rationale

**Why delete these operations?**

1. **Localhost-only architecture** - rbee no longer manages remote hives
2. **No SSH/remote operations** - TEAM-284 removed all remote functionality
3. **Simplified deployment** - Users run hives manually, not via queen
4. **Reduced complexity** - Less code to maintain

**What remains?**
- ✅ `HiveList` - List configured hives (from hives.conf)
- ✅ `HiveGet` - Get hive details
- ✅ `HiveStatus` - Check hive health
- ✅ `HiveRefreshCapabilities` - Update hive capabilities

These are **query operations** that don't manage lifecycle, just read state.

## Impact Analysis

**Breaking Changes:**
- CLI commands `rbee hive start` and `rbee hive stop` will no longer exist
- Users must start/stop hives manually (e.g., `systemd`, `docker`, or direct execution)

**Migration Path:**
- Document how to run hives manually
- Update README with new deployment instructions
- Remove references to lifecycle management from docs

## Next Steps

1. Execute deletions in order (CLI → Router → Contract → Crate files)
2. Update documentation to reflect localhost-only architecture
3. Verify all packages compile
4. Update `ADDING_NEW_OPERATIONS.md` to remove HiveStart/HiveStop from examples
