# Hive Catalog Removal Notice

**Date:** 2025-10-22  
**Teams:** TEAM-193 through TEAM-199 (originally planned as TEAM-188 through TEAM-194)

## What Was Removed

The SQLite-based `hive-catalog` crate has been completely removed and replaced with file-based configuration.

## Removed Components

- `bin/15_queen_rbee_crates/hive-catalog/` (entire crate)
- `bin/15_queen_rbee_crates/scheduler/` (unused crate that depended on hive-catalog)
- SQLite dependencies (`rusqlite` or `sqlx`)
- Database files (`*.db`)
- Old CLI arguments (`--id`, `--ssh-host`, etc.)
- Dead code in `hive-lifecycle` crate (`execute_hive_start`, `spawn_hive` functions)

## Replacement

File-based configuration in `~/.config/rbee/`:
- `config.toml` - Queen settings
- `hives.conf` - Hive definitions (SSH config style)
- `capabilities.yaml` - Auto-generated capabilities

## Migration

See `docs/MIGRATION_GUIDE.md` for migration instructions.

## Why This Change?

1. **Simpler:** Text files are easier to understand than SQLite
2. **Transparent:** Users can see and edit config directly
3. **Standard:** Follows Unix conventions (SSH config style)
4. **Version Control:** Config files can be committed to git
5. **No Database:** No SQLite dependency or database corruption issues

## Rollback

Not supported. The old system is completely removed.

## Teams Involved

- TEAM-193: Config parser (Phase 1)
- TEAM-194: Replace SQLite in job_router (Phase 2)
- TEAM-195: Preflight validation (Phase 3)
- TEAM-196: Capabilities auto-generation (Phase 4)
- TEAM-197: Code peer review (Phase 5)
- TEAM-198: Documentation (Phase 6)
- TEAM-199: Self-destruct (Phase 7 - this phase)

## Lines of Code

- **Removed:** ~1,000 LOC (hive-catalog crate + scheduler crate + dead code)
- **Added:** ~600 LOC (rbee-config crate)
- **Net:** -400 LOC

## Verification

```bash
# Verify crate is gone
ls bin/15_queen_rbee_crates/hive-catalog/
# Should return: No such file or directory

# Verify scheduler is gone
ls bin/15_queen_rbee_crates/scheduler/
# Should return: No such file or directory

# Verify no references remain
grep -r "hive-catalog" --include="*.toml" .
# Should return: No results

# Verify builds work
cargo build --workspace
# Should succeed
```

---

**Status:** ✅ Complete  
**Archived by:** TEAM-199 (Phase 7)  
**Date Completed:** 2025-10-22
