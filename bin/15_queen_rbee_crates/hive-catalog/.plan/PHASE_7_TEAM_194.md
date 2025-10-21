# Phase 7: Self-Destruct

**Team:** TEAM-194  
**Duration:** 1-2 hours  
**Dependencies:** Phase 6 (TEAM-193) complete  
**Deliverables:** Complete removal of `hive-catalog` crate and all SQLite dependencies

---

## Mission

**DESTROY THE OLD SYSTEM.** Remove all traces of the SQLite-based hive catalog. This is the final cleanup phase.

---

## Self-Destruct Checklist

### 7.1 Remove hive-catalog Crate

#### Delete the Crate Directory

```bash
# Nuclear option: delete the entire crate
rm -rf bin/15_queen_rbee_crates/hive-catalog/
```

**Verify deletion:**
```bash
# Should return "No such file or directory"
ls bin/15_queen_rbee_crates/hive-catalog/
```

#### Remove from Workspace

**In root `Cargo.toml`:**

```toml
[workspace]
members = [
    # ... other members ...
    # REMOVE THIS LINE:
    # "bin/15_queen_rbee_crates/hive-catalog",
]
```

### 7.2 Remove Dependencies

#### Remove from queen-rbee

**In `bin/10_queen_rbee/Cargo.toml`:**

```toml
[dependencies]
# REMOVE THIS LINE:
# queen-rbee-hive-catalog = { path = "../15_queen_rbee_crates/hive-catalog" }

# Verify rbee-config is present:
rbee-config = { path = "../15_queen_rbee_crates/rbee-config" }
```

#### Check for Other References

Search for any remaining references:

```bash
# Search for hive-catalog references
grep -r "hive-catalog" --include="*.toml" .

# Search for hive_catalog imports
grep -r "use.*hive_catalog" --include="*.rs" .

# Search for HiveCatalog type
grep -r "HiveCatalog" --include="*.rs" .
```

**All of these should return NO results.**

### 7.3 Remove SQLite Dependencies

#### Check for SQLite Usage

```bash
# Search for rusqlite or sqlx
grep -r "rusqlite\|sqlx" --include="*.toml" .

# Search for SQL queries
grep -r "CREATE TABLE\|INSERT INTO\|SELECT.*FROM" --include="*.rs" .
```

**If found in `hive-catalog` context, remove them.**

#### Clean Up Unused Dependencies

```bash
# Check for unused dependencies
cargo machete

# Or manually check each Cargo.toml
```

### 7.4 Remove Old Database Files

#### Delete SQLite Databases

```bash
# Find and list any .db files
find . -name "*.db" -type f

# Common locations:
rm -f ~/.config/rbee/hive-catalog.db
rm -f ~/.local/share/rbee/hive-catalog.db
rm -f ./hive-catalog.db
rm -f ./queen-hive-catalog.db
```

**Document locations for users:**

Create `docs/CLEANUP_OLD_DATABASE.md`:

```markdown
# Cleanup Old Database Files

If you previously used the SQLite-based hive catalog, you can safely delete these files:

```bash
# Common locations
rm -f ~/.config/rbee/hive-catalog.db
rm -f ~/.local/share/rbee/hive-catalog.db
rm -f ./hive-catalog.db
rm -f ./queen-hive-catalog.db
```

These files are no longer used after migrating to file-based config.
```

### 7.5 Remove Old Code References

#### Search for Dead Code

```bash
# Search for catalog-related functions
grep -r "catalog\." --include="*.rs" bin/10_queen_rbee/src/

# Search for old operation handlers
grep -r "hive_catalog" --include="*.rs" .
```

**Remove any dead code found.**

#### Update Imports

Verify no old imports remain:

```bash
# Should return NO results:
grep -r "use.*hive_catalog" --include="*.rs" .
grep -r "queen_rbee_hive_catalog" --include="*.rs" .
```

### 7.6 Clean Build Artifacts

```bash
# Clean all build artifacts
cargo clean

# Rebuild to verify everything works
cargo build --workspace

# Run tests
cargo test --workspace
```

### 7.7 Update Documentation

#### Remove Old References

Search documentation for old references:

```bash
# Search docs for old patterns
grep -r "hive_catalog\|HiveCatalog" --include="*.md" .

# Search for old CLI examples
grep -r "\-\-id.*\-\-ssh-host" --include="*.md" .
```

**Update or remove any found references.**

#### Add Deprecation Notice

**Create:** `.archive/HIVE_CATALOG_REMOVED.md`

```markdown
# Hive Catalog Removal Notice

**Date:** 2025-10-21  
**Teams:** TEAM-188 through TEAM-194

## What Was Removed

The SQLite-based `hive-catalog` crate has been completely removed and replaced with file-based configuration.

## Removed Components

- `bin/15_queen_rbee_crates/hive-catalog/` (entire crate)
- SQLite dependencies (`rusqlite` or `sqlx`)
- Database files (`*.db`)
- Old CLI arguments (`--id`, `--ssh-host`, etc.)

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

- TEAM-188: Config parser
- TEAM-189: Replace SQLite in job_router
- TEAM-190: Preflight validation
- TEAM-191: Capabilities auto-generation
- TEAM-192: Code peer review
- TEAM-193: Documentation
- TEAM-194: Self-destruct (this phase)

## Lines of Code

- **Removed:** ~800 LOC (hive-catalog crate)
- **Added:** ~600 LOC (rbee-config crate)
- **Net:** -200 LOC

## Verification

```bash
# Verify crate is gone
ls bin/15_queen_rbee_crates/hive-catalog/
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
**Archived by:** TEAM-194
```

### 7.8 Git Cleanup

#### Commit the Removal

```bash
# Stage the deletion
git rm -rf bin/15_queen_rbee_crates/hive-catalog/

# Stage Cargo.toml changes
git add Cargo.toml
git add bin/10_queen_rbee/Cargo.toml

# Stage documentation
git add docs/
git add .archive/HIVE_CATALOG_REMOVED.md

# Commit
git commit -m "feat: Remove SQLite-based hive-catalog, replace with file-based config

BREAKING CHANGE: hive-catalog crate removed

- Removed bin/15_queen_rbee_crates/hive-catalog/
- Replaced with rbee-config (file-based)
- Updated CLI to use -h <alias> instead of --id <id>
- See docs/MIGRATION_GUIDE.md for migration steps

Teams: TEAM-188 through TEAM-194
Phases: 1-7 complete"
```

#### Tag the Release

```bash
# Tag as breaking change
git tag -a v0.2.0 -m "BREAKING: Remove SQLite hive-catalog

- File-based config replaces SQLite
- New CLI arguments
- See MIGRATION_GUIDE.md"

# Push
git push origin main --tags
```

---

## Verification Checklist

### Build Verification

- [ ] `cargo clean` completes
- [ ] `cargo build --workspace` succeeds
- [ ] `cargo test --workspace` passes
- [ ] `cargo clippy --workspace` has no warnings
- [ ] `cargo doc --workspace` builds successfully

### File Verification

- [ ] `bin/15_queen_rbee_crates/hive-catalog/` does not exist
- [ ] No `*.db` files in workspace
- [ ] No `hive-catalog` references in `Cargo.toml` files
- [ ] No `hive_catalog` imports in Rust files

### Functional Verification

- [ ] `./rbee hive list` works (reads from `hives.conf`)
- [ ] `./rbee hive install -h <alias>` works
- [ ] `./rbee hive start -h <alias>` works
- [ ] `./rbee hive stop -h <alias>` works
- [ ] Old CLI args (`--id`, `--ssh-host`) are rejected

### Documentation Verification

- [ ] `docs/HIVE_CONFIGURATION.md` exists
- [ ] `docs/MIGRATION_GUIDE.md` exists
- [ ] `docs/HIVE_QUICK_REFERENCE.md` exists
- [ ] `.archive/HIVE_CATALOG_REMOVED.md` exists
- [ ] No broken links in documentation

---

## Final Verification Commands

```bash
# 1. Verify crate is gone
test ! -d bin/15_queen_rbee_crates/hive-catalog && echo "✅ Crate deleted" || echo "❌ Crate still exists"

# 2. Verify no references
! grep -r "hive-catalog" --include="*.toml" . && echo "✅ No Cargo.toml references" || echo "❌ References found"

# 3. Verify no imports
! grep -r "hive_catalog" --include="*.rs" . && echo "✅ No Rust imports" || echo "❌ Imports found"

# 4. Build workspace
cargo build --workspace && echo "✅ Build successful" || echo "❌ Build failed"

# 5. Run tests
cargo test --workspace && echo "✅ Tests pass" || echo "❌ Tests failed"

# 6. Check clippy
cargo clippy --workspace -- -D warnings && echo "✅ No clippy warnings" || echo "❌ Clippy warnings found"

# 7. Verify docs exist
test -f docs/HIVE_CONFIGURATION.md && \
test -f docs/MIGRATION_GUIDE.md && \
test -f docs/HIVE_QUICK_REFERENCE.md && \
echo "✅ Documentation complete" || echo "❌ Documentation missing"
```

---

## Success Criteria

- [ ] `hive-catalog` crate completely removed
- [ ] All SQLite dependencies removed
- [ ] All old database files deleted
- [ ] Workspace builds successfully
- [ ] All tests pass
- [ ] No clippy warnings
- [ ] Documentation is complete
- [ ] Git commit is clean
- [ ] Breaking change is tagged

---

## Post-Destruction Report

**Destroyed by:** TEAM-194  
**Date:** [Date]  
**Duration:** [Hours]

### What Was Destroyed

- ✅ `bin/15_queen_rbee_crates/hive-catalog/` (entire crate)
- ✅ SQLite dependencies
- ✅ Database files
- ✅ Old CLI arguments
- ✅ Dead code references

### What Was Created

- ✅ `rbee-config` crate (file-based)
- ✅ User documentation
- ✅ Migration guide
- ✅ Deprecation notice

### Metrics

- **Files Deleted:** [Count]
- **Lines Removed:** [Count]
- **Dependencies Removed:** [Count]
- **Build Time:** [Before] → [After]

### Verification

All verification commands passed: **YES / NO**

### Sign-off

- [ ] All old code removed
- [ ] New system works
- [ ] Documentation complete
- [ ] Ready for production

---

## Celebration

```
🎉 MISSION ACCOMPLISHED 🎉

The SQLite-based hive-catalog has been completely removed.
Long live file-based configuration!

Teams TEAM-188 through TEAM-194:
You have successfully migrated rbee to a simpler, more maintainable system.

Total effort: ~25-35 hours across 7 phases
Result: Cleaner codebase, better UX, easier maintenance

Well done! 🚀
```

---

**Created by:** TEAM-187  
**For:** TEAM-194  
**Status:** 📋 Ready to implement  
**Final Phase:** 💣 Self-Destruct Sequence
