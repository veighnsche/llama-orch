# Worker Crates Migration — Quick Start

## TL;DR

```bash
# 1. Preview what will happen (SAFE, no changes)
./tools/worker-crates-migration/migrate-worker-gguf.sh --dry-run

# 2. Execute migration (creates backup first)
./tools/worker-crates-migration/migrate-worker-gguf.sh

# 3. If something goes wrong, rollback
git reset --hard migration-backup
```

## Safety First

✅ **Always run `--dry-run` first**  
✅ **Backup branch created automatically**  
✅ **Git history preserved via `git mv`**  
✅ **Compilation verified before commit**  
✅ **Easy rollback available**

## Step-by-Step Guide

### Step 1: Preview Changes (Dry Run)

```bash
cd /home/vince/Projects/llama-orch

# Preview worker-gguf migration
./tools/worker-crates-migration/migrate-worker-gguf.sh --dry-run
```

**What you'll see:**
- All file moves that will happen
- Import changes in worker-orcd
- Cargo.toml updates
- Compilation and test commands
- Commit message preview

**No changes are made!** This is 100% safe.

### Step 2: Review Dry Run Output

Check the output carefully:
- ✓ Source file exists
- ✓ Destination path is correct
- ✓ Import replacements look right
- ✓ Cargo.toml changes are correct

### Step 3: Execute Migration

```bash
# Run the actual migration
./tools/worker-crates-migration/migrate-worker-gguf.sh
```

**What happens:**
1. Creates backup branch `migration-backup`
2. Moves file with `git mv` (preserves history)
3. Updates imports in worker-orcd
4. Updates Cargo.toml
5. Verifies compilation (both crates)
6. Runs tests (both crates)
7. Commits changes with descriptive message

**Duration:** ~2-3 minutes (including compilation)

### Step 4: Verify Results

```bash
# Check compilation
cargo check -p worker-gguf
cargo check -p worker-orcd

# Run tests
cargo test -p worker-gguf
cargo test -p worker-orcd

# Verify git history preserved
git log --follow bin/worker-crates/worker-gguf/src/lib.rs

# Review commit
git show
```

### Step 5: Rollback (If Needed)

If something goes wrong:

```bash
# Option 1: Reset to backup branch
git reset --hard migration-backup

# Option 2: Undo last commit (if already committed)
git reset --soft HEAD~1

# Option 3: Revert the commit (keeps history)
git revert HEAD
```

## Migration Order

Run migrations in this order (respects dependencies):

1. ✅ **worker-gguf** (no dependencies)
   ```bash
   ./tools/worker-crates-migration/migrate-worker-gguf.sh
   ```

2. ⏳ **worker-tokenizer** (no dependencies) [TODO: script not created yet]
   ```bash
   ./tools/worker-crates-migration/migrate-worker-tokenizer.sh
   ```

3. ⏳ **worker-models** (depends on worker-gguf) [TODO: script not created yet]
   ```bash
   ./tools/worker-crates-migration/migrate-worker-models.sh
   ```

4. ⏳ **worker-common** (no dependencies) [TODO: script not created yet]
   ```bash
   ./tools/worker-crates-migration/migrate-worker-common.sh
   ```

5. ⏳ **worker-http** (depends on worker-common) [TODO: script not created yet]
   ```bash
   ./tools/worker-crates-migration/migrate-worker-http.sh
   ```

Or run all at once:

```bash
# Dry run all migrations
./tools/worker-crates-migration/migrate-all.sh --dry-run

# Execute all migrations
./tools/worker-crates-migration/migrate-all.sh
```

## Common Issues

### Issue: "Source file not found"

**Cause:** File already moved or path incorrect  
**Solution:** Check if file exists at expected location

```bash
ls -la bin/worker-orcd/src/gguf/mod.rs
```

### Issue: "Compilation failed"

**Cause:** Import paths incorrect or missing dependencies  
**Solution:** Check error log and fix imports manually

```bash
# Check error log
cat /tmp/worker-gguf-check.log

# Fix imports manually
vim bin/worker-crates/worker-gguf/src/lib.rs
```

### Issue: "Tests failed"

**Cause:** Test dependencies or test code needs updating  
**Solution:** Review test failures and update tests

```bash
# See detailed test output
cat /tmp/worker-gguf-test.log

# Run tests with verbose output
cargo test -p worker-gguf -- --nocapture
```

### Issue: "Git history not preserved"

**Cause:** Used `cp` instead of `git mv`  
**Solution:** This script uses `git mv` automatically, so this shouldn't happen

```bash
# Verify history
git log --follow bin/worker-crates/worker-gguf/src/lib.rs
```

## Best Practices

### ✅ DO

- Run `--dry-run` first
- Review output carefully
- Test after each migration
- Commit after each successful migration
- Keep migrations small and focused

### ❌ DON'T

- Skip dry-run mode
- Run multiple migrations without testing
- Modify files manually during migration
- Delete backup branch until all migrations complete
- Rush through the process

## Verification Checklist

After each migration:

- [ ] Dry run completed and reviewed
- [ ] Migration executed successfully
- [ ] Extracted crate compiles: `cargo check -p worker-<name>`
- [ ] worker-orcd compiles: `cargo check -p worker-orcd`
- [ ] Extracted crate tests pass: `cargo test -p worker-<name>`
- [ ] worker-orcd tests pass: `cargo test -p worker-orcd`
- [ ] Git history preserved: `git log --follow <path>`
- [ ] Commit message is descriptive
- [ ] Changes reviewed: `git show`

## Example Session

```bash
# Start migration
cd /home/vince/Projects/llama-orch

# Preview
./tools/worker-crates-migration/migrate-worker-gguf.sh --dry-run
# [Review output...]

# Execute
./tools/worker-crates-migration/migrate-worker-gguf.sh
# [Wait for compilation and tests...]

# Verify
cargo check -p worker-gguf
cargo check -p worker-orcd
git log --follow bin/worker-crates/worker-gguf/src/lib.rs
git show

# Success! Continue with next migration
./tools/worker-crates-migration/migrate-worker-tokenizer.sh --dry-run
```

## Getting Help

If you encounter issues:

1. Check the error logs in `/tmp/worker-*-*.log`
2. Review the migration script: `cat tools/worker-crates-migration/migrate-worker-gguf.sh`
3. Check git status: `git status`
4. Review recent commits: `git log --oneline -5`
5. Rollback if needed: `git reset --hard migration-backup`

## References

- **Full README**: `tools/worker-crates-migration/README.md`
- **Development Plan**: `.docs/WORKER_AARMD_DEVELOPMENT_PLAN.md`
- **Scaffold Complete**: `.docs/WORKER_CRATES_SCAFFOLD_COMPLETE.md`
