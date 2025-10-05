# Worker Crates Migration Scripts

**Purpose**: Safe, auditable migration of code from `worker-orcd` to shared `worker-crates/`.

## Safety Features

- ✅ **Dry-run mode** — Preview all changes before execution
- ✅ **Git history preservation** — Uses `git mv` to maintain blame/log
- ✅ **Automatic backups** — Creates backup branch before migration
- ✅ **Verification steps** — Tests compilation after each extraction
- ✅ **Rollback support** — Easy revert if something goes wrong

## Scripts

### 1. `migrate-worker-gguf.sh`
Extract GGUF parser (single file, easiest).

### 2. `migrate-worker-tokenizer.sh`
Extract tokenizer (multiple files).

### 3. `migrate-worker-models.sh`
Extract model adapters (multiple files).

### 4. `migrate-worker-common.sh`
Extract common types (scattered files).

### 5. `migrate-worker-http.sh`
Extract HTTP server (multiple files).

### 6. `migrate-all.sh`
Run all migrations in sequence with verification.

## Usage

### Dry Run (Safe, No Changes)
```bash
# Preview what will happen
./tools/worker-crates-migration/migrate-worker-gguf.sh --dry-run

# See all changes without executing
./tools/worker-crates-migration/migrate-all.sh --dry-run
```

### Execute Migration
```bash
# Run single migration
./tools/worker-crates-migration/migrate-worker-gguf.sh

# Run all migrations
./tools/worker-crates-migration/migrate-all.sh
```

### Rollback
```bash
# If something goes wrong
git checkout migration-backup
git branch -D worker-crates-migration
```

## Migration Order

1. **worker-gguf** — No dependencies, pure Rust (2 hours)
2. **worker-tokenizer** — No dependencies (6 hours)
3. **worker-models** — Depends on worker-gguf (4 hours)
4. **worker-common** — No dependencies (3 hours)
5. **worker-http** — Depends on worker-common (4 hours)

## Verification Checklist

After each migration:
- [ ] Extracted crate compiles: `cargo check -p worker-<name>`
- [ ] worker-orcd compiles: `cargo check -p worker-orcd`
- [ ] Extracted crate tests pass: `cargo test -p worker-<name>`
- [ ] worker-orcd tests pass: `cargo test -p worker-orcd`
- [ ] Git history preserved: `git log --follow <new-path>`
- [ ] Changes committed with descriptive message

## Safety Guarantees

1. **Backup branch created** before any changes
2. **Dry-run shows** all file moves and import changes
3. **Compilation verified** after each step
4. **Tests run** before committing
5. **Git history preserved** via `git mv`
6. **Rollback available** via backup branch

## Example Output (Dry Run)

```
🔍 DRY RUN MODE - No changes will be made
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 Extracting worker-gguf from worker-orcd

Step 1: Create backup branch
  ✓ Would create branch: migration-backup

Step 2: Move files with git mv
  ✓ Would execute: git mv bin/worker-orcd/src/gguf/mod.rs \
                           bin/worker-crates/worker-gguf/src/lib.rs

Step 3: Update imports in worker-orcd
  ✓ Would replace: use crate::gguf:: → use worker_gguf::
  Files affected:
    - bin/worker-orcd/src/main.rs
    - bin/worker-orcd/src/cuda/model.rs

Step 4: Update worker-orcd Cargo.toml
  ✓ Would add: worker-gguf = { path = "../worker-crates/worker-gguf" }

Step 5: Verify compilation
  ✓ Would run: cargo check -p worker-gguf
  ✓ Would run: cargo check -p worker-orcd

Step 6: Run tests
  ✓ Would run: cargo test -p worker-gguf
  ✓ Would run: cargo test -p worker-orcd

Step 7: Commit changes
  ✓ Would commit: "refactor: extract worker-gguf from worker-orcd"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Dry run complete. Run without --dry-run to execute.
```

## References

- **Development Plan**: `.docs/WORKER_AARMD_DEVELOPMENT_PLAN.md`
- **Scaffold Complete**: `.docs/WORKER_CRATES_SCAFFOLD_COMPLETE.md`
