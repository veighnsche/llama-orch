# MD Files Archive Summary

## Script: `archive_md_files.sh`

Successfully archived MD files into `.archive` subfolders throughout the repository.

## Edge Cases Handled

### 1. Protected Files (Never Archived)
- `README.md`
- `LICENSE.md`
- `CONTRIBUTING.md`
- `CODEOWNERS`
- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`

### 2. Pure MD Folders (Skipped)
Folders containing ONLY MD files (no code, no mixed content):
- `docs/`
- `reference/`
- `ONE_MONTH_PLAN/`
- `bin/shared-crates/`
- `bin/llm-worker-rbee/docs/`
- `test-harness/` (parent folder)

### 3. Mixed Content Folders (Archived)
Folders with MD files + other file types get `.archive` subfolder:
- Root directory: 29 MD files archived
- `frontend/packages/rbee-ui/`: 100+ MD files archived
- Page development folders: Multiple MD files per page

### 4. **NEW: Code Project Folders (Archived)**
Folders with MD files + helper files (Cargo.toml, scripts) + code subdirs (src/, tests/):
- ✅ `test-harness/bdd/`: **200 MD files archived**
  - Has: Cargo.toml, scripts, logs
  - Has code subdirs: `src/`, `tests/`
  - Result: `.archive` created with all non-README MD files
  
- ✅ `bin/shared-crates/*/bdd/`: Multiple BDD test crates
  - `secrets-management/bdd/`
  - `narration-core/bdd/`
  - `input-validation/bdd/`
  - `audit-logging/bdd/`

## Statistics

- **Total directories processed**: 34
- **Largest archive**: `test-harness/bdd/.archive/` (200 files)
- **Protected files preserved**: All README.md files remain in place
- **Pure MD folders skipped**: 8+ folders

## Logic

The script determines archival based on:

1. **Skip if**: File is in protected list (README, LICENSE, etc.)
2. **Skip folder if**: Contains ONLY MD files AND no code subdirectories
3. **Archive if**: Folder has MD files mixed with:
   - Non-MD source files, OR
   - Helper files (Cargo.toml, scripts) + code subdirectories (src/, tests/, lib/, bin/)

## Helper Files (Don't Count as "Mixed")

These files don't prevent a folder from being considered "pure MD":
- `Cargo.toml`, `Cargo.lock`
- `package.json`, `pnpm-lock.yaml`
- `.gitignore`, `.editorconfig`
- `*.toml`, `*.lock`, `*.log`
- `*.sh`, `*.py`, `*.js`

**UNLESS** the folder also contains code subdirectories (`src/`, `tests/`, etc.)

## Example: test-harness/bdd

**Before:**
```
test-harness/bdd/
├── Cargo.toml
├── README.md (protected)
├── TEAM_057_SUMMARY.md
├── TEAM_058_SUMMARY.md
├── ... (198 more MD files)
├── extract_features.py
├── test_edge_cases.sh
├── src/ (code directory)
└── tests/ (code directory)
```

**After:**
```
test-harness/bdd/
├── .archive/
│   ├── TEAM_057_SUMMARY.md
│   ├── TEAM_058_SUMMARY.md
│   └── ... (198 more MD files)
├── Cargo.toml
├── README.md (protected, stays in root)
├── extract_features.py
├── test_edge_cases.sh
├── src/
└── tests/
```

## Verification

```bash
# Check test-harness/bdd archive
ls test-harness/bdd/.archive/ | wc -l
# Output: 200

# Verify README stayed in place
ls test-harness/bdd/README.md
# Output: test-harness/bdd/README.md

# Check other BDD crates
ls bin/shared-crates/secrets-management/bdd/.archive/
ls bin/shared-crates/narration-core/bdd/.archive/
```

## Script Location

`/home/vince/Projects/llama-orch/archive_md_files.sh`

Reusable for future archival needs.
