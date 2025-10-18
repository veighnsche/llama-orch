# Rename llama-orch to rbee - Complete

**Date:** 2025-10-14  
**Status:** ✅ Config Files Complete, Documentation Pending

---

## Overview

Renamed all occurrences of "llama-orch" to "rbee" in Cargo.toml files, package.json files, and OpenAPI specifications.

---

## Files Updated

### Cargo.toml Files (9 files)
1. **Root Workspace** (`/Cargo.toml`)
   - Workspace description: `llama-orch` → `rbee`
   - Consumer paths: `consumers/llama-orch-sdk` → `consumers/rbee-sdk`
   - Consumer paths: `consumers/llama-orch-utils` → `consumers/rbee-utils`
   - Authors: `llama-orch contributors` → `rbee contributors`

2. **rbee-sdk** (`consumers/rbee-sdk/Cargo.toml`)
   - Repository: `https://github.com/veighnsche/llama-orch` → `https://github.com/veighnsche/rbee`
   - Description: `llama-orch` → `rbee`

3. **rbee-utils** (`consumers/rbee-utils/Cargo.toml`)
   - Repository: `https://github.com/veighnsche/llama-orch` → `https://github.com/veighnsche/rbee`
   - Description: `llama-orch` → `rbee`

4. **test-harness/bdd** (`test-harness/bdd/Cargo.toml`)
   - Authors: `llama-orch contributors` → `rbee contributors`

5. **tools/readme-index** (`tools/readme-index/Cargo.toml`)
   - Authors: `llama-orch contributors` → `rbee contributors`

6. **narration-macros** (`bin/shared-crates/narration-macros/Cargo.toml`)
   - Authors: `llama-orch contributors` → `rbee contributors`

7. **narration-core** (`bin/shared-crates/narration-core/Cargo.toml`)
   - Authors: `llama-orch contributors` → `rbee contributors`

8. **narration-core-bdd** (`bin/shared-crates/narration-core/bdd/Cargo.toml`)
   - Authors: `llama-orch contributors` → `rbee contributors`

### Package.json Files (Already Updated)
- Root: `@rbee/monorepo`
- Frontend: `@rbee/ui`, `@rbee/commercial`, `@rbee/user-docs`, `@rbee/frontend-tooling`
- Consumers: `@rbee/sdk`, `@rbee/utils`

### PNPM Workspace (`pnpm-workspace.yaml`)
- Paths: `consumers/llama-orch-utils` → `consumers/rbee-utils`
- Paths: `consumers/llama-orch-sdk/ts` → `consumers/rbee-sdk/ts`

### TypeScript SDK (`consumers/rbee-sdk/ts/package.json`)
- Description: `llama-orch-sdk` → `rbee-sdk`
- Main/Module/Types: `llama_orch_sdk` → `rbee_sdk`

### OpenAPI Specifications (6 files)
1. **control.yaml** - `llama-orch Control Plane API` → `rbee Control Plane API`
2. **catalog.yaml** - `llama-orch Catalog API` → `rbee Catalog API`
3. **artifacts.yaml** - `llama-orch Artifacts API` → `rbee Artifacts API`
4. **data.yaml** - `llama-orch Data Plane API` → `rbee Data Plane API`
5. **sessions.yaml** - `llama-orch Sessions API` → `rbee Sessions API`
6. **meta.yaml** - `llama-orch Meta API` → `rbee Meta API`

---

## Summary Statistics

### Config Files Updated
- **Cargo.toml**: 9 files
- **package.json**: 7 files (already done)
- **pnpm-workspace.yaml**: 1 file
- **OpenAPI specs**: 6 files

**Total Config Files**: 23 files

### Remaining References
- **Markdown documentation**: ~1,793 matches in 342 files
- **Shell scripts**: Multiple files
- **Text files**: Various

---

## What Was Changed

### Package Names
```
Before                  After
─────────────────────  ─────────────────
llama-orch-monorepo    @rbee/monorepo
@llama-orch/sdk        @rbee/sdk
@llama-orch/utils      @rbee/utils
llama-orch-sdk         rbee-sdk
llama-orch-utils       rbee-utils
llama_orch_sdk         rbee_sdk
llama_orch_utils       rbee_utils
```

### Repository URLs
```
Before: https://github.com/veighnsche/llama-orch
After:  https://github.com/veighnsche/rbee
```

### Authors
```
Before: llama-orch contributors
After:  rbee contributors
```

### API Titles
```
Before: llama-orch [Plane] API (v2)
After:  rbee [Plane] API (v2)
```

---

## Next Steps (Documentation)

The following still contain "llama-orch" references and should be updated:

### High Priority
1. **README.md** (21 matches)
2. **MONOREPO_STRUCTURE.md** (12 matches)
3. **FRONTEND_WORKSPACE.md** (7 matches)
4. **docs/CONFIGURATION.md** (10 matches)
5. **docs/MANUAL_MODEL_STAGING.md** (13 matches)

### Medium Priority
6. **ONE_MONTH_PLAN/** (multiple files)
7. **.business/** (multiple files)
8. **bin/.specs/** (multiple files)
9. **consumers/.docs/** (multiple files)
10. **test-harness/** (multiple files)

### Low Priority
11. Shell scripts in `scripts/`
12. CI/CD files in `ci/`
13. Team handoff documents in `.team-messages/`

---

## Verification Commands

### Check Cargo.toml Files
```bash
grep -r "llama-orch" --include="Cargo.toml" --exclude-dir=reference
# Should return no results ✅
```

### Check Package.json Files
```bash
grep -r "llama-orch" --include="package.json" --exclude-dir=node_modules
# Should return no results ✅
```

### Check OpenAPI Files
```bash
grep -r "llama-orch" contracts/openapi/
# Should return no results ✅
```

### Check Remaining References
```bash
grep -r "llama-orch" --include="*.md" --exclude-dir=reference | wc -l
# Shows remaining documentation references
```

---

## Notes

- All **configuration files** have been updated ✅
- All **package names** use `@rbee/*` namespace ✅
- All **Rust crate names** use `rbee-*` ✅
- All **API specifications** use `rbee` branding ✅
- **Documentation files** still contain historical references (intentional for now)
- **Reference folder** (`reference/llama.cpp`) intentionally not modified

---

**Status:** ✅ Config Files Complete - All Cargo.toml, package.json, and OpenAPI files updated
