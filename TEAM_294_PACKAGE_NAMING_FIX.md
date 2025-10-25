# TEAM-294: Package Naming Consistency Fix

**Status:** ✅ COMPLETE  
**Date:** 2025-01-25

## Problem

3 UI app packages had inconsistent names that didn't follow the monorepo naming convention:

| Package | Old Name | Issue |
|---------|----------|-------|
| Keeper UI | `00_rbee_keeper` | Missing `@rbee/` scope, using folder number |
| Hive UI | `20_rbee_hive` | Missing `@rbee/` scope, using folder number |
| Worker UI | `30_llm_worker_rbee` | Missing `@rbee/` scope, using folder number |

## Solution

Renamed all 3 packages to follow the `@rbee/<binary-name>-ui` convention:

### 1. Keeper UI
**File:** `bin/00_rbee_keeper/ui/package.json`  
**Change:** `"name": "00_rbee_keeper"` → `"name": "@rbee/keeper-ui"`

### 2. Hive UI
**File:** `bin/20_rbee_hive/ui/app/package.json`  
**Change:** `"name": "20_rbee_hive"` → `"name": "@rbee/rbee-hive-ui"`

### 3. Worker UI
**File:** `bin/30_llm_worker_rbee/ui/app/package.json`  
**Change:** `"name": "30_llm_worker_rbee"` → `"name": "@rbee/llm-worker-ui"`

## Naming Convention

All packages now follow these patterns:

| Type | Pattern | Example |
|------|---------|---------|
| **Shared Configs** | `@repo/<name>-config` | `@repo/typescript-config` |
| **Component Library** | `@rbee/ui` | `@rbee/ui` |
| **Binary UI Apps** | `@rbee/<binary-name>-ui` | `@rbee/queen-rbee-ui` |
| **Binary SDKs** | `@rbee/<binary-name>-sdk` | `@rbee/queen-rbee-sdk` |
| **Binary React Hooks** | `@rbee/<binary-name>-react` | `@rbee/queen-rbee-react` |
| **Marketing/Docs** | `@rbee/<app-name>` | `@rbee/commercial` |

## Verification

```bash
# All packages now have consistent names
pnpm list --depth 0 --filter "@rbee/*"
```

**Result:** ✅ All 24 packages follow the naming convention

## Benefits

✅ **Consistency** - All UI apps use `@rbee/<name>-ui` pattern  
✅ **Clarity** - Package names clearly indicate their purpose  
✅ **Scoping** - All packages properly scoped under `@rbee/` or `@repo/`  
✅ **Discoverability** - Easy to find related packages  
✅ **Professional** - Follows npm/pnpm best practices  

## Documentation

Created `PACKAGE_NAMING_REGISTRY.md` with:
- ✅ Quick reference table of all 24 packages
- ✅ Detailed dependency mapping
- ✅ Step-by-step change procedure
- ✅ Quick reference commands
- ✅ Naming convention rules

## Files Changed

### Modified (3 files)
1. `bin/00_rbee_keeper/ui/package.json` - Name updated
2. `bin/20_rbee_hive/ui/app/package.json` - Name updated
3. `bin/30_llm_worker_rbee/ui/app/package.json` - Name updated

### Created (1 file)
4. `PACKAGE_NAMING_REGISTRY.md` - Central registry and documentation

### Updated (1 file)
5. `pnpm-lock.yaml` - Lockfile updated (via `pnpm install`)

## Impact

**No breaking changes** - These are top-level app packages with no dependents.

The packages are:
- ✅ Private packages (`"private": true`)
- ✅ Top-level apps (not dependencies of other packages)
- ✅ No import statements to update

## Summary

**All 24 packages in the monorepo now follow a consistent naming convention! 🎉**

| Status | Count | Packages |
|--------|-------|----------|
| ✅ Consistent | 24 | All packages |
| ⚠️ Needs Fix | 0 | None |
| 🗑️ Deprecated | 1 | `web-ui` (migrated) |

---

**Last Updated:** 2025-01-25 by TEAM-294  
**Status:** ✅ COMPLETE
