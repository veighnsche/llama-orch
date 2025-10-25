# Package Naming Registry

**Last Updated:** 2025-01-25 by TEAM-294  
**Purpose:** Central registry of all package names and their dependencies to maintain consistency across the monorepo.

---

## 📋 Quick Reference Table - All Package Names

**Edit this table to plan name changes, then follow the detailed sections below for implementation.**

| Category | Current Name | Location | Status | Recommended Name | Notes |
|----------|--------------|----------|--------|------------------|-------|
| **Root** | `@rbee/monorepo` | `/package.json` | ✅ OK | - | - |
| **Shared Configs** |
| | `@repo/typescript-config` | `frontend/packages/typescript-config/` | ✅ OK | - | - |
| | `@repo/eslint-config` | `frontend/packages/eslint-config/` | ✅ OK | - | - |
| | `@repo/vite-config` | `frontend/packages/vite-config/` | ✅ OK | - | Not yet used |
| | `@repo/tailwind-config` | `frontend/packages/tailwind-config/` | ✅ OK | - | - |
| **Component Library** |
| | `@rbee/ui` | `frontend/packages/rbee-ui/` | ✅ OK | - | - |
| **Binary UI Apps** |
| | `@rbee/keeper-ui` | `bin/00_rbee_keeper/ui/` | ✅ OK | - | Tauri GUI |
| | `@rbee/queen-rbee-ui` | `bin/10_queen_rbee/ui/app/` | ✅ OK | - | - |
| | `@rbee/rbee-hive-ui` | `bin/20_rbee_hive/ui/app/` | ✅ OK | - | - |
| | `@rbee/llm-worker-ui` | `bin/30_llm_worker_rbee/ui/app/` | ✅ OK | - | - |
| **Queen SDKs** |
| | `@rbee/queen-rbee-sdk` | `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/` | ✅ OK | - | WASM SDK |
| | `@rbee/queen-rbee-react` | `bin/10_queen_rbee/ui/packages/queen-rbee-react/` | ✅ OK | - | React hooks |
| **Hive SDKs** |
| | `@rbee/rbee-hive-sdk` | `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/` | ✅ OK | - | WASM SDK |
| | `@rbee/rbee-hive-react` | `bin/20_rbee_hive/ui/packages/rbee-hive-react/` | ✅ OK | - | React hooks |
| **Worker SDKs** |
| | `@rbee/llm-worker-sdk` | `bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk/` | ✅ OK | - | WASM SDK |
| | `@rbee/llm-worker-react` | `bin/30_llm_worker_rbee/ui/packages/llm-worker-react/` | ✅ OK | - | React hooks |
| **Marketing/Docs Apps** |
| | `@rbee/commercial` | `frontend/apps/commercial/` | ✅ OK | - | Next.js |
| | `@rbee/user-docs` | `frontend/apps/user-docs/` | ✅ OK | - | Next.js |
| | `web-ui` | `frontend/apps/web-ui/` | 🗑️ DEPRECATED | - | Migrated to queen-rbee-ui |

**Legend:**
- ✅ OK = Name follows convention
- ⚠️ FIX = Needs renaming for consistency
- 🗑️ DEPRECATED = No longer used

---

## ⚠️ CRITICAL: Changing Package Names

**Before changing any package name, you MUST:**
1. Check this document for all dependents
2. Update ALL files listed in the "Dependents" section
3. Update `pnpm-workspace.yaml` if it's a workspace package
4. Run `pnpm install` to update lockfile
5. Update this document with the new name

---

## 📦 Package Registry

### Root Package

| Package Name | Location | Type | Dependents |
|--------------|----------|------|------------|
| `@rbee/monorepo` | `/package.json` | Root | N/A (root package) |

---

## 🎨 Shared Configuration Packages

### @repo/typescript-config

**Location:** `frontend/packages/typescript-config/package.json`  
**Type:** Shared Config  
**Exports:** `base.json`, `react-app.json`, `vite.json`

**Dependents:**
- `bin/00_rbee_keeper/ui/package.json`
- `bin/10_queen_rbee/ui/app/package.json`
- ⚠️ **TODO:** `bin/20_rbee_hive/ui/app/package.json` (needs migration)
- ⚠️ **TODO:** `bin/30_llm_worker_rbee/ui/app/package.json` (needs migration)

**Files to Update if Name Changes:**
- All `tsconfig.app.json` files that extend `@repo/typescript-config/react-app.json`
- All `tsconfig.node.json` files that extend `@repo/typescript-config/vite.json`
- `pnpm-workspace.yaml`

---

### @repo/eslint-config

**Location:** `frontend/packages/eslint-config/package.json`  
**Type:** Shared Config  
**Exports:** `react.js`

**Dependents:**
- `bin/00_rbee_keeper/ui/package.json`
- `bin/10_queen_rbee/ui/app/package.json`
- ⚠️ **TODO:** `bin/20_rbee_hive/ui/app/package.json` (needs migration)
- ⚠️ **TODO:** `bin/30_llm_worker_rbee/ui/app/package.json` (needs migration)

**Files to Update if Name Changes:**
- All `eslint.config.js` files that import from `@repo/eslint-config/react.js`
- `pnpm-workspace.yaml`

---

### @repo/vite-config

**Location:** `frontend/packages/vite-config/package.json`  
**Type:** Shared Config  
**Exports:** `index.js`, `index.d.ts`

**Dependents:**
- ⚠️ **NONE** (created but not yet used - kept for future use)

**Files to Update if Name Changes:**
- Any `vite.config.ts` files that import from `@repo/vite-config`
- `pnpm-workspace.yaml`

---

### @repo/tailwind-config

**Location:** `frontend/packages/tailwind-config/package.json`  
**Type:** Shared Config

**Dependents:**
- `frontend/packages/rbee-ui/package.json`
- `frontend/apps/commercial/package.json`

**Files to Update if Name Changes:**
- `pnpm-workspace.yaml`

---

## 🎨 UI Component Library

### @rbee/ui

**Location:** `frontend/packages/rbee-ui/package.json`  
**Type:** Component Library  
**Exports:** Atoms, Molecules, Organisms, Templates, Styles

**Dependents:**
- `bin/00_rbee_keeper/ui/package.json`
- `bin/10_queen_rbee/ui/app/package.json`
- `frontend/apps/commercial/package.json`
- `frontend/apps/user-docs/package.json`
- `frontend/apps/web-ui/package.json` (deprecated)

**Files to Update if Name Changes:**
- All app `package.json` files listed above
- Any source files importing from `@rbee/ui/*`
- `pnpm-workspace.yaml`

---

## 🐝 Binary-Specific UI Apps

### @rbee/keeper-ui

**Location:** `bin/00_rbee_keeper/ui/package.json`  
**Type:** Tauri GUI App  
**Current Name:** `@rbee/keeper-ui`  
**✅ CONSISTENT**

**Dependencies:**
- `@rbee/ui`
- `@repo/typescript-config`
- `@repo/eslint-config`
- `@tauri-apps/api`

**Dependents:** NONE (top-level app)

**Files to Update if Name Changes:**
- `pnpm-workspace.yaml`
- Any scripts/docs referencing this package

---

### @rbee/queen-rbee-ui

**Location:** `bin/10_queen_rbee/ui/app/package.json`  
**Type:** Web App  
**Current Name:** `@rbee/queen-rbee-ui`  
**✅ CONSISTENT**

**Dependencies:**
- `@rbee/queen-rbee-react`
- `@rbee/queen-rbee-sdk`
- `@rbee/ui`
- `@repo/typescript-config`
- `@repo/eslint-config`

**Dependents:** NONE (top-level app)

**Files to Update if Name Changes:**
- `pnpm-workspace.yaml`
- Any scripts/docs referencing this package

---

### @rbee/rbee-hive-ui

**Location:** `bin/20_rbee_hive/ui/app/package.json`  
**Type:** Web App  
**Current Name:** `@rbee/rbee-hive-ui`  
**✅ CONSISTENT**

**Dependencies:**
- ⚠️ **TODO:** Needs migration to use shared configs

**Dependents:** NONE (top-level app)

**Files to Update if Name Changes:**
- `pnpm-workspace.yaml`
- Any scripts/docs referencing this package

---

### @rbee/llm-worker-ui

**Location:** `bin/30_llm_worker_rbee/ui/app/package.json`  
**Type:** Web App  
**Current Name:** `@rbee/llm-worker-ui`  
**✅ CONSISTENT**

**Dependencies:**
- ⚠️ **TODO:** Needs migration to use shared configs

**Dependents:** NONE (top-level app)

**Files to Update if Name Changes:**
- `pnpm-workspace.yaml`
- Any scripts/docs referencing this package

---

## 📚 Binary-Specific SDK Packages

### @rbee/queen-rbee-sdk

**Location:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/package.json`  
**Type:** WASM SDK  
**Current Name:** `@rbee/queen-rbee-sdk`  
**✅ CONSISTENT**

**Dependencies:** NONE (base SDK)

**Dependents:**
- `bin/10_queen_rbee/ui/packages/queen-rbee-react/package.json`
- `bin/10_queen_rbee/ui/app/package.json`

**Files to Update if Name Changes:**
- `bin/10_queen_rbee/ui/packages/queen-rbee-react/package.json` (dependencies)
- `bin/10_queen_rbee/ui/app/package.json` (dependencies)
- `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/loader.ts` (import statement)
- `pnpm-workspace.yaml`

---

### @rbee/queen-rbee-react

**Location:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/package.json`  
**Type:** React Hooks  
**Current Name:** `@rbee/queen-rbee-react`  
**✅ CONSISTENT**

**Dependencies:**
- `@rbee/queen-rbee-sdk`

**Dependents:**
- `bin/10_queen_rbee/ui/app/package.json`

**Files to Update if Name Changes:**
- `bin/10_queen_rbee/ui/app/package.json` (dependencies)
- `bin/10_queen_rbee/ui/app/src/hooks/useHeartbeat.ts` (import statement)
- `bin/10_queen_rbee/ui/app/src/stores/rbeeStore.ts` (import statement)
- `pnpm-workspace.yaml`

---

### @rbee/rbee-hive-sdk

**Location:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/package.json`  
**Type:** WASM SDK  
**Current Name:** `@rbee/rbee-hive-sdk`  
**✅ CONSISTENT**

**Dependencies:** NONE (base SDK)

**Dependents:**
- `bin/20_rbee_hive/ui/packages/rbee-hive-react/package.json`

**Files to Update if Name Changes:**
- `bin/20_rbee_hive/ui/packages/rbee-hive-react/package.json` (dependencies)
- Any source files in rbee-hive-react that import from this SDK
- `pnpm-workspace.yaml`

---

### @rbee/rbee-hive-react

**Location:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/package.json`  
**Type:** React Hooks  
**Current Name:** `@rbee/rbee-hive-react`  
**✅ CONSISTENT**

**Dependencies:**
- `@rbee/rbee-hive-sdk`

**Dependents:**
- ⚠️ **TODO:** `bin/20_rbee_hive/ui/app/package.json` (when migrated)

**Files to Update if Name Changes:**
- `bin/20_rbee_hive/ui/app/package.json` (dependencies, when migrated)
- Any source files in hive app that import from this package
- `pnpm-workspace.yaml`

---

### @rbee/llm-worker-sdk

**Location:** `bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk/package.json`  
**Type:** WASM SDK  
**Current Name:** `@rbee/llm-worker-sdk`  
**✅ CONSISTENT**

**Dependencies:** NONE (base SDK)

**Dependents:**
- `bin/30_llm_worker_rbee/ui/packages/llm-worker-react/package.json`

**Files to Update if Name Changes:**
- `bin/30_llm_worker_rbee/ui/packages/llm-worker-react/package.json` (dependencies)
- Any source files in llm-worker-react that import from this SDK
- `pnpm-workspace.yaml`

---

### @rbee/llm-worker-react

**Location:** `bin/30_llm_worker_rbee/ui/packages/llm-worker-react/package.json`  
**Type:** React Hooks  
**Current Name:** `@rbee/llm-worker-react`  
**✅ CONSISTENT**

**Dependencies:**
- `@rbee/llm-worker-sdk`

**Dependents:**
- ⚠️ **TODO:** `bin/30_llm_worker_rbee/ui/app/package.json` (when migrated)

**Files to Update if Name Changes:**
- `bin/30_llm_worker_rbee/ui/app/package.json` (dependencies, when migrated)
- Any source files in worker app that import from this package
- `pnpm-workspace.yaml`

---

## 🌐 Frontend Apps (Marketing/Docs)

### @rbee/commercial

**Location:** `frontend/apps/commercial/package.json`  
**Type:** Next.js App (Marketing)  
**Current Name:** `@rbee/commercial`  
**✅ CONSISTENT**

**Dependencies:**
- `@rbee/ui`
- `@repo/tailwind-config`

**Dependents:** NONE (top-level app)

---

### @rbee/user-docs

**Location:** `frontend/apps/user-docs/package.json`  
**Type:** Next.js App (Documentation)  
**Current Name:** `@rbee/user-docs`  
**✅ CONSISTENT**

**Dependencies:**
- `@rbee/ui`

**Dependents:** NONE (top-level app)

---

### web-ui (DEPRECATED)

**Location:** `frontend/apps/web-ui/package.json`  
**Type:** Web App (DEPRECATED)  
**Current Name:** `web-ui`  
**⚠️ DEPRECATED:** Migrated to `@rbee/queen-rbee-ui`

**Status:** Commented out in `pnpm-workspace.yaml`

---

## 📋 Naming Convention Rules

### Package Name Patterns

| Type | Pattern | Example |
|------|---------|---------|
| **Shared Configs** | `@repo/<name>-config` | `@repo/typescript-config` |
| **Component Library** | `@rbee/ui` | `@rbee/ui` |
| **Binary UI Apps** | `@rbee/<binary-name>-ui` | `@rbee/queen-rbee-ui` |
| **Binary SDKs** | `@rbee/<binary-name>-sdk` | `@rbee/queen-rbee-sdk` |
| **Binary React Hooks** | `@rbee/<binary-name>-react` | `@rbee/queen-rbee-react` |
| **Marketing/Docs Apps** | `@rbee/<app-name>` | `@rbee/commercial` |

### Binary Name Mapping

| Binary | Folder | UI App Name | SDK Name | React Hooks Name |
|--------|--------|-------------|----------|------------------|
| **Keeper** | `00_rbee_keeper` | `@rbee/keeper-ui` ✅ | N/A (Tauri) | N/A (Tauri) |
| **Queen** | `10_queen_rbee` | `@rbee/queen-rbee-ui` ✅ | `@rbee/queen-rbee-sdk` ✅ | `@rbee/queen-rbee-react` ✅ |
| **Hive** | `20_rbee_hive` | `@rbee/rbee-hive-ui` ✅ | `@rbee/rbee-hive-sdk` ✅ | `@rbee/rbee-hive-react` ✅ |
| **LLM Worker** | `30_llm_worker_rbee` | `@rbee/llm-worker-ui` ✅ | `@rbee/llm-worker-sdk` ✅ | `@rbee/llm-worker-react` ✅ |

---

## ✅ All Packages Now Consistent

**TEAM-294 Fixed (2025-01-25):**

1. ✅ **`00_rbee_keeper` → `@rbee/keeper-ui`**
   - File: `bin/00_rbee_keeper/ui/package.json`
   - Status: COMPLETE

2. ✅ **`20_rbee_hive` → `@rbee/rbee-hive-ui`**
   - File: `bin/20_rbee_hive/ui/app/package.json`
   - Status: COMPLETE

3. ✅ **`30_llm_worker_rbee` → `@rbee/llm-worker-ui`**
   - File: `bin/30_llm_worker_rbee/ui/app/package.json`
   - Status: COMPLETE

### Medium Priority

4. **Migrate hive and worker UIs to use shared configs**
   - Add `@repo/typescript-config` dependency
   - Add `@repo/eslint-config` dependency
   - Update `tsconfig.app.json` to extend shared config
   - Update `tsconfig.node.json` to extend shared config
   - Update `eslint.config.js` to use shared config

---

## 🔄 Change Procedure

### Step-by-Step Guide

When changing a package name, follow these steps **IN ORDER**:

#### 1. Identify All Dependents
```bash
# Search for the old package name in all package.json files
grep -r "old-package-name" --include="package.json" --exclude-dir=node_modules
```

#### 2. Update the Package Itself
- Edit the package's `package.json`
- Change the `"name"` field

#### 3. Update All Dependents
For each dependent found in step 1:
- Update `package.json` dependencies
- Update any import statements in source files

#### 4. Update Workspace Configuration
- Edit `pnpm-workspace.yaml`
- Update the package path if needed

#### 5. Update This Document
- Update the package name in the registry
- Update all cross-references

#### 6. Install and Verify
```bash
# Reinstall dependencies
pnpm install

# Verify no broken references
pnpm list | grep "UNMET"

# Build all packages
turbo build
```

---

## 📝 Quick Reference Commands

### Find All Package Names
```bash
find . -name "package.json" -not -path "*/node_modules/*" -not -path "*/.next/*" -not -path "*/dist/*" -exec grep -H '"name":' {} \;
```

### Find All Dependencies on a Package
```bash
grep -r "@rbee/package-name" --include="package.json" --exclude-dir=node_modules
```

### Find All Import Statements
```bash
grep -r "from '@rbee/package-name'" --include="*.ts" --include="*.tsx" --exclude-dir=node_modules
```

### Verify Workspace Packages
```bash
pnpm list --depth 0 --filter "@rbee/*"
```

---

## ✅ Naming Alignment Complete

**All packages now follow the consistent naming convention:**

```bash
# ✅ 1. Keeper UI - COMPLETE
bin/00_rbee_keeper/ui/package.json
  "name": "@rbee/keeper-ui"

# ✅ 2. Hive UI - COMPLETE
bin/20_rbee_hive/ui/app/package.json
  "name": "@rbee/rbee-hive-ui"

# ✅ 3. Worker UI - COMPLETE
bin/30_llm_worker_rbee/ui/app/package.json
  "name": "@rbee/llm-worker-ui"
```

**All 24 packages now follow the consistent naming convention! 🎉**

---

**Maintained by:** TEAM-294  
**Review Frequency:** After any package name change  
**Last Audit:** 2025-01-25
