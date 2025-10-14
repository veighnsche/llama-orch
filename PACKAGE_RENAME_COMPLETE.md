# Package Rename to @rbee/* - Complete

**Date:** 2025-10-14  
**Status:** ✅ Complete

---

## Overview

All package names have been updated to use the `@rbee/*` namespace, and consumer folders have been renamed from `llama-orch-*` to `rbee-*`.

---

## Changes Made

### 1. Root Package
- **File**: `package.json`
- **Before**: `llama-orch-monorepo`
- **After**: `@rbee/monorepo`

### 2. Frontend Packages

#### rbee-ui (already correct)
- **File**: `frontend/libs/rbee-ui/package.json`
- **Name**: `@rbee/ui` ✅

#### Commercial Frontend (already correct)
- **File**: `frontend/bin/commercial/package.json`
- **Name**: `@rbee/commercial` ✅

#### User Documentation
- **File**: `frontend/bin/user-docs/package.json`
- **Before**: `user-docs`
- **After**: `@rbee/user-docs` ✅

#### Frontend Tooling
- **File**: `frontend/libs/frontend-tooling/package.json`
- **Before**: `rbee-frontend-tooling`
- **After**: `@rbee/frontend-tooling` ✅

### 3. Consumer Packages (Renamed Folders + Packages)

#### rbee-sdk
- **Folder**: `consumers/llama-orch-sdk` → `consumers/rbee-sdk` ✅
- **Package**: `consumers/rbee-sdk/ts/package.json`
  - **Before**: `@llama-orch/sdk`
  - **After**: `@rbee/sdk` ✅
- **Cargo**: `consumers/rbee-sdk/Cargo.toml`
  - **Before**: `name = "llama-orch-sdk"`
  - **After**: `name = "rbee-sdk"` ✅
  - **Lib name**: `llama_orch_sdk` → `rbee_sdk` ✅
- **README**: Updated all references ✅

#### rbee-utils
- **Folder**: `consumers/llama-orch-utils` → `consumers/rbee-utils` ✅
- **Package**: `consumers/rbee-utils/package.json`
  - **Before**: `@llama-orch/utils`
  - **After**: `@rbee/utils` ✅
- **Cargo**: `consumers/rbee-utils/Cargo.toml`
  - **Before**: `name = "llama-orch-utils"`
  - **After**: `name = "rbee-utils"` ✅
  - **Lib name**: `llama_orch_utils` → `rbee_utils` ✅
  - **Dependency**: `llama-orch-sdk` → `rbee-sdk` ✅
- **README**: Updated all references ✅

---

## Complete Package List

All packages now use `@rbee/*` namespace:

1. `@rbee/monorepo` (root)
2. `@rbee/ui` (frontend library)
3. `@rbee/commercial` (commercial frontend)
4. `@rbee/user-docs` (documentation site)
5. `@rbee/frontend-tooling` (shared tooling)
6. `@rbee/sdk` (Rust/WASM SDK)
7. `@rbee/utils` (TypeScript utilities)

---

## Folder Structure

```
llama-orch/
├── package.json (@rbee/monorepo)
├── frontend/
│   ├── bin/
│   │   ├── commercial/ (@rbee/commercial)
│   │   └── user-docs/ (@rbee/user-docs)
│   └── libs/
│       ├── rbee-ui/ (@rbee/ui)
│       └── frontend-tooling/ (@rbee/frontend-tooling)
└── consumers/
    ├── rbee-sdk/ (@rbee/sdk)
    └── rbee-utils/ (@rbee/utils)
```

---

## Import Changes

### Before
```typescript
import { utils } from '@llama-orch/utils'
import { Client } from '@llama-orch/sdk'
```

### After
```typescript
import { utils } from '@rbee/utils'
import { Client } from '@rbee/sdk'
```

### Rust
```toml
# Before
[dependencies]
llama-orch-sdk = "0.0.0"
llama-orch-utils = "0.0.0"

# After
[dependencies]
rbee-sdk = "0.0.0"
rbee-utils = "0.0.0"
```

---

## Updated Files

### Package.json Files (7 total)
1. `/package.json`
2. `/frontend/libs/rbee-ui/package.json` (already correct)
3. `/frontend/bin/commercial/package.json` (already correct)
4. `/frontend/bin/user-docs/package.json`
5. `/frontend/libs/frontend-tooling/package.json`
6. `/consumers/rbee-sdk/ts/package.json`
7. `/consumers/rbee-utils/package.json`

### Cargo.toml Files (2 total)
1. `/consumers/rbee-sdk/Cargo.toml`
2. `/consumers/rbee-utils/Cargo.toml`

### README Files (2 total)
1. `/consumers/rbee-sdk/README.md`
2. `/consumers/rbee-utils/README.md`

### Folders Renamed (2 total)
1. `consumers/llama-orch-sdk` → `consumers/rbee-sdk`
2. `consumers/llama-orch-utils` → `consumers/rbee-utils`

---

## Verification

### Check Package Names
```bash
grep -r '"name":' --include="package.json" --exclude-dir=node_modules
# All should show @rbee/* ✅
```

### Check Cargo Names
```bash
grep "^name = " consumers/*/Cargo.toml
# Should show rbee-sdk and rbee-utils ✅
```

### Check Folder Names
```bash
ls consumers/
# Should show rbee-sdk and rbee-utils ✅
```

---

## Next Steps

1. **Update imports** in any code that references the old names
2. **Rebuild packages** to ensure everything compiles
3. **Update CI/CD** if it references old package names
4. **Update documentation** that mentions old package names

---

## Notes

- All packages are marked as `private: true` (not published to npm)
- Repository URLs updated to GitHub
- Rust crate names follow snake_case convention (`rbee_sdk`, `rbee_utils`)
- Package names follow kebab-case convention (`@rbee/sdk`, `@rbee/utils`)

---

**Status:** ✅ Complete - All packages renamed to @rbee/* namespace
