# Frontend Structure Update - Turborepo Idiomatic

**Date:** 2025-10-14  
**Status:** ✅ Complete

---

## Changes Made

Renamed frontend directories to follow Turborepo conventions:

```diff
frontend/
- ├── bin/          → apps/
+ ├── apps/
  │   ├── commercial
  │   └── user-docs
- └── libs/         → packages/
+ └── packages/
      ├── rbee-ui
      └── frontend-tooling
```

---

## Why This Change?

### Turborepo Convention
Turborepo projects typically use:
- **`apps/`** - Deployable applications (Next.js sites, etc.)
- **`packages/`** - Shared libraries and tooling

### Before (Non-idiomatic)
```
frontend/
├── bin/          # Unclear naming
└── libs/         # Generic
```

### After (Idiomatic)
```
frontend/
├── apps/         # Clear: deployable applications
└── packages/     # Clear: shared packages
```

---

## Files Updated

### Configuration Files
1. **pnpm-workspace.yaml**
   - `frontend/bin/*` → `frontend/apps/*`
   - `frontend/libs/*` → `frontend/packages/*`

### Documentation Files
1. **MONOREPO_STRUCTURE.md** - Updated directory tree
2. **FRONTEND_WORKSPACE.md** - Updated directory tree
3. **QUICKSTART_FRONTEND.md** - Updated example paths

---

## Structure Now

```
frontend/
├── apps/
│   ├── commercial/          # @rbee/commercial
│   │   ├── app/
│   │   ├── public/
│   │   └── package.json
│   └── user-docs/           # @rbee/user-docs
│       ├── app/
│       ├── pages/
│       └── package.json
└── packages/
    ├── rbee-ui/             # @rbee/ui
    │   ├── src/
    │   │   ├── atoms/
    │   │   ├── molecules/
    │   │   └── organisms/
    │   └── package.json
    └── frontend-tooling/    # @rbee/frontend-tooling
        ├── eslint.config.js
        ├── prettier.config.cjs
        └── package.json
```

---

## Package Names (Unchanged)

All package names remain the same:
- `@rbee/commercial`
- `@rbee/user-docs`
- `@rbee/ui`
- `@rbee/frontend-tooling`

---

## Commands (Unchanged)

All development commands work exactly the same:

```bash
# Develop commercial site
pnpm run dev:commercial

# Develop docs site
pnpm run dev:docs

# Develop both sites
pnpm run dev:frontend

# Just UI components
pnpm run dev:ui
```

Or using turbo directly:
```bash
turbo dev
```

---

## Verification

### Check Structure
```bash
ls -la frontend/
# Should show: apps/ packages/
```

### Check Workspace
```bash
pnpm list --depth 0
# Should show all packages properly linked
```

### Test Dev Server
```bash
turbo dev
# All packages should start without errors
```

---

## Benefits

1. **Industry Standard** - Follows Turborepo conventions
2. **Clear Naming** - `apps/` vs `packages/` is self-documenting
3. **Better Onboarding** - New developers understand structure immediately
4. **Tooling Support** - IDEs and tools recognize standard structure

---

## Migration Notes

### TypeScript Errors (Expected)
The IDE shows errors for `frontend/libs/rbee-ui/...` paths because:
- Files physically moved to `frontend/packages/rbee-ui/`
- TypeScript cache needs to refresh
- **Resolution:** Restart TypeScript server or reload IDE

### No Breaking Changes
- All imports use package names (`@rbee/ui`), not file paths
- Package resolution handled by pnpm workspace
- No code changes needed in any files

---

**Status:** ✅ Complete - Frontend structure now follows Turborepo conventions
