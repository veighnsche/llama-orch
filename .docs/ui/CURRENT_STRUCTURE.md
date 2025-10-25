# Current UI Structure (As of 2025-01-25)

**TEAM-293: Actual current state of the project**

## Actual pnpm-workspace.yaml

```yaml
packages:
  - frontend/apps/commercial
  - frontend/apps/user-docs
  - frontend/apps/web-ui # DEPRECATED -> should be migrated to `bin/10_queen_rbee/ui/app`
  - frontend/packages/rbee-ui
  - frontend/packages/rbee-sdk # DEPRECATED -> should be migrated to bin/10_queen_rbee/ui/packages/queen-rbee-sdk
  - frontend/packages/rbee-react # DEPRECATED -> should be migrated to bin/10_queen_rbee/ui/packages/queen-rbee-react
  - frontend/packages/tailwind-config
  - bin/00_rbee_keeper/ui
  - bin/10_queen_rbee/ui/app
  - bin/10_queen_rbee/ui/packages/queen-rbee-sdk
  - bin/10_queen_rbee/ui/packages/queen-rbee-react
  - bin/20_rbee_hive/ui/app
  - bin/20_rbee_hive/ui/packages/rbee-hive-sdk
  - bin/20_rbee_hive/ui/packages/rbee-hive-react
  - bin/30_llm_worker_rbee/ui/app
  - bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk
  - bin/30_llm_worker_rbee/ui/packages/llm-worker-react
```

## Actual Directory Structure

### ✅ Implemented (Co-located in bin/)

```
bin/00_rbee_keeper/ui/
  ├── src/
  ├── index.html
  ├── package.json
  └── vite.config.ts

bin/10_queen_rbee/ui/
  ├── app/                           # NOT YET CREATED
  └── packages/
      ├── queen-rbee-sdk/            # CREATED (tsconfig, src/index.ts)
      └── queen-rbee-react/          # CREATED (tsconfig, src/index.ts)

bin/20_rbee_hive/ui/
  ├── app/                           # NOT YET CREATED
  └── packages/
      ├── rbee-hive-sdk/             # CREATED (tsconfig, src/index.ts)
      └── rbee-hive-react/           # CREATED (tsconfig, src/index.ts)

bin/30_llm_worker_rbee/ui/
  ├── app/                           # NOT YET CREATED
  └── packages/
      ├── llm-worker-sdk/            # CREATED (tsconfig, src/index.ts)
      └── llm-worker-react/          # CREATED (tsconfig, src/index.ts)
```

### ⚠️ Still in frontend/ (DEPRECATED)

```
frontend/apps/web-ui/                # Should migrate to bin/10_queen_rbee/ui/app/
frontend/packages/rbee-sdk/          # Should migrate to bin/10_queen_rbee/ui/packages/queen-rbee-sdk/
frontend/packages/rbee-react/        # Should migrate to bin/10_queen_rbee/ui/packages/queen-rbee-react/
```

### ✅ Staying in frontend/ (Shared)

```
frontend/apps/commercial/            # Marketing site
frontend/apps/user-docs/             # Documentation site
frontend/packages/rbee-ui/           # Shared Storybook components
frontend/packages/tailwind-config/   # Shared Tailwind config
```

## What Exists vs What Doesn't

### ✅ Exists
- `bin/00_rbee_keeper/ui/` (Keeper UI files created)
- `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/` (SDK created)
- `bin/10_queen_rbee/ui/packages/queen-rbee-react/` (React hooks created)
- `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/` (SDK created)
- `bin/20_rbee_hive/ui/packages/rbee-hive-react/` (React hooks created)
- `bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk/` (SDK created)
- `bin/30_llm_worker_rbee/ui/packages/llm-worker-react/` (React hooks created)

### ❌ Doesn't Exist Yet
- `bin/10_queen_rbee/ui/app/` (Queen UI app - still in `frontend/apps/web-ui/`)
- `bin/20_rbee_hive/ui/app/` (Hive UI app - not created yet)
- `bin/30_llm_worker_rbee/ui/app/` (Worker UI app - not created yet)

## Migration Status

### Phase 1: SDK Packages ✅ COMPLETE
- Created all SDK packages in `bin/*/ui/packages/`
- Created TypeScript configs
- Created minimal implementations
- Added to pnpm-workspace.yaml

### Phase 2: UI Apps ⚠️ IN PROGRESS
- ✅ Keeper UI created in `bin/00_rbee_keeper/ui/`
- ❌ Queen UI still in `frontend/apps/web-ui/` (needs migration)
- ❌ Hive UI not created yet
- ❌ Worker UI not created yet

### Phase 3: Deprecation ⚠️ PENDING
- ❌ `frontend/apps/web-ui/` still exists (should be removed after migration)
- ❌ `frontend/packages/rbee-sdk/` still exists (should be removed after migration)
- ❌ `frontend/packages/rbee-react/` still exists (should be removed after migration)

## Current turbo dev Status

**Running:** ✅ All dev servers running successfully

```
✅ bin/00_rbee_keeper/ui              (port 5173)
✅ frontend/apps/commercial           (port 7822)
✅ frontend/apps/user-docs            (port 7811)
✅ frontend/apps/web-ui               (port 5179) - DEPRECATED
✅ frontend/packages/rbee-ui          (port 6006) - Storybook
✅ All SDK packages compiling         (TypeScript watch mode)
```

## Next Steps

1. **Migrate Queen UI**
   ```bash
   mkdir -p bin/10_queen_rbee/ui/app
   mv frontend/apps/web-ui/* bin/10_queen_rbee/ui/app/
   # Update imports to use new SDK packages
   ```

2. **Create Hive UI**
   ```bash
   mkdir -p bin/20_rbee_hive/ui/app
   # Create new Vite + React app
   # Use rbee-hive-sdk and rbee-hive-react
   ```

3. **Create Worker UI**
   ```bash
   mkdir -p bin/30_llm_worker_rbee/ui/app
   # Create new Vite + React app
   # Use llm-worker-sdk and llm-worker-react
   ```

4. **Remove Deprecated**
   ```bash
   rm -rf frontend/apps/web-ui
   rm -rf frontend/packages/rbee-sdk
   rm -rf frontend/packages/rbee-react
   # Update pnpm-workspace.yaml
   ```

## Key Insight: Keeper is Special

**Keeper has no `app/` subfolder:**
- ✅ `bin/00_rbee_keeper/ui/` (direct UI files)
- ❌ NOT `bin/00_rbee_keeper/ui/app/`

**Why:** Keeper doesn't need SDK packages (uses Tauri commands), so no need for the `app/` + `packages/` split.

**Other components need `app/` subfolder:**
- ✅ `bin/10_queen_rbee/ui/app/` + `bin/10_queen_rbee/ui/packages/`
- ✅ `bin/20_rbee_hive/ui/app/` + `bin/20_rbee_hive/ui/packages/`
- ✅ `bin/30_llm_worker_rbee/ui/app/` + `bin/30_llm_worker_rbee/ui/packages/`

---

**Last Updated:** 2025-01-25  
**TEAM-293:** Documentation reflects actual current state
