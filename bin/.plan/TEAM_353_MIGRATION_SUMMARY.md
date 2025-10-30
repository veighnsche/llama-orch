# TEAM-353: Hive UI Migration - Complete Guide

**Status:** 📋 READY FOR MIGRATION  
**Location:** `bin/20_rbee_hive/ui` (EXISTING packages)  
**Estimated Time:** 2-3 hours  
**Priority:** HIGH

---

## ⚠️ CRITICAL: This is a MIGRATION, Not New Implementation

**Existing packages at:** `bin/20_rbee_hive/ui`
- `packages/rbee-hive-react` - React hooks (EXISTS!)
- `packages/rbee-hive-sdk` - WASM SDK (EXISTS!)
- `app` - Hive UI app (EXISTS!)

**DO NOT create new packages!**

---

## Step-by-Step Migration Guide

### Step 1: Add Dependencies (15-20 min)
**File:** `TEAM_353_STEP_1_DEPENDENCY_MIGRATION.md`

**What to do:**
- Add @rbee/sdk-loader to rbee-hive-react
- Add @rbee/narration-client to rbee-hive-react
- Add @rbee/shared-config to rbee-hive-react
- Add @tanstack/react-query to rbee-hive-react
- Add @rbee/dev-utils to app
- Run `pnpm install`

**Result:** All shared packages available

### Step 2: Migrate Hooks (30-45 min)
**File:** `TEAM_353_STEP_2_HOOKS_MIGRATION.md`

**What to do:**
- Replace useModels manual state with TanStack Query
- Replace useWorkers manual state with TanStack Query
- Add error handling
- Add auto-refetch
- Add QueryClient to App.tsx

**Result:** ~40 LOC → ~20 LOC (50% reduction)

### Step 3: Add Narration (30-45 min)
**File:** `TEAM_353_STEP_3_NARRATION_INTEGRATION.md`

**What to do:**
- Create useHiveOperations hook
- Use createStreamHandler from @rbee/narration-client
- Use SERVICES.hive config
- Test narration flow to Keeper

**Result:** Narration events flow to Keeper

### Step 4: Remove Hardcoded URLs (15-20 min)
**File:** `TEAM_353_STEP_4_CONFIG_CLEANUP.md`

**What to do:**
- Find all hardcoded URLs
- Replace with getIframeUrl/getServiceUrl
- Use logStartupMode from @rbee/dev-utils
- Verify no hardcoded URLs remain

**Result:** No hardcoded URLs

### Step 5: Test Everything (30-45 min)
**File:** `TEAM_353_STEP_5_TESTING.md`

**What to do:**
- Build all packages
- Test in development mode
- Test in production mode
- Verify narration flow
- Check TypeScript
- Document results

**Result:** All tests pass

---

## Current vs After Migration

### Current Implementation

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/index.ts`

```typescript
// Manual state management
const [models, setModels] = useState<Model[]>([])
const [loading, setLoading] = useState(true)

useEffect(() => {
  const fetchModels = async () => {
    const data = await listModels()
    setModels(data)
    setLoading(false)
  }
  fetchModels()
}, [])
```

**Problems:**
- ❌ Manual state management
- ❌ No error handling
- ❌ No retry logic
- ❌ Manual polling
- ❌ No caching

### After Migration

```typescript
// TEAM-353: TanStack Query
const { data: models, isLoading, error } = useQuery({
  queryKey: ['hive-models'],
  queryFn: listModels,
  staleTime: 30000,
  retry: 3,
})
```

**Benefits:**
- ✅ Automatic state management
- ✅ Automatic error handling
- ✅ Automatic retry
- ✅ Declarative polling
- ✅ Automatic caching

---

## Code Savings

**useModels + useWorkers:**
- Before: ~40 LOC (manual state)
- After: ~20 LOC (TanStack Query)
- **Savings: ~20 LOC (50% reduction)**

**Additional benefits:**
- Automatic caching
- Automatic retry
- Better error handling
- Easier testing
- Industry-standard library

---

## Shared Packages Used

✅ @rbee/sdk-loader (if needed for SDK loading)  
✅ @rbee/react-hooks (SSE hooks if needed)  
✅ @rbee/narration-client (for narration)  
✅ @rbee/shared-config (for URLs)  
✅ @rbee/dev-utils (for logging)  
✅ @tanstack/react-query (for async state)

---

## Success Criteria

✅ All dependencies added  
✅ Hooks migrated to TanStack Query  
✅ Narration support added  
✅ Hardcoded URLs removed  
✅ All builds pass  
✅ All tests pass  
✅ Narration flows to Keeper  
✅ TEAM-353 signatures everywhere

---

## Quick Start

```bash
# 1. Read the summary
cat bin/.plan/TEAM_353_MIGRATION_SUMMARY.md

# 2. Start with Step 1
cat bin/.plan/TEAM_353_STEP_1_DEPENDENCY_MIGRATION.md

# 3. Follow each step in order
# Step 1 → Step 2 → Step 3 → Step 4 → Step 5

# 4. Document results
# Create TEAM_353_MIGRATION_COMPLETE.md
```

---

## Reference

**Pattern to follow:**
- TEAM-352 (Queen migration) - Same pattern, different UI

**Existing packages:**
- `bin/20_rbee_hive/ui/packages/rbee-hive-react`
- `bin/20_rbee_hive/ui/packages/rbee-hive-sdk`
- `bin/20_rbee_hive/ui/app`

**DO NOT create:**
- `bin/15_rbee_hive` (wrong path!)
- `@rbee/hive-rbee-react` (wrong name!)
- New packages (they exist!)

---

**Ready to migrate! Follow the step-by-step guides.** 🚀
