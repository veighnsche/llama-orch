# TEAM-354: Worker UI Migration - Complete Guide

**Status:** 📋 READY FOR MIGRATION  
**Location:** `bin/30_llm_worker_rbee/ui` (EXISTING packages)  
**Estimated Time:** 2-3 hours  
**Priority:** HIGH

---

## ⚠️ CRITICAL: This is a MIGRATION, Not New Implementation

**Existing packages at:** `bin/30_llm_worker_rbee/ui`
- `packages/rbee-worker-react` - React hooks (EXISTS!)
- `packages/rbee-worker-sdk` - WASM SDK (EXISTS!)
- `app` - Worker UI app (EXISTS!)

**DO NOT create new packages!**

---

## Step-by-Step Migration Guide

### Step 1: Add Dependencies (15-20 min)
**File:** `TEAM_354_STEP_1_DEPENDENCY_MIGRATION.md`

**What to do:**
- Add @rbee/sdk-loader to rbee-worker-react
- Add @rbee/narration-client to rbee-worker-react
- Add @rbee/shared-config to rbee-worker-react
- Add @tanstack/react-query to rbee-worker-react
- Add @rbee/dev-utils to app
- Run `pnpm install`

**Result:** All shared packages available

### Step 2: Migrate Hooks (30-45 min)
**File:** `TEAM_354_STEP_2_HOOKS_MIGRATION.md`

**What to do:**
- Migrate hooks to TanStack Query
- Add error handling
- Add QueryClient to App.tsx

**Result:** Automatic state management

### Step 3: Add Narration (30-45 min)
**File:** `TEAM_354_STEP_3_NARRATION_INTEGRATION.md`

**What to do:**
- Create useInferenceWithNarration hook
- Use createStreamHandler from @rbee/narration-client
- Use SERVICES.worker config
- Test narration flow to Keeper

**Result:** Narration events flow to Keeper

### Step 4: Remove Hardcoded URLs (15-20 min)
**File:** `TEAM_354_STEP_4_CONFIG_CLEANUP.md`

**What to do:**
- Find all hardcoded URLs
- Replace with getIframeUrl/getServiceUrl
- Use logStartupMode from @rbee/dev-utils
- Verify no hardcoded URLs remain

**Result:** No hardcoded URLs

### Step 5: Test Everything (30-45 min)
**File:** `TEAM_354_STEP_5_TESTING.md`

**What to do:**
- Build all packages
- Test in development mode
- Test in production mode
- Verify narration flow
- Check TypeScript
- Document results

**Result:** All tests pass

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
✅ TEAM-354 signatures everywhere

---

## Quick Start

```bash
# 1. Read the summary
cat bin/.plan/TEAM_354_MIGRATION_SUMMARY.md

# 2. Start with Step 1
cat bin/.plan/TEAM_354_STEP_1_DEPENDENCY_MIGRATION.md

# 3. Follow each step in order
# Step 1 → Step 2 → Step 3 → Step 4 → Step 5

# 4. Document results
# Create TEAM_354_MIGRATION_COMPLETE.md
```

---

## Reference

**Pattern to follow:**
- TEAM-352 (Queen migration) - Same pattern, different UI
- TEAM-353 (Hive migration) - Same pattern, different UI

**Existing packages:**
- `bin/30_llm_worker_rbee/ui/packages/rbee-worker-react`
- `bin/30_llm_worker_rbee/ui/packages/rbee-worker-sdk`
- `bin/30_llm_worker_rbee/ui/app`

**DO NOT create:**
- `bin/20_llm_worker` (wrong path!)
- `@rbee/worker-react` (wrong name!)
- New packages (they exist!)

---

**Ready to migrate! Follow the step-by-step guides.** 🚀
