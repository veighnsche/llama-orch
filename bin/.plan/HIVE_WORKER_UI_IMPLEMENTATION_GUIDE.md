# Hive & Worker UI Migration Guide

**Created:** Oct 30, 2025  
**Updated:** Oct 30, 2025  
**Status:** 📋 READY FOR MIGRATION  
**Total Estimated Time:** 4-6 hours (both UIs)

---

## ⚠️ CRITICAL: UIs Already Exist!

**Hive UI:** `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui`  
**Worker UI:** `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/ui`

**This is a MIGRATION guide, not an implementation guide!**

---

## Overview

This guide provides instructions for **migrating** existing Hive UI and Worker UI to use shared packages, following the pattern established by TEAM-352 (Queen migration).

**Key Principle:** Migrate to ALL shared packages. Remove custom implementations.

---

## Prerequisites

**Must be complete before starting:**

✅ TEAM-356 (Shared packages extraction)
- @rbee/sdk-loader (34 tests passing)
- @rbee/react-hooks (19 tests passing)
- @rbee/shared-config
- @rbee/narration-client
- @rbee/dev-utils

✅ TEAM-352 (Queen migration validates pattern)
- Proves shared packages work
- Establishes best practices
- Documents pitfalls to avoid

---

## TEAM-353: Hive UI Migration

**Total Time:** 2-3 hours  
**Priority:** HIGH  
**Location:** `bin/20_rbee_hive/ui`

### Migration Guide

**File:** `TEAM_353_HIVE_MIGRATION_GUIDE.md`

**Steps:**
1. Add shared package dependencies
2. Migrate useModels/useWorkers to TanStack Query
3. Add narration support (if needed)
4. Update App to use QueryClient
5. Remove hardcoded URLs
6. Test and verify

**Current State:**
- ❌ Manual async state management
- ❌ No error handling
- ❌ No retry logic
- ❌ Hardcoded polling intervals

**After Migration:**
- ✅ TanStack Query for state
- ✅ Automatic error handling
- ✅ Automatic retry
- ✅ Declarative polling

---

## TEAM-354: Worker UI Migration

**Total Time:** 2-3 hours  
**Priority:** HIGH  
**Location:** `bin/30_llm_worker_rbee/ui`

### Migration Guide

**File:** `TEAM_354_WORKER_MIGRATION_GUIDE.md`

**Steps:**
1. Add shared package dependencies
2. Migrate hooks to TanStack Query
3. Add narration support for inference
4. Update App to use QueryClient
5. Remove hardcoded URLs
6. Test and verify

**After Migration:**
- ✅ TanStack Query for state
- ✅ Narration support
- ✅ Shared config for URLs
- ✅ Consistent with Queen/Hive

---

## Shared Packages Pattern

### ✅ ALWAYS Use These Packages

**SDK Loading:**
```typescript
import { createSDKLoader } from '@rbee/sdk-loader'

const loader = createSDKLoader<MySDK>({
  packageName: '@rbee/my-sdk',
  requiredExports: ['MyClient'],
  timeout: 15000,
  maxAttempts: 3,
})
```

**Narration:**
```typescript
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

const handler = createStreamHandler(SERVICES.hive, (event) => {
  console.log('Narration:', event)
}, {
  debug: true,
  validate: true,
})
```

**Configuration:**
```typescript
import { getIframeUrl, getAllowedOrigins } from '@rbee/shared-config'

const url = getIframeUrl('hive', isDev)
const origins = getAllowedOrigins()
```

**Logging:**
```typescript
import { logStartupMode } from '@rbee/dev-utils'

logStartupMode("HIVE UI", import.meta.env.DEV, 7836)
```

**Async State:**
```typescript
import { useQuery } from '@tanstack/react-query'

const { data, isLoading, error } = useQuery({
  queryKey: ['workers'],
  queryFn: fetchWorkers,
  staleTime: 5000,
})
```

---

## ❌ NEVER Do These Things

### 1. Custom SDK Loader
```typescript
// ❌ WRONG - Don't create custom loader
export function loadSDK() {
  // Custom retry logic
  // Custom timeout logic
  // Custom singleflight
}

// ✅ RIGHT - Use shared loader
import { createSDKLoader } from '@rbee/sdk-loader'
```

### 2. Custom Narration
```typescript
// ❌ WRONG - Don't parse narration manually
export function parseNarrationLine(line: string) {
  // Custom JSON parsing
  // Custom [DONE] handling
}

// ✅ RIGHT - Use shared narration client
import { createStreamHandler } from '@rbee/narration-client'
```

### 3. Hardcoded URLs
```typescript
// ❌ WRONG - Don't hardcode URLs
const url = "http://localhost:7835"

// ✅ RIGHT - Use shared config
import { getIframeUrl } from '@rbee/shared-config'
const url = getIframeUrl('hive', isDev)
```

### 4. Manual Async State
```typescript
// ❌ WRONG - Don't manage state manually
const [data, setData] = useState([])
const [loading, setLoading] = useState(false)
useEffect(() => {
  setLoading(true)
  fetch().then(setData).finally(() => setLoading(false))
}, [])

// ✅ RIGHT - Use TanStack Query
const { data, isLoading } = useQuery({
  queryKey: ['data'],
  queryFn: fetch,
})
```

### 5. Re-export Wrappers (RULE ZERO)
```typescript
// ❌ WRONG - Don't create wrapper exports
export { createStreamHandler as createNarrationHandler } from '@rbee/narration-client'

// ✅ RIGHT - Import directly
import { createStreamHandler } from '@rbee/narration-client'
```

---

## Code Savings Summary

### Queen UI (TEAM-352 - Migration)
- SDK Loader: 150 LOC saved
- Hooks: 55 LOC saved
- Narration: 97 LOC saved
- Config: 13 LOC saved
- **Total: 315 LOC saved (70%)**

### Hive UI (TEAM-353 - New Implementation)
- SDK Loader: 130 LOC saved
- Narration: 79 LOC saved
- Async State: 50 LOC saved
- **Total: 259 LOC saved (70%)**

### Worker UI (TEAM-354 - New Implementation)
- SDK Loader: 130 LOC saved
- Narration: 79 LOC saved
- **Total: 209 LOC saved (65%)**

### Combined Total
- **783 LOC saved across 3 UIs**
- **Average: 68% code reduction**

---

## Success Criteria

**For each UI, verify:**

✅ No custom SDK loader code  
✅ No custom narration code  
✅ No hardcoded URLs  
✅ No manual async state management  
✅ No re-export wrappers  
✅ All shared packages used  
✅ TypeScript builds pass  
✅ Narration flows to Keeper  
✅ All tests pass  
✅ TEAM signatures added

---

## Testing Strategy

### Development Mode
1. Start backend
2. Start UI dev server
3. Start Keeper UI
4. Test all features
5. Verify narration flow

### Production Mode
1. Build UI
2. Start backend (release mode)
3. Test all features
4. Verify narration flow

### Critical Tests
- SDK loads successfully
- Narration appears in Keeper
- No console errors
- Hot reload works (dev mode)
- Performance acceptable

---

## Common Pitfalls

### 1. Message Type Mismatch
**Problem:** Narration doesn't appear in Keeper  
**Cause:** Wrong message type  
**Fix:** Use `"NARRATION_EVENT"` (from @rbee/narration-client)

### 2. Origin Mismatch
**Problem:** postMessage blocked  
**Cause:** Keeper doesn't allow origin  
**Fix:** Use `getAllowedOrigins()` from @rbee/shared-config

### 3. SDK Not Loading
**Problem:** SDK fails to load  
**Cause:** WASM file not built  
**Fix:** Build SDK package first

### 4. Port Conflicts
**Problem:** Dev server won't start  
**Cause:** Port already in use  
**Fix:** Check @rbee/shared-config for correct ports

---

## Documentation References

**Shared Packages:**
- `frontend/packages/sdk-loader/README.md`
- `frontend/packages/react-hooks/README.md`
- `frontend/packages/narration-client/README.md`
- `frontend/packages/shared-config/README.md`

**Migration Examples:**
- `bin/.plan/TEAM_352_QUEEN_MIGRATION_PHASE.md`
- `bin/.plan/TEAM_352_STEP_1_SDK_LOADER_MIGRATION.md`
- `bin/.plan/TEAM_352_STEP_3_NARRATION_MIGRATION.md`

**Extraction Process:**
- `bin/.plan/TEAM_356_EXTRACTION_EXTRAVAGANZA.md`

---

## Quick Start

### For Hive UI:
```bash
# Read the guides
cat bin/.plan/TEAM_353_SUMMARY.md
cat bin/.plan/TEAM_353_STEP_1_PROJECT_SETUP.md

# Start implementation
cd bin/15_rbee_hive
# Follow Step 1 instructions...
```

### For Worker UI:
```bash
# Read the guides
cat bin/.plan/TEAM_354_SUMMARY.md
cat bin/.plan/TEAM_354_STEP_1_PROJECT_SETUP.md

# Start implementation
cd bin/20_llm_worker
# Follow Step 1 instructions...
```

---

## Support

**Questions?** Check these resources:
1. Step-by-step guides (this directory)
2. Queen UI implementation (reference example)
3. Shared package READMEs
4. TEAM-352 migration docs

**Found a bug in shared packages?**
1. Fix it in the shared package
2. All UIs benefit automatically
3. Document the fix

---

**Ready to implement! Follow the step-by-step guides.** 🚀
