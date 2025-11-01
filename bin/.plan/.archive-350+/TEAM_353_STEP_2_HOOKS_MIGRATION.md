# TEAM-353 Step 2: Hive UI - Migrate Hooks to TanStack Query

**Estimated Time:** 30-45 minutes  
**Priority:** HIGH  
**Previous Step:** TEAM_353_STEP_1_DEPENDENCY_MIGRATION.md  
**Next Step:** TEAM_353_STEP_3_NARRATION_INTEGRATION.md

---

## Mission

Migrate existing useModels and useWorkers hooks from manual state management to TanStack Query.

**Location:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/index.ts`

---

## Deliverables Checklist

- [ ] useModels migrated to TanStack Query
- [ ] useWorkers migrated to TanStack Query
- [ ] Error handling added
- [ ] Auto-refetch configured
- [ ] Package builds successfully
- [ ] TEAM-353 signatures added

---

## Current Implementation

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/index.ts`

```typescript
// CURRENT - Manual state management
import { listModels, listWorkers, type Model, type Worker } from '@rbee/rbee-hive-sdk'
import { useEffect, useState } from 'react'

export function useModels() {
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

  return { models, loading }
}

export function useWorkers() {
  const [workers, setWorkers] = useState<Worker[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchWorkers = async () => {
      const data = await listWorkers()
      setWorkers(data)
      setLoading(false)
    }

    fetchWorkers()
    const interval = setInterval(fetchWorkers, 2000)
    return () => clearInterval(interval)
  }, [])

  return { workers, loading }
}
```

**Problems:**
- ❌ Manual state management (useState, useEffect)
- ❌ No error handling
- ❌ No retry logic
- ❌ Manual polling with setInterval
- ❌ No caching
- ❌ No stale data handling

---

## New Implementation

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/index.ts`

```typescript
// TEAM-353: Migrated to use TanStack Query (no manual state management)
import { useQuery } from '@tanstack/react-query'
import { listModels, listWorkers, type Model, type Worker } from '@rbee/rbee-hive-sdk'

/**
 * Hook for fetching model list from Hive
 * 
 * TEAM-353: Migrated to TanStack Query
 * - Automatic caching
 * - Automatic error handling
 * - Automatic retry
 * - Stale data management
 */
export function useModels() {
  const { 
    data: models, 
    isLoading: loading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['hive-models'],
    queryFn: listModels,
    staleTime: 30000, // Models change less frequently (30 seconds)
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })

  return { 
    models: models || [], 
    loading,
    error: error as Error | null,
    refetch
  }
}

/**
 * Hook for fetching worker list from Hive
 * 
 * TEAM-353: Migrated to TanStack Query
 * - Automatic polling (refetchInterval)
 * - Automatic caching
 * - Automatic error handling
 * - Automatic retry
 */
export function useWorkers() {
  const { 
    data: workers, 
    isLoading: loading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['hive-workers'],
    queryFn: listWorkers,
    staleTime: 5000, // Workers change frequently (5 seconds)
    refetchInterval: 2000, // Auto-refetch every 2 seconds
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })

  return { 
    workers: workers || [], 
    loading,
    error: error as Error | null,
    refetch
  }
}

// Re-export types
export type { Model, Worker } from '@rbee/rbee-hive-sdk'
```

**Benefits:**
- ✅ Automatic caching (no redundant fetches)
- ✅ Automatic error handling
- ✅ Automatic retry with exponential backoff
- ✅ Declarative polling (no manual setInterval)
- ✅ Stale data management
- ✅ Manual refetch capability
- ✅ ~40 LOC → ~20 LOC (50% reduction)

---

## Step 3: Build Package

```bash
cd bin/20_rbee_hive/ui/packages/rbee-hive-react
pnpm build
```

**Expected output:**
```
> @rbee/rbee-hive-react@0.1.0 build
> tsc

✓ Built successfully
```

---

## Step 4: Update App to Use QueryClient

**File:** `bin/20_rbee_hive/ui/app/src/App.tsx`

**Add QueryClient setup:**

```typescript
// TEAM-353: Hive UI - Worker & Model Management
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { logStartupMode } from '@rbee/dev-utils'

// TEAM-353: Use shared startup logging
logStartupMode("HIVE UI", import.meta.env.DEV, 7836)

// TEAM-353: Create QueryClient for TanStack Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      {/* Your existing app content */}
    </QueryClientProvider>
  )
}

export default App
```

---

## Step 5: Build App

```bash
cd bin/20_rbee_hive/ui/app
pnpm build
```

---

## Testing Checklist

- [ ] `pnpm build` (rbee-hive-react) - success
- [ ] `pnpm build` (app) - success
- [ ] useModels returns models array
- [ ] useWorkers returns workers array
- [ ] Auto-refetch works (workers update every 2s)
- [ ] Error handling works
- [ ] No TypeScript errors
- [ ] TEAM-353 signatures added

---

## Code Comparison

**Before (Manual State):**
- useState: 2 lines
- useEffect: 10 lines
- setInterval: 2 lines
- Cleanup: 1 line
- **Total: ~15 LOC per hook**

**After (TanStack Query):**
- useQuery: 8 lines
- **Total: ~8 LOC per hook**
- **Savings: ~7 LOC per hook (47% reduction)**

**Additional benefits:**
- Automatic caching
- Automatic retry
- Automatic error handling
- Better TypeScript support
- Easier testing

---

## Success Criteria

✅ useModels migrated to TanStack Query  
✅ useWorkers migrated to TanStack Query  
✅ Error handling added  
✅ Auto-refetch configured  
✅ Package builds successfully  
✅ App builds successfully  
✅ TEAM-353 signatures added

---

## Next Step

Continue to **TEAM_353_STEP_3_NARRATION_INTEGRATION.md** to add narration support.

---

**TEAM-353 Step 2: Hooks migrated to TanStack Query!** ✅
