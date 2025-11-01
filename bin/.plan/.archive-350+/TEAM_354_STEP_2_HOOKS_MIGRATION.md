# TEAM-354 Step 2: Worker UI - Migrate Hooks to TanStack Query

**Estimated Time:** 30-45 minutes  
**Priority:** HIGH  
**Previous Step:** TEAM_354_STEP_1_DEPENDENCY_MIGRATION.md  
**Next Step:** TEAM_354_STEP_3_NARRATION_INTEGRATION.md

---

## Mission

Migrate existing Worker hooks from manual state management to TanStack Query.

**Location:** `bin/30_llm_worker_rbee/ui/packages/rbee-worker-react/src/`

---

## Deliverables Checklist

- [ ] Hooks migrated to TanStack Query
- [ ] Error handling added
- [ ] Package builds successfully
- [ ] TEAM-354 signatures added

---

## Step 1: Check Current Implementation

**File:** `bin/30_llm_worker_rbee/ui/packages/rbee-worker-react/src/index.ts`

Read the current implementation to understand what hooks exist.

---

## Step 2: Migrate to TanStack Query

**File:** `bin/30_llm_worker_rbee/ui/packages/rbee-worker-react/src/index.ts`

**Replace manual state management with TanStack Query:**

```typescript
// TEAM-354: Migrated to use TanStack Query (no manual state management)
import { useQuery, useMutation } from '@tanstack/react-query'
import type { WorkerClient } from '@rbee/rbee-worker-sdk'

/**
 * Hook for worker status
 * 
 * TEAM-354: Migrated to TanStack Query
 */
export function useWorkerStatus() {
  const { 
    data: status, 
    isLoading: loading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['worker-status'],
    queryFn: async () => {
      // TODO: Replace with actual SDK call
      // const client = new WorkerClient(baseUrl)
      // return await client.getStatus()
      return { status: 'idle', model: null }
    },
    staleTime: 5000,
    refetchInterval: 2000, // Auto-refetch every 2 seconds
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })

  return { 
    status: status || { status: 'idle', model: null }, 
    loading,
    error: error as Error | null,
    refetch
  }
}

/**
 * Hook for inference operations
 * 
 * TEAM-354: Uses TanStack Query mutations
 */
export function useInference() {
  const mutation = useMutation({
    mutationFn: async (prompt: string) => {
      // TODO: Replace with actual SDK call
      // const client = new WorkerClient(baseUrl)
      // return await client.infer(prompt)
      return { text: 'Response', tokens: 0 }
    },
    retry: 1,
  })

  return {
    infer: mutation.mutate,
    inferAsync: mutation.mutateAsync,
    inferring: mutation.isPending,
    result: mutation.data,
    error: mutation.error as Error | null,
  }
}

// Re-export types
export type { WorkerClient } from '@rbee/rbee-worker-sdk'
```

---

## Step 3: Build Package

```bash
cd bin/30_llm_worker_rbee/ui/packages/rbee-worker-react
pnpm build
```

**Expected output:**
```
> @rbee/rbee-worker-react@0.1.0 build
> tsc

✓ Built successfully
```

---

## Step 4: Update App to Use QueryClient

**File:** `bin/30_llm_worker_rbee/ui/app/src/App.tsx`

```typescript
// TEAM-354: Worker UI - Inference monitoring and control
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from "next-themes"
import { logStartupMode } from '@rbee/dev-utils'

// TEAM-354: Use shared startup logging
logStartupMode("WORKER UI", import.meta.env.DEV, 7838)

// TEAM-354: Create QueryClient for TanStack Query
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
      <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
        <div className="min-h-screen bg-background font-sans">
          <div className="p-6">
            <h1 className="text-3xl font-bold">Worker Dashboard</h1>
            <p className="text-muted-foreground">
              Inference Monitoring & Control
            </p>
          </div>
        </div>
      </ThemeProvider>
    </QueryClientProvider>
  )
}

export default App
```

---

## Step 5: Build App

```bash
cd bin/30_llm_worker_rbee/ui/app
pnpm build
```

---

## Testing Checklist

- [ ] `pnpm build` (rbee-worker-react) - success
- [ ] `pnpm build` (app) - success
- [ ] Hooks use TanStack Query
- [ ] Error handling works
- [ ] No TypeScript errors
- [ ] TEAM-354 signatures added

---

## Success Criteria

✅ Hooks migrated to TanStack Query  
✅ Error handling added  
✅ Package builds successfully  
✅ App builds successfully  
✅ TEAM-354 signatures added

---

## Next Step

Continue to **TEAM_354_STEP_3_NARRATION_INTEGRATION.md** to add narration support.

---

**TEAM-354 Step 2: Hooks migrated to TanStack Query!** ✅
