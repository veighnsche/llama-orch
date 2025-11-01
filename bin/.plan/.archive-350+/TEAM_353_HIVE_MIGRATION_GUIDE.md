# TEAM-353: Hive UI Migration to Shared Packages

**Status:** 📋 READY FOR MIGRATION  
**Estimated Time:** 2-3 hours  
**Priority:** HIGH  
**Pattern:** Follow TEAM-352 (Queen migration)

---

## ⚠️ CRITICAL: Packages Already Exist!

**Location:** `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui`

**Existing structure:**
```
bin/20_rbee_hive/ui/
├── app/                        # Hive UI app (already exists)
└── packages/
    ├── rbee-hive-react/        # React hooks (already exists)
    └── rbee-hive-sdk/          # WASM SDK (already exists)
```

**This is a MIGRATION, not a new implementation!**

---

## Current State Analysis

### What Exists

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/index.ts`

```typescript
// CURRENT IMPLEMENTATION - Custom async state management
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

### What's Missing

❌ No @rbee/sdk-loader usage  
❌ No @rbee/narration-client usage  
❌ No @rbee/shared-config usage  
❌ No @rbee/dev-utils usage  
❌ Manual async state management  
❌ No TanStack Query  
❌ Hardcoded polling intervals  
❌ No error handling  
❌ No retry logic  

---

## Migration Steps

### Step 1: Add Shared Package Dependencies

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/package.json`

**Add these dependencies:**
```json
{
  "dependencies": {
    "@rbee/rbee-hive-sdk": "workspace:*",
    "@rbee/sdk-loader": "workspace:*",
    "@rbee/react-hooks": "workspace:*",
    "@rbee/narration-client": "workspace:*",
    "@tanstack/react-query": "^5.0.0"
  }
}
```

### Step 2: Migrate to TanStack Query

**Replace manual state management with TanStack Query:**

```typescript
// TEAM-353: Migrated to use TanStack Query
import { useQuery } from '@tanstack/react-query'
import { listModels, listWorkers, type Model, type Worker } from '@rbee/rbee-hive-sdk'

export function useModels() {
  const { data: models, isLoading: loading, error } = useQuery({
    queryKey: ['hive-models'],
    queryFn: listModels,
    staleTime: 30000, // Models change less frequently
  })

  return { 
    models: models || [], 
    loading,
    error: error as Error | null
  }
}

export function useWorkers() {
  const { data: workers, isLoading: loading, error } = useQuery({
    queryKey: ['hive-workers'],
    queryFn: listWorkers,
    staleTime: 5000,
    refetchInterval: 2000, // Auto-refetch every 2 seconds
  })

  return { 
    workers: workers || [], 
    loading,
    error: error as Error | null
  }
}
```

**Benefits:**
- ✅ Automatic caching
- ✅ Automatic refetching
- ✅ Built-in error handling
- ✅ No manual state management
- ✅ Declarative polling

### Step 3: Add Narration Support

**If Hive SDK has operations that need narration:**

```typescript
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

export function useHiveOperations() {
  const spawnWorker = async (modelId: string) => {
    const narrationHandler = createStreamHandler(SERVICES.hive, (event) => {
      console.log('[Hive] Narration:', event)
    }, {
      debug: true,
      validate: true,
    })

    // Use narration handler with SDK operations
    // ...
  }

  return { spawnWorker }
}
```

### Step 4: Update App to Use QueryClient

**File:** `bin/20_rbee_hive/ui/app/src/App.tsx`

```typescript
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { logStartupMode } from '@rbee/dev-utils'

logStartupMode("HIVE UI", import.meta.env.DEV, 7836)

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
      {/* Your app content */}
    </QueryClientProvider>
  )
}
```

### Step 5: Remove Hardcoded URLs

**Search for hardcoded URLs:**
```bash
grep -r "localhost:[0-9]" bin/20_rbee_hive/ui/app/src --include="*.ts" --include="*.tsx"
```

**Replace with @rbee/shared-config:**
```typescript
import { getIframeUrl, getAllowedOrigins } from '@rbee/shared-config'

const url = getIframeUrl('hive', isDev)
const origins = getAllowedOrigins()
```

---

## Testing Checklist

- [ ] `pnpm install` - dependencies added
- [ ] `pnpm build` (rbee-hive-react) - success
- [ ] `pnpm build` (hive-ui app) - success
- [ ] `pnpm dev` - app starts
- [ ] useModels hook works
- [ ] useWorkers hook works
- [ ] Auto-refetch works
- [ ] No TypeScript errors
- [ ] No console errors
- [ ] TEAM-353 signatures added

---

## Code Savings

**Before (custom implementation):**
- Manual state: ~40 LOC
- Manual polling: ~10 LOC
- No error handling
- No retry logic

**After (shared packages):**
- TanStack Query: ~20 LOC
- Automatic error handling
- Automatic retry
- **Savings: ~30 LOC + better features**

---

## Reference

**Follow the same pattern as Queen migration:**
- `bin/.plan/TEAM_352_STEP_1_SDK_LOADER_MIGRATION.md`
- `bin/.plan/TEAM_352_STEP_3_NARRATION_MIGRATION.md`
- `bin/.plan/TEAM_352_STEP_4_CONFIG_CLEANUP.md`

---

**TEAM-353: Migrate existing Hive UI to use shared packages!** 🚀
