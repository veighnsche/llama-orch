# TEAM-356: Re-exports Removed - Use TanStack Query Directly

**Date:** Oct 30, 2025  
**Status:** ✅ COMPLETE  
**Decision:** Remove re-exports, users import TanStack Query directly

---

## What Changed

### Removed from @rbee/react-hooks
- ❌ Re-exports of `useQuery`, `useMutation`, `useQueryClient`, `useInfiniteQuery`
- ❌ `@tanstack/react-query` dependency

### What @rbee/react-hooks Now Exports
- ✅ `useSSEWithHealthCheck` only (custom hook with unique requirements)

### Users Now Import Directly
```typescript
// Before (re-exported)
import { useQuery } from '@rbee/react-hooks'

// After (direct import)
import { useQuery } from '@tanstack/react-query'
import { useSSEWithHealthCheck } from '@rbee/react-hooks'
```

---

## Why This Is Better

1. **Clearer Dependencies** - Users explicitly install what they use
2. **No Middleman** - Direct import from source library
3. **Version Control** - Users control TanStack Query version
4. **Smaller Package** - @rbee/react-hooks is now minimal (just SSE hook)
5. **Standard Practice** - Don't re-export third-party libraries

---

## Installation Instructions

### For New Projects
```bash
# Install TanStack Query directly
pnpm add @tanstack/react-query @tanstack/react-query-devtools

# Install our custom hooks
pnpm add @rbee/react-hooks
```

### Setup (One-time)
```typescript
// App.tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      retry: 1,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <YourApp />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}
```

### Usage
```typescript
// Data fetching - import from TanStack Query
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

// SSE with health check - import from our package
import { useSSEWithHealthCheck } from '@rbee/react-hooks'

function MyComponent() {
  // TanStack Query for data
  const { data, isLoading, error } = useQuery({
    queryKey: ['data'],
    queryFn: () => fetchData(),
  })

  // Our custom hook for SSE
  const { data: heartbeat, connected } = useSSEWithHealthCheck(
    (url) => new sdk.Monitor(url),
    baseUrl
  )
}
```

---

## Updated Planning Documents

All planning documents updated to use direct imports:

### ✅ Updated Files
1. `TEAM_352_QUEEN_MIGRATION_PHASE.md` - Use `@tanstack/react-query` directly
2. `TEAM_352_STEP_2_HOOKS_MIGRATION.md` - Added QueryClientProvider setup
3. `TEAM_353_HIVE_UI_PHASE.md` - Use `@tanstack/react-query` directly
4. `TEAM_354_WORKER_UI_PHASE.md` - Use `@tanstack/react-query` directly
5. `frontend/packages/react-hooks/README.md` - Updated all examples
6. `frontend/packages/react-hooks/src/index.ts` - Removed re-exports
7. `frontend/packages/react-hooks/package.json` - Removed dependency

### Key Changes in Plans
- **Before:** `import { useQuery } from '@rbee/react-hooks'`
- **After:** `import { useQuery } from '@tanstack/react-query'`
- **Added:** QueryClientProvider setup step in all migration guides
- **Updated:** All code examples to show direct imports

---

## Package Structure

### @rbee/react-hooks (Final)
```
frontend/packages/react-hooks/
├── package.json              # No TanStack Query dependency
├── tsconfig.json
├── vitest.config.ts
├── README.md                 # Updated with direct import examples
└── src/
    ├── index.ts              # Exports only useSSEWithHealthCheck
    ├── useSSEWithHealthCheck.ts        # Custom hook
    └── useSSEWithHealthCheck.test.tsx  # 11 tests
```

**Dependencies:**
- Peer: `react@^19.0.0`
- Dev: Testing libraries only
- **No runtime dependencies** (except React peer)

---

## Verification

### Build Status ✅
```bash
$ cd frontend/packages/react-hooks
$ pnpm build && pnpm test
# ✅ Build: SUCCESS
# ✅ Tests: 11/11 passing
```

### Package Size
- Before (with re-exports): ~150 LOC
- After (SSE hook only): ~100 LOC
- **Reduction:** 50 LOC (33%)

---

## Migration Guide for Existing Code

If you have code using the old re-exports:

### Step 1: Install TanStack Query
```bash
pnpm add @tanstack/react-query @tanstack/react-query-devtools
```

### Step 2: Update Imports
```typescript
// Before
import { useQuery, useSSEWithHealthCheck } from '@rbee/react-hooks'

// After
import { useQuery } from '@tanstack/react-query'
import { useSSEWithHealthCheck } from '@rbee/react-hooks'
```

### Step 3: Setup Provider (if not already done)
```typescript
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient()

<QueryClientProvider client={queryClient}>
  <App />
</QueryClientProvider>
```

---

## Benefits Summary

| Aspect | Before (Re-exports) | After (Direct) |
|--------|---------------------|----------------|
| **Clarity** | 🟡 Hidden dependency | ✅ Explicit |
| **Version Control** | 🟡 Locked to our version | ✅ User controls |
| **Package Size** | 🟡 150 LOC | ✅ 100 LOC |
| **Maintenance** | 🟡 We maintain re-exports | ✅ Minimal |
| **Standard Practice** | ❌ Non-standard | ✅ Standard |
| **Dependencies** | 🟡 Transitive | ✅ Direct |

---

## Final Metrics

### @rbee/react-hooks Package
- **Exports:** 1 hook (`useSSEWithHealthCheck`)
- **Tests:** 11 passing
- **LOC:** ~100
- **Dependencies:** 0 runtime (React peer only)
- **Purpose:** Custom hooks with unique requirements only

### TanStack Query
- **Installation:** User installs directly
- **Version:** User controls
- **Usage:** Import directly from `@tanstack/react-query`

---

**TEAM-356: Clean separation of concerns achieved!** ✅

**Our package:** Custom hooks only  
**TanStack Query:** Users install and import directly  
**Result:** Clearer, simpler, more maintainable
