# TEAM-356: Migrate to TanStack Query

**Date:** Oct 30, 2025  
**Status:** 📋 PLAN  
**Decision:** Use TanStack Query for consistency across codebase

---

## Decision Rationale

**Original Plan:** Keep custom `useAsyncState` (smaller, simpler)

**Updated Decision:** ✅ **MIGRATE TO TANSTACK QUERY**

**Why:**
- ✅ **Consistency** - Already using TanStack Query elsewhere in codebase
- ✅ **Team familiarity** - Team already knows TanStack Query API
- ✅ **Better DX** - DevTools for debugging
- ✅ **Future-proof** - Caching/deduplication available when needed
- ✅ **Industry standard** - 47k+ stars, active maintenance

**Trade-offs:**
- ❌ +11kb bundle size (acceptable for consistency)
- ❌ More complex API (but team already familiar)

---

## Migration Plan

### Phase 1: Remove Custom useAsyncState

**Files to Delete:**
1. `frontend/packages/react-hooks/src/useAsyncState.ts` (117 LOC)
2. `frontend/packages/react-hooks/src/useAsyncState.test.tsx` (8 tests)

**Keep:**
- `useSSEWithHealthCheck.ts` - Still needed (health check is unique)
- `useSSEWithHealthCheck.test.tsx` - Still needed

### Phase 2: Update Package Dependencies

**File:** `frontend/packages/react-hooks/package.json`

**Remove:**
```json
{
  "name": "@rbee/react-hooks",
  "dependencies": {
    "react": "^19.0.0"
  }
}
```

**Update to:**
```json
{
  "name": "@rbee/react-hooks",
  "dependencies": {
    "@tanstack/react-query": "^5.0.0",
    "react": "^19.0.0"
  },
  "peerDependencies": {
    "@tanstack/react-query": "^5.0.0",
    "react": "^19.0.0"
  }
}
```

### Phase 3: Update Exports

**File:** `frontend/packages/react-hooks/src/index.ts`

**Before:**
```typescript
export * from './useAsyncState'
export * from './useSSEWithHealthCheck'
```

**After:**
```typescript
// Re-export TanStack Query for convenience
export { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

// Custom hooks
export * from './useSSEWithHealthCheck'
```

### Phase 4: Update README

**File:** `frontend/packages/react-hooks/README.md`

Add TanStack Query section:

```markdown
## Hooks

### Data Fetching: TanStack Query

For async data fetching, use TanStack Query directly:

\`\`\`typescript
import { useQuery } from '@rbee/react-hooks'

const { data, isLoading, error, refetch } = useQuery({
  queryKey: ['scripts', userId],
  queryFn: async () => {
    const response = await fetch(`/api/scripts?user=${userId}`)
    return response.json()
  },
})
\`\`\`

See [TanStack Query docs](https://tanstack.com/query/latest) for full API.

### SSE with Health Check: useSSEWithHealthCheck

For SSE connections with health check (prevents CORS errors):

\`\`\`typescript
import { useSSEWithHealthCheck } from '@rbee/react-hooks'

const { data, connected, loading, error, retry } = useSSEWithHealthCheck(
  (url) => new sdk.HeartbeatMonitor(url),
  'http://localhost:7833',
  { autoRetry: true, maxRetries: 3 }
)
\`\`\`
```

---

## Usage Examples

### Before (Custom useAsyncState)

```typescript
import { useAsyncState } from '@rbee/react-hooks'

const { data: scripts, loading, error, refetch } = useAsyncState(
  async () => {
    const client = new sdk.RhaiClient(baseUrl)
    return client.listScripts()
  },
  [baseUrl]
)
```

### After (TanStack Query)

```typescript
import { useQuery } from '@rbee/react-hooks'

const { data: scripts, isLoading, error, refetch } = useQuery({
  queryKey: ['scripts', baseUrl],
  queryFn: async () => {
    const client = new sdk.RhaiClient(baseUrl)
    return client.listScripts()
  },
})
```

**Differences:**
- `loading` → `isLoading`
- Explicit `queryKey` (enables caching)
- No dependency array (handled by `queryKey`)

---

## Migration Checklist

### Step 1: Update @rbee/react-hooks Package
- [ ] Delete `src/useAsyncState.ts`
- [ ] Delete `src/useAsyncState.test.tsx`
- [ ] Update `package.json` dependencies
- [ ] Update `src/index.ts` exports
- [ ] Update `README.md` with TanStack Query examples
- [ ] Run `pnpm install`
- [ ] Run `pnpm build` - should pass
- [ ] Run `pnpm test` - should pass (11 tests for SSE hook)

### Step 2: Update Planning Documents
- [ ] Update `TEAM_352_QUEEN_MIGRATION_PHASE.md`
- [ ] Update `TEAM_353_HIVE_UI_PHASE.md`
- [ ] Update `TEAM_354_WORKER_UI_PHASE.md`
- [ ] Update `TEAM_356_FINAL_SUMMARY.md`

### Step 3: Add TanStack Query Setup to Apps

**Queen App:**
```typescript
// bin/10_queen_rbee/ui/app/src/main.tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000, // 5 seconds
      retry: 1,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </React.StrictMode>
)
```

**Hive App:** Same pattern  
**Worker App:** Same pattern

---

## Updated Package Structure

### @rbee/react-hooks (After Migration)

```
frontend/packages/react-hooks/
├── package.json              # Updated with TanStack Query
├── tsconfig.json
├── vitest.config.ts
├── README.md                 # Updated docs
└── src/
    ├── index.ts              # Re-exports TanStack Query + custom hooks
    ├── useSSEWithHealthCheck.ts        # Keep (unique requirement)
    └── useSSEWithHealthCheck.test.tsx  # Keep (11 tests)
```

**Test Count:**
- Before: 19 tests (8 useAsyncState + 11 SSE)
- After: 11 tests (11 SSE only)
- TanStack Query: Thousands of tests (library)

---

## Benefits After Migration

### 1. Consistency ✅
- Same data fetching pattern across all UIs
- Team already familiar with TanStack Query
- No context switching between custom and library hooks

### 2. DevTools ✅
```typescript
// Automatic DevTools in development
<ReactQueryDevtools initialIsOpen={false} />
```

**Features:**
- View all queries and their state
- Inspect cache
- Manually trigger refetch
- See query timings
- Debug stale data issues

### 3. Advanced Features (When Needed) ✅
- **Caching:** Automatic deduplication of requests
- **Background refetch:** Keep data fresh
- **Optimistic updates:** Update UI before server responds
- **Pagination:** Built-in support
- **Infinite scroll:** Built-in support

### 4. Better Error Handling ✅
```typescript
const { data, error, isError, isLoading } = useQuery({
  queryKey: ['data'],
  queryFn: fetchData,
  retry: 3,
  retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
})
```

---

## Code Reduction

### Before Migration
- Custom `useAsyncState`: 117 LOC
- Custom tests: 8 tests
- Total custom code: ~200 LOC

### After Migration
- TanStack Query: 0 LOC (library)
- Re-export: 1 line
- Total custom code: 1 LOC

**Savings:** ~199 LOC removed from our codebase

---

## Updated Metrics

### Package Size
- Before: ~2kb (custom useAsyncState)
- After: ~13kb (TanStack Query)
- **Increase:** +11kb (acceptable for consistency)

### Test Count
- Before: 19 tests (8 useAsyncState + 11 SSE)
- After: 11 tests (11 SSE only)
- **Reduction:** -8 tests (TanStack Query already tested)

### Maintenance Burden
- Before: Maintain custom hook + tests
- After: Just use library
- **Reduction:** ~200 LOC we don't maintain

---

## Migration Timeline

**Estimated Time:** 1-2 hours

1. **Update @rbee/react-hooks** (30 min)
   - Delete files
   - Update dependencies
   - Update exports
   - Update README

2. **Update planning docs** (30 min)
   - TEAM_352, TEAM_353, TEAM_354
   - TEAM_356_FINAL_SUMMARY

3. **Test** (30 min)
   - Build package
   - Run tests
   - Verify exports work

---

## Next Steps for TEAM-352

When migrating Queen UI:

**Before:**
```typescript
import { useAsyncState } from '@rbee/react-hooks'

const { data, loading, error } = useAsyncState(
  async () => fetchScripts(),
  [userId]
)
```

**After:**
```typescript
import { useQuery } from '@rbee/react-hooks'

const { data, isLoading, error } = useQuery({
  queryKey: ['scripts', userId],
  queryFn: () => fetchScripts(),
})
```

**Setup required:**
```typescript
// Add to App.tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient()

<QueryClientProvider client={queryClient}>
  <App />
</QueryClientProvider>
```

---

## Acceptance Criteria

- [ ] `useAsyncState.ts` deleted
- [ ] `useAsyncState.test.tsx` deleted
- [ ] TanStack Query added to dependencies
- [ ] Exports updated to re-export TanStack Query
- [ ] README updated with TanStack Query examples
- [ ] Package builds successfully
- [ ] 11 SSE tests still passing
- [ ] Planning docs updated
- [ ] TEAM-352 has clear migration guide

---

**TEAM-356: Migrate to TanStack Query for consistency!** 🎯
