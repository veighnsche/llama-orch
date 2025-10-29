# TEAM-356: NPM Library Alternatives Analysis

**Date:** Oct 29, 2025  
**Status:** 🔍 RESEARCH COMPLETE  
**Purpose:** Evaluate existing npm libraries vs custom implementation

---

## Executive Summary

**Recommendation:** Use existing npm libraries instead of building custom packages.

**Why:**
- **TanStack Query** replaces need for custom async state hooks (~90% of use cases)
- **exponential-backoff** npm package replaces custom retry logic
- **react-hooks-sse** replaces custom SSE hooks
- **Total savings:** Don't write ~500 LOC, use battle-tested libraries instead

---

## Library 1: TanStack Query (React Query)

### What It Is
Industry-standard data fetching and caching library for React. Used by thousands of production apps.

### What It Replaces
- ❌ Custom `useAsyncState` hook
- ❌ Custom `useCRUD` hook
- ❌ Custom loading/error state management
- ❌ Custom refetch logic

### Features
- ✅ Automatic caching and deduplication
- ✅ Background refetching
- ✅ Stale-while-revalidate
- ✅ Optimistic updates
- ✅ Infinite scroll support
- ✅ Pagination support
- ✅ DevTools (official GUI debugger)
- ✅ SSR support
- ✅ TypeScript support
- ✅ 47k+ GitHub stars
- ✅ Active maintenance

### Bundle Size
- **TanStack Query:** ~13kb gzipped
- **Custom hooks:** ~2kb gzipped
- **Difference:** +11kb (worth it for features)

### Usage Example

**Before (custom hook):**
```typescript
const [scripts, setScripts] = useState([])
const [loading, setLoading] = useState(true)
const [error, setError] = useState(null)
const mounted = useRef(true)

useEffect(() => {
  mounted.current = true
  
  loadScripts()
    .then(result => {
      if (mounted.current) {
        setScripts(result)
        setLoading(false)
      }
    })
    .catch(err => {
      if (mounted.current) {
        setError(err)
        setLoading(false)
      }
    })
  
  return () => {
    mounted.current = false
  }
}, [])
```

**After (TanStack Query):**
```typescript
import { useQuery } from '@tanstack/react-query'

const { data: scripts, isLoading, error, refetch } = useQuery({
  queryKey: ['scripts'],
  queryFn: () => client.listScripts(),
})
```

**Lines saved:** ~25 lines per hook

### Mutations (CRUD operations)

```typescript
import { useMutation, useQueryClient } from '@tanstack/react-query'

const queryClient = useQueryClient()

const saveMutation = useMutation({
  mutationFn: (script) => client.saveScript(script),
  onSuccess: () => {
    // Invalidate and refetch
    queryClient.invalidateQueries({ queryKey: ['scripts'] })
  },
})

// Usage
saveMutation.mutate(newScript)
```

### Installation
```bash
pnpm add @tanstack/react-query
```

### Setup (one-time)
```typescript
// App.tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

const queryClient = new QueryClient()

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <YourApp />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}
```

### Verdict
✅ **HIGHLY RECOMMENDED** - Industry standard, saves ~150 LOC per UI, includes DevTools

---

## Library 2: SWR (Alternative to TanStack Query)

### What It Is
Lightweight data fetching library by Vercel (Next.js team). Simpler API than TanStack Query.

### Comparison with TanStack Query

| Feature | TanStack Query | SWR |
|---------|---------------|-----|
| Bundle Size | 13kb | 5kb |
| DevTools | ✅ Official | 🟡 Community |
| Features | More | Less |
| API Complexity | Higher | Lower |
| GitHub Stars | 47k | 30k |
| Mutations | Built-in | Manual |
| Offline Support | ✅ | ❌ |

### Usage Example
```typescript
import useSWR from 'swr'

const { data: scripts, error, isLoading, mutate } = useSWR(
  '/api/scripts',
  () => client.listScripts()
)
```

### Verdict
✅ **GOOD ALTERNATIVE** - Simpler API, smaller bundle, but fewer features than TanStack Query

---

## Library 3: exponential-backoff (npm)

### What It Is
Utility for retrying functions with exponential backoff. **483 projects** use it.

### What It Replaces
- ❌ Custom retry logic in SDK loader
- ❌ Custom exponential backoff calculation
- ❌ Custom jitter implementation

### Features
- ✅ Exponential backoff with jitter
- ✅ Configurable max attempts
- ✅ Configurable delays
- ✅ TypeScript support
- ✅ 0 dependencies
- ✅ Well-tested (used by 483 projects)

### Bundle Size
- **exponential-backoff:** ~1kb gzipped
- **Custom implementation:** ~1kb gzipped
- **Difference:** Same size

### Usage Example

**Before (custom):**
```typescript
for (let attempt = 1; attempt <= maxAttempts; attempt++) {
  try {
    const result = await loadSDK()
    return result
  } catch (err) {
    if (attempt < maxAttempts) {
      const baseDelay = 2 ** (attempt - 1) * baseBackoffMs
      const jitter = Math.random() * baseBackoffMs
      await sleep(baseDelay + jitter)
    }
  }
}
```

**After (exponential-backoff):**
```typescript
import { backOff } from 'exponential-backoff'

const result = await backOff(() => loadSDK(), {
  numOfAttempts: 3,
  startingDelay: 300,
  timeMultiple: 2,
  jitter: 'full',
})
```

### Installation
```bash
pnpm add exponential-backoff
```

### Verdict
✅ **RECOMMENDED** - Battle-tested, same size as custom, cleaner API

---

## Library 4: react-hooks-sse

### What It Is
React hooks for Server-Sent Events (SSE). Handles connection lifecycle.

### What It Replaces
- ❌ Custom `useSSEWithHealthCheck` hook
- ❌ Custom SSE connection management
- ❌ Custom cleanup logic

### Features
- ✅ Automatic connection management
- ✅ Automatic cleanup on unmount
- ✅ Multiple event subscriptions
- ✅ Lazy source creation
- ✅ TypeScript support

### Bundle Size
- **react-hooks-sse:** ~2kb gzipped
- **Custom hook:** ~2kb gzipped
- **Difference:** Same size

### Usage Example

**Before (custom):**
```typescript
const [data, setData] = useState(null)
const [connected, setConnected] = useState(false)
const monitorRef = useRef(null)

useEffect(() => {
  const monitor = new sdk.HeartbeatMonitor(baseUrl)
  monitorRef.current = monitor
  
  monitor.start((snapshot) => {
    setData(snapshot)
    setConnected(true)
  })
  
  return () => {
    monitor.stop()
  }
}, [baseUrl])
```

**After (react-hooks-sse):**
```typescript
import { useSSE, SSEProvider } from 'react-hooks-sse'

// In App.tsx
<SSEProvider endpoint="http://localhost:7833/v1/heartbeat">
  <YourApp />
</SSEProvider>

// In component
const state = useSSE('heartbeat', { connected: false })
```

### Limitations
- ⚠️ No built-in health check before connection
- ⚠️ Requires SSEProvider wrapper
- ⚠️ Less flexible than custom implementation

### Verdict
🟡 **MAYBE** - Simpler for basic SSE, but lacks health check feature. Custom hook might be better.

---

## Library 5: react-sse-hooks (Alternative)

### What It Is
Another SSE hooks library with more features than react-hooks-sse.

### Features
- ✅ Multiple SSE sources
- ✅ Automatic reconnection
- ✅ Error handling
- ✅ TypeScript support

### Usage Example
```typescript
import { useEventSource } from 'react-sse-hooks'

const { data, error, isConnected } = useEventSource({
  source: 'http://localhost:7833/v1/heartbeat',
  options: {
    withCredentials: true,
  },
})
```

### Verdict
🟡 **MAYBE** - Better than react-hooks-sse, but still lacks health check feature

---

## Recommendation Summary

### ✅ MUST USE

**1. TanStack Query** (or SWR)
- **Replaces:** Custom async state hooks, CRUD hooks
- **Reason:** Industry standard, saves ~150 LOC per UI, includes DevTools
- **Bundle:** +13kb (worth it)
- **Install:** `pnpm add @tanstack/react-query @tanstack/react-query-devtools`

**2. exponential-backoff**
- **Replaces:** Custom retry logic in SDK loader
- **Reason:** Battle-tested by 483 projects, same size as custom
- **Bundle:** ~1kb
- **Install:** `pnpm add exponential-backoff`

### 🟡 MAYBE USE

**3. react-sse-hooks**
- **Replaces:** Part of custom SSE hook
- **Reason:** Handles connection management, but lacks health check
- **Decision:** Keep custom `useSSEWithHealthCheck` for now (health check is critical)
- **Alternative:** Use library + add health check wrapper

### ❌ DON'T BUILD CUSTOM

**Don't build:**
- ❌ `@rbee/react-hooks` - Use TanStack Query instead
- ❌ Custom retry logic - Use exponential-backoff instead
- ❌ Custom async state hooks - Use TanStack Query instead

**Still build (if needed):**
- ✅ `@rbee/sdk-loader` - But use exponential-backoff for retry logic
- 🟡 `useSSEWithHealthCheck` - Keep custom (health check is unique requirement)

---

## Updated TEAM-356 Plan

### Phase 1: Install Libraries

```bash
# Core data fetching
pnpm add @tanstack/react-query @tanstack/react-query-devtools

# Retry logic
pnpm add exponential-backoff

# Optional: SSE (if we don't keep custom)
pnpm add react-sse-hooks
```

### Phase 2: Simplify SDK Loader

**Before:** Custom retry logic (~50 LOC)

**After:** Use exponential-backoff
```typescript
import { backOff } from 'exponential-backoff'

export async function loadSDK(options: LoadOptions) {
  return backOff(
    async () => {
      const mod = await import(options.packageName)
      const wasmModule = mod.default ?? mod
      
      if (wasmModule.init) {
        await wasmModule.init(options.initArg)
      }
      
      // Validate exports
      for (const exp of options.requiredExports) {
        if (!wasmModule[exp]) {
          throw new Error(`Missing export: ${exp}`)
        }
      }
      
      return wasmModule
    },
    {
      numOfAttempts: options.maxAttempts ?? 3,
      startingDelay: options.baseBackoffMs ?? 300,
      timeMultiple: 2,
      jitter: 'full',
    }
  )
}
```

**Lines saved:** ~30 lines (retry logic extracted)

### Phase 3: Migrate to TanStack Query

**Before:** `useRhaiScripts.ts` (250 lines)

**After:** Use TanStack Query
```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

export function useRhaiScripts(baseUrl: string) {
  const { data: sdk } = useSDK()
  const queryClient = useQueryClient()
  
  // List scripts
  const { data: scripts, isLoading, error } = useQuery({
    queryKey: ['scripts', baseUrl],
    queryFn: async () => {
      if (!sdk) return []
      const client = new sdk.RhaiClient(baseUrl)
      return client.listScripts()
    },
    enabled: !!sdk,
  })
  
  // Save script
  const saveMutation = useMutation({
    mutationFn: (script) => {
      const client = new sdk.RhaiClient(baseUrl)
      return client.saveScript(script)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scripts'] })
    },
  })
  
  // Delete script
  const deleteMutation = useMutation({
    mutationFn: (id) => {
      const client = new sdk.RhaiClient(baseUrl)
      return client.deleteScript(id)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scripts'] })
    },
  })
  
  return {
    scripts: scripts ?? [],
    loading: isLoading,
    error,
    save: saveMutation.mutate,
    delete: deleteMutation.mutate,
  }
}
```

**Lines saved:** ~180 lines

### Phase 4: Keep Custom SSE Hook (with improvements)

**Decision:** Keep `useSSEWithHealthCheck` because:
- Health check before SSE is unique requirement
- Prevents CORS errors when backend offline
- Libraries don't provide this pattern

**Improvement:** Use exponential-backoff for retry logic
```typescript
import { backOff } from 'exponential-backoff'

const startMonitoring = async () => {
  await backOff(
    async () => {
      const monitor = createMonitor(baseUrl)
      const isHealthy = await monitor.checkHealth()
      
      if (!isHealthy) {
        throw new Error('Service offline')
      }
      
      monitor.start(onData)
    },
    {
      numOfAttempts: maxRetries,
      startingDelay: retryDelayMs,
    }
  )
}
```

---

## ROI Comparison

### Original Plan (Custom Packages)

| Item | LOC | Time |
|------|-----|------|
| Write @rbee/sdk-loader | 200 | 2-3h |
| Write @rbee/react-hooks | 300 | 2-3h |
| Write tests | 70 tests | 2-3h |
| Migrate Queen | -300 | 2-3h |
| **Total** | **+200 net** | **8-12h** |

### New Plan (Use Libraries)

| Item | LOC | Time |
|------|-----|------|
| Install TanStack Query | 0 | 5min |
| Install exponential-backoff | 0 | 5min |
| Simplify SDK loader | -30 | 30min |
| Migrate to TanStack Query | -180 | 1-2h |
| Keep custom SSE hook | 0 | 0h |
| **Total** | **-210 net** | **2-3h** |

**Savings:**
- **Time:** 5-9 hours saved
- **Code:** 410 fewer lines to maintain
- **Tests:** Don't need to write 70 tests (libraries already tested)
- **Maintenance:** Libraries maintained by community

---

## Final Recommendation

### ✅ DO THIS

1. **Install TanStack Query** - Replace all custom async state hooks
2. **Install exponential-backoff** - Simplify SDK loader retry logic
3. **Keep custom SSE hook** - Health check is unique requirement
4. **Don't create @rbee/react-hooks** - TanStack Query is better
5. **Simplify @rbee/sdk-loader** - Use exponential-backoff internally

### 📊 Expected Results

**Code reduction:**
- Queen UI: ~210 lines removed
- Hive UI: ~210 lines prevented
- Worker UI: ~210 lines prevented
- **Total: 630 lines saved**

**Time investment:**
- Setup: 10 minutes
- Migration: 2-3 hours
- **Total: 2-3 hours** (vs 8-12 hours for custom)

**Quality improvement:**
- ✅ Battle-tested libraries (millions of downloads)
- ✅ Active maintenance
- ✅ DevTools included (TanStack Query)
- ✅ Better TypeScript support
- ✅ More features (caching, deduplication, etc.)

---

## Migration Checklist

### Step 1: Install Dependencies
- [ ] `pnpm add @tanstack/react-query @tanstack/react-query-devtools`
- [ ] `pnpm add exponential-backoff`

### Step 2: Setup TanStack Query
- [ ] Wrap App with `QueryClientProvider`
- [ ] Add `ReactQueryDevtools` in dev mode

### Step 3: Migrate Hooks
- [ ] Replace `useRhaiScripts` with TanStack Query
- [ ] Replace `useHeartbeat` - keep custom or use library
- [ ] Update SDK loader to use exponential-backoff

### Step 4: Test
- [ ] All functionality works
- [ ] DevTools show queries
- [ ] No regressions

### Step 5: Document
- [ ] Update README with library usage
- [ ] Document migration decisions
- [ ] Handoff to next team

---

**TEAM-356: Use battle-tested libraries instead of building custom!** 🎯
