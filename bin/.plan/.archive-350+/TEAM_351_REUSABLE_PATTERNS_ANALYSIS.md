# TEAM-351: Reusable Patterns Analysis

**Date:** Oct 29, 2025  
**Status:** 🔍 ANALYSIS COMPLETE  
**Scope:** Identify patterns in Queen that will be duplicated in Hive/Worker

---

## Executive Summary

Found **5 major patterns** in Queen React packages that will be duplicated when building Hive/Worker UIs.

**Total duplication prevented:** ~250+ lines per service × 2 services = **500+ lines saved**

---

## Pattern 1: WASM/SDK Loader with Retry Logic

### Location
- `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/loader.ts` (120 lines)
- `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/globalSlot.ts` (40 lines)

### What It Does
```typescript
// Loads WASM SDK with:
// - Exponential backoff (2^attempt × baseDelay)
// - Jitter (random delay to avoid thundering herd)
// - Timeout handling (15 seconds default)
// - Retry attempts (3 attempts default)
// - Singleflight pattern (only one load at a time)
// - Environment guards (SSR, WebAssembly support)
// - Export validation (ensures SDK has required exports)
```

### Code Pattern
```typescript
async function loadSDK(opts: Required<LoadOptions>): Promise<{ sdk: SDK }> {
  // Environment guards
  if (typeof window === 'undefined') {
    throw new Error('SDK can only be initialized in browser')
  }

  let lastError: Error | undefined

  for (let attempt = 1; attempt <= opts.maxAttempts; attempt++) {
    try {
      // Import with timeout
      const mod = await withTimeout(
        import('@rbee/package-sdk'),
        opts.timeoutMs,
        `SDK import (attempt ${attempt}/${opts.maxAttempts})`
      )

      // Handle ESM/CJS shims
      const wasmModule = (mod as any).default ?? mod

      // Initialize if needed
      if (typeof wasmModule.init === 'function') {
        await withTimeout(wasmModule.init(opts.initArg), opts.timeoutMs)
      }

      // Validate exports
      if (!wasmModule.Client || !wasmModule.Monitor) {
        throw new Error('SDK exports missing')
      }

      return { sdk: wasmModule }
    } catch (err) {
      lastError = err as Error

      if (attempt < opts.maxAttempts) {
        // Exponential backoff with jitter
        const baseDelay = 2 ** (attempt - 1) * opts.baseBackoffMs
        const jitter = Math.random() * opts.baseBackoffMs
        await sleep(baseDelay + jitter)
      }
    }
  }

  throw lastError || new Error('SDK load failed after all retries')
}

// Singleflight pattern
export function loadSDKOnce(options?: LoadOptions) {
  const slot = getGlobalSlot()
  
  if (slot.value) return Promise.resolve(slot.value)
  if (slot.error) return Promise.reject(slot.error)
  if (slot.promise) return slot.promise
  
  slot.promise = loadSDK(opts)
    .then(result => {
      slot.value = result
      slot.promise = undefined
      return result
    })
    .catch(err => {
      slot.error = err
      slot.promise = undefined
      throw err
    })
  
  return slot.promise
}
```

### Will Repeat In
- **Hive UI:** If Hive uses WASM SDK
- **Worker UI:** If Worker uses WASM SDK

### Extraction Target
**Package:** `@rbee/sdk-loader` or `@rbee/wasm-loader`

**API:**
```typescript
export function createSDKLoader<T>(config: {
  packageName: string          // e.g., '@rbee/queen-rbee-sdk'
  requiredExports: string[]    // e.g., ['Client', 'Monitor']
  timeout?: number             // Default: 15000
  maxAttempts?: number         // Default: 3
  baseBackoffMs?: number       // Default: 300
}) {
  return {
    load: (options?) => Promise<T>
    loadOnce: (options?) => Promise<T>
  }
}
```

**Usage:**
```typescript
// Queen
const queenLoader = createSDKLoader({
  packageName: '@rbee/queen-rbee-sdk',
  requiredExports: ['QueenClient', 'HeartbeatMonitor', 'RhaiClient']
})

// Hive (future)
const hiveLoader = createSDKLoader({
  packageName: '@rbee/rbee-hive-sdk',
  requiredExports: ['HiveClient', 'ModelManager']
})
```

### Lines Saved
- **Per service:** 160 lines (loader + globalSlot + utils)
- **Total (3 services):** 480 lines

---

## Pattern 2: Async State Management Hook

### Location
Used in every hook:
- `useHeartbeat.ts` (lines 36-39, 74-86)
- `useRhaiScripts.ts` (lines 59-65, 98-104, 154-160)
- Future: `useModels`, `useWorkers`, `useInference`

### What It Does
```typescript
// Standard pattern for async data loading:
// - loading state (boolean)
// - error state (Error | null)
// - data state (T | null)
// - mounted ref (prevents state updates after unmount)
```

### Code Pattern
```typescript
export function useAsyncData<T>() {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true

    asyncOperation()
      .then(result => {
        if (mountedRef.current) {
          setData(result)
          setLoading(false)
        }
      })
      .catch(err => {
        if (mountedRef.current) {
          setError(err)
          setLoading(false)
        }
      })

    return () => {
      mountedRef.current = false
    }
  }, [deps])

  return { data, loading, error }
}
```

### Will Repeat In
- **Every hook in Hive UI:** `useModels`, `useWorkers`, `useDevices`
- **Every hook in Worker UI:** `useInference`, `useHealth`, `useMetrics`

### Extraction Target
**Package:** `@rbee/react-hooks`

**API:**
```typescript
export function useAsyncState<T>(
  asyncFn: () => Promise<T>,
  deps: DependencyList
): {
  data: T | null
  loading: boolean
  error: Error | null
  refetch: () => void
}
```

**Usage:**
```typescript
// Instead of this boilerplate in every hook:
const [models, setModels] = useState<Model[]>([])
const [loading, setLoading] = useState(true)
const [error, setError] = useState<Error | null>(null)
const mounted = useRef(true)
// ... 20 lines of useEffect logic

// Use this:
const { data: models, loading, error } = useAsyncState(
  () => listModels(),
  []
)
```

### Lines Saved
- **Per hook:** ~30 lines
- **Per service:** ~90 lines (3 hooks average)
- **Total (3 services):** 270 lines

---

## Pattern 3: Health Check Before SSE Connection

### Location
- `useHeartbeat.ts` (lines 48-74)

### What It Does
```typescript
// Prevents CORS errors when backend is offline
// 1. Check health endpoint first
// 2. Only start SSE if healthy
// 3. Show proper error if offline
```

### Code Pattern
```typescript
const startMonitoring = async () => {
  try {
    const monitor = new sdk.HeartbeatMonitor(baseUrl)
    
    // Check health before SSE
    const isHealthy = await monitor.checkHealth()
    
    if (!mounted) return
    
    if (!isHealthy) {
      setError(new Error('Service is offline'))
      setConnected(false)
      return
    }
    
    // Only start SSE if healthy
    monitor.start((snapshot: any) => {
      if (!mounted) return
      setData(snapshot)
      setConnected(true)
      setError(null)
    })
  } catch (err) {
    if (!mounted) return
    setError(err as Error)
    setConnected(false)
  }
}
```

### Will Repeat In
- **Hive UI:** Monitoring hive heartbeat/status
- **Worker UI:** Monitoring worker heartbeat/status

### Extraction Target
**Package:** `@rbee/react-hooks`

**API:**
```typescript
export function useSSEWithHealthCheck<T>(
  createMonitor: (baseUrl: string) => Monitor<T>,
  baseUrl: string
): {
  data: T | null
  connected: boolean
  loading: boolean
  error: Error | null
}
```

**Usage:**
```typescript
// Queen
const { data, connected } = useSSEWithHealthCheck(
  (baseUrl) => new sdk.HeartbeatMonitor(baseUrl),
  'http://localhost:7833'
)

// Hive (future)
const { data, connected } = useSSEWithHealthCheck(
  (baseUrl) => new sdk.HiveMonitor(baseUrl),
  'http://localhost:7835'
)
```

### Lines Saved
- **Per service:** ~25 lines
- **Total (3 services):** 75 lines

---

## Pattern 4: Hardcoded Base URLs

### Location
**EVERY hook has this:**
- `useHeartbeat.ts` line 35: `baseUrl: string = 'http://localhost:7833'`
- `useRhaiScripts.ts` line 57: `baseUrl: string = 'http://localhost:7833'`

### What It Does
```typescript
// Hardcodes service URLs instead of using config
export function useHeartbeat(
  baseUrl: string = 'http://localhost:7833'  // ❌ Hardcoded
)
```

### Will Repeat In
- **Every Hive hook:** `baseUrl = 'http://localhost:7835'`
- **Every Worker hook:** `baseUrl = 'http://localhost:8080'`

### Extraction Target
**Package:** `@rbee/shared-config` (already created!)

**Fix:**
```typescript
// Before (hardcoded)
export function useHeartbeat(
  baseUrl: string = 'http://localhost:7833'
)

// After (use shared config)
import { getServiceUrl } from '@rbee/shared-config'

export function useHeartbeat(
  baseUrl: string = getServiceUrl('queen', 'prod')
)
```

### Lines Saved
- **Not about lines saved** - about **single source of truth**
- Prevents port configuration drift
- Easier to change ports (one place)

---

## Pattern 5: CRUD Operation Pattern

### Location
- `useRhaiScripts.ts` (lines 74-256)

### What It Does
```typescript
// Standard CRUD operations:
// - list() - Get all items
// - get(id) - Get single item
// - create(item) - Create new item
// - update(item) - Update existing item
// - delete(id) - Delete item
// Each with loading/error/validation
```

### Code Pattern
```typescript
// List operation
const loadScripts = async () => {
  if (!sdk) return
  
  setLoading(true)
  setError(null)
  try {
    const client = new sdk.RhaiClient(baseUrl)
    const result = await client.listScripts()
    const scriptList = JSON.parse(JSON.stringify(result))
    
    if (Array.isArray(scriptList)) {
      setScripts(scriptList)
    } else {
      console.warn('Backend returned non-array')
      setScripts([])
    }
  } catch (err) {
    setError(err as Error)
    setScripts([])
  } finally {
    setLoading(false)
  }
}

// Similar patterns for:
// - selectScript (get)
// - saveScript (create/update)
// - deleteScript (delete)
```

### Will Repeat In
- **Hive UI:** Managing models, workers, devices
- **Worker UI:** Managing inference requests, configs

### Extraction Target
**Package:** `@rbee/react-hooks`

**API:**
```typescript
export function useCRUD<T>(
  client: {
    list: () => Promise<T[]>
    get: (id: string) => Promise<T>
    save: (item: T) => Promise<T>
    delete: (id: string) => Promise<void>
  }
): {
  items: T[]
  current: T | null
  loading: boolean
  error: Error | null
  load: () => Promise<void>
  select: (id: string) => Promise<void>
  save: (item: T) => Promise<void>
  delete: (id: string) => Promise<void>
}
```

**Usage:**
```typescript
// Instead of 180 lines of CRUD boilerplate:
const {
  items: scripts,
  current: currentScript,
  loading,
  save,
  delete: deleteScript
} = useCRUD({
  list: () => client.listScripts(),
  get: (id) => client.getScript(id),
  save: (script) => client.saveScript(script),
  delete: (id) => client.deleteScript(id)
})
```

### Lines Saved
- **Per CRUD resource:** ~150 lines
- **Per service:** ~300 lines (2 resources average)
- **Total (3 services):** 900 lines

---

## Recommended Packages to Create

### 1. `@rbee/sdk-loader`
**Purpose:** Generic WASM/SDK loading with retry logic

**Files:**
- `src/loader.ts` - Core loader with retry/backoff
- `src/singleflight.ts` - Global slot pattern
- `src/types.ts` - LoadOptions, SDKConfig types
- `src/index.ts` - Exports

**Estimated Size:** 200 lines

**Saves:** 160 lines × 3 services = **480 lines**

---

### 2. `@rbee/react-hooks`
**Purpose:** Reusable React hooks for common patterns

**Files:**
- `src/useAsyncState.ts` - Async data loading
- `src/useSSEWithHealthCheck.ts` - SSE with health check
- `src/useCRUD.ts` - CRUD operations
- `src/usePolling.ts` - Polling with cleanup
- `src/index.ts` - Exports

**Estimated Size:** 300 lines

**Saves:** 400 lines × 3 services = **1,200 lines**

---

## Total Impact

### Lines of Code

| Pattern | Lines per Service | Services | Total Saved |
|---------|-------------------|----------|-------------|
| SDK Loader | 160 | 3 | 480 |
| Async State | 90 | 3 | 270 |
| SSE Health Check | 25 | 3 | 75 |
| CRUD Operations | 150 | 3 | 450 |
| **TOTAL** | **425** | **3** | **1,275** |

### Package Investment

| Package | Lines to Write | Lines Saved | ROI |
|---------|----------------|-------------|-----|
| `@rbee/sdk-loader` | 200 | 480 | 2.4x |
| `@rbee/react-hooks` | 300 | 795 | 2.7x |
| **TOTAL** | **500** | **1,275** | **2.5x** |

---

## Recommendation

### Add to TEAM-351 (Phase 1)

**Create 2 more packages:**

1. ✅ `@rbee/sdk-loader` - WASM/SDK loading utilities
2. ✅ `@rbee/react-hooks` - Reusable React hooks

**Update Queen to use them:**
- Migrate `loader.ts` to use `@rbee/sdk-loader`
- Migrate hooks to use `@rbee/react-hooks`
- Validate pattern works

**Benefits:**
- Prevents duplication BEFORE building Hive/Worker
- Validates shared packages work in real code
- Queen becomes cleaner (~400 lines removed)
- Hive/Worker start with clean foundation

---

## Phased Approach (Alternative)

If TEAM-351 shouldn't grow:

### Phase 1 (TEAM-351) - Current
- ✅ `@rbee/shared-config`
- ✅ `@rbee/narration-client`
- ✅ `@rbee/iframe-bridge`
- ✅ `@rbee/dev-utils`

### Phase 1.5 (TEAM-351-EXTENDED)
- 🆕 `@rbee/sdk-loader`
- 🆕 `@rbee/react-hooks`
- 🔄 Update Queen to use all 6 packages

### Phase 2 (TEAM-352+)
- Build Hive UI using shared packages
- Build Worker UI using shared packages
- No duplication!

---

## Conclusion

**YES - Significant duplication will occur if we don't extract these patterns!**

**Patterns found:**
1. ✅ SDK Loader (160 lines × 3 = 480 lines)
2. ✅ Async State (90 lines × 3 = 270 lines)
3. ✅ SSE Health Check (25 lines × 3 = 75 lines)
4. ✅ Hardcoded URLs (already solved by `@rbee/shared-config`)
5. ✅ CRUD Operations (150 lines × 3 = 450 lines)

**Total duplication prevented:** 1,275 lines

**Recommended action:** Create 2 more shared packages before building Hive/Worker UIs.

---

**TEAM-351: Reusable patterns analysis complete!** 🎯
