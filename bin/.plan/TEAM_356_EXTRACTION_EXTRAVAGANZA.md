# TEAM-356: Extraction Extravaganza

**Mission:** Extract reusable patterns from Queen UI to prevent duplication in Hive/Worker UIs  
**Status:** 📋 PLANNED  
**Based on:** TEAM-351 Reusable Patterns Analysis

---

## Mission Overview

Create 2 additional shared packages to prevent **~500-1,275 lines of duplication** across Hive and Worker UIs.

**Packages to create:**
1. `@rbee/sdk-loader` - WASM/SDK loading with retry logic
2. `@rbee/react-hooks` - Reusable React hooks for common patterns

**Then:** Migrate Queen UI to use these packages to validate they work.

---

## Package 1: @rbee/sdk-loader

### Purpose
Generic WASM/SDK loader with exponential backoff, retry logic, timeout handling, and singleflight pattern.

### Files to Create

```
frontend/packages/sdk-loader/
├── package.json
├── tsconfig.json
├── vitest.config.ts
├── README.md
├── src/
│   ├── index.ts           # Main exports
│   ├── types.ts           # LoadOptions, SDKConfig types
│   ├── loader.ts          # Core loader with retry/backoff
│   ├── singleflight.ts    # Global slot pattern
│   └── utils.ts           # withTimeout, sleep, jitter
└── src/
    ├── loader.test.ts     # Loader tests
    ├── singleflight.test.ts # Singleflight tests
    └── utils.test.ts      # Utils tests
```

### Implementation

**1. Create types.ts:**
```typescript
/**
 * TEAM-356: SDK loader types
 */

export interface LoadOptions {
  /** Package name to import (e.g., '@rbee/queen-rbee-sdk') */
  packageName: string
  
  /** Required exports to validate (e.g., ['Client', 'Monitor']) */
  requiredExports: string[]
  
  /** Timeout in milliseconds (default: 15000) */
  timeout?: number
  
  /** Max retry attempts (default: 3) */
  maxAttempts?: number
  
  /** Base backoff delay in ms (default: 300) */
  baseBackoffMs?: number
  
  /** Initialization argument (for WASM init) */
  initArg?: any
}

export interface SDKLoadResult<T> {
  sdk: T
  loadTime: number
  attempts: number
}

export interface GlobalSlot<T> {
  value?: SDKLoadResult<T>
  error?: Error
  promise?: Promise<SDKLoadResult<T>>
}
```

**2. Create utils.ts:**
```typescript
/**
 * TEAM-356: Utility functions for SDK loading
 */

/**
 * Sleep for specified milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Add random jitter to prevent thundering herd
 */
export function addJitter(baseMs: number, maxJitterMs: number): number {
  return baseMs + Math.random() * maxJitterMs
}

/**
 * Execute promise with timeout
 */
export function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  operation: string
): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(
        () => reject(new Error(`Timeout after ${timeoutMs}ms: ${operation}`)),
        timeoutMs
      )
    ),
  ])
}

/**
 * Calculate exponential backoff delay
 */
export function calculateBackoff(
  attempt: number,
  baseMs: number,
  maxJitterMs: number
): number {
  const exponential = 2 ** (attempt - 1) * baseMs
  return addJitter(exponential, maxJitterMs)
}
```

**3. Create singleflight.ts:**
```typescript
/**
 * TEAM-356: Singleflight pattern - ensure only one load at a time
 */

import type { GlobalSlot } from './types'

const GLOBAL_SLOTS = new Map<string, GlobalSlot<any>>()

/**
 * Get or create global slot for package
 */
export function getGlobalSlot<T>(packageName: string): GlobalSlot<T> {
  if (!GLOBAL_SLOTS.has(packageName)) {
    GLOBAL_SLOTS.set(packageName, {})
  }
  return GLOBAL_SLOTS.get(packageName)!
}

/**
 * Clear global slot (for testing)
 */
export function clearGlobalSlot(packageName: string): void {
  GLOBAL_SLOTS.delete(packageName)
}

/**
 * Clear all global slots (for testing)
 */
export function clearAllGlobalSlots(): void {
  GLOBAL_SLOTS.clear()
}
```

**4. Create loader.ts:**
```typescript
/**
 * TEAM-356: Core SDK loader with retry logic
 */

import type { LoadOptions, SDKLoadResult } from './types'
import { withTimeout, sleep, calculateBackoff } from './utils'
import { getGlobalSlot } from './singleflight'

/**
 * Load SDK with retry logic and timeout
 */
export async function loadSDK<T>(options: LoadOptions): Promise<SDKLoadResult<T>> {
  const {
    packageName,
    requiredExports,
    timeout = 15000,
    maxAttempts = 3,
    baseBackoffMs = 300,
    initArg,
  } = options

  // Environment guards
  if (typeof window === 'undefined') {
    throw new Error('SDK can only be loaded in browser environment')
  }

  if (typeof WebAssembly === 'undefined') {
    throw new Error('WebAssembly not supported in this browser')
  }

  const startTime = Date.now()
  let lastError: Error | undefined

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      // Dynamic import with timeout
      const mod = await withTimeout(
        import(/* @vite-ignore */ packageName),
        timeout,
        `SDK import (attempt ${attempt}/${maxAttempts})`
      )

      // Handle ESM/CJS shims (default export vs named exports)
      const wasmModule = (mod as any).default ?? mod

      // Initialize WASM if init function exists
      if (typeof wasmModule.init === 'function') {
        await withTimeout(
          wasmModule.init(initArg),
          timeout,
          'WASM initialization'
        )
      }

      // Validate required exports
      for (const exportName of requiredExports) {
        if (!wasmModule[exportName]) {
          throw new Error(`SDK missing required export: ${exportName}`)
        }
      }

      const loadTime = Date.now() - startTime
      return {
        sdk: wasmModule as T,
        loadTime,
        attempts: attempt,
      }
    } catch (err) {
      lastError = err as Error

      // Don't retry on last attempt
      if (attempt < maxAttempts) {
        const backoffMs = calculateBackoff(attempt, baseBackoffMs, baseBackoffMs)
        console.warn(
          `[sdk-loader] Attempt ${attempt}/${maxAttempts} failed, retrying in ${backoffMs}ms:`,
          lastError.message
        )
        await sleep(backoffMs)
      }
    }
  }

  throw lastError || new Error(`SDK load failed after ${maxAttempts} attempts`)
}

/**
 * Load SDK once (singleflight pattern)
 * Ensures only one load operation happens at a time per package
 */
export async function loadSDKOnce<T>(options: LoadOptions): Promise<SDKLoadResult<T>> {
  const slot = getGlobalSlot<T>(options.packageName)

  // Already loaded successfully
  if (slot.value) {
    return slot.value
  }

  // Previous load failed
  if (slot.error) {
    throw slot.error
  }

  // Load in progress
  if (slot.promise) {
    return slot.promise
  }

  // Start new load
  slot.promise = loadSDK<T>(options)
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

/**
 * Create SDK loader factory
 */
export function createSDKLoader<T>(defaultOptions: Omit<LoadOptions, 'initArg'>) {
  return {
    load: (initArg?: any) => loadSDK<T>({ ...defaultOptions, initArg }),
    loadOnce: (initArg?: any) => loadSDKOnce<T>({ ...defaultOptions, initArg }),
  }
}
```

**5. Create index.ts:**
```typescript
/**
 * TEAM-356: SDK Loader - WASM/SDK loading with retry logic
 */

export * from './types'
export * from './loader'
export * from './singleflight'
export * from './utils'
```

**6. Create package.json:**
```json
{
  "name": "@rbee/sdk-loader",
  "version": "0.1.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": "./dist/index.js"
  },
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "devDependencies": {
    "typescript": "^5.2.2",
    "vitest": "^3.2.4"
  }
}
```

**7. Create README.md:**
```markdown
# @rbee/sdk-loader

Generic WASM/SDK loader with exponential backoff, retry logic, and singleflight pattern.

## Features

- ✅ Exponential backoff with jitter
- ✅ Configurable retry attempts
- ✅ Timeout handling
- ✅ Singleflight pattern (one load at a time)
- ✅ Export validation
- ✅ Environment guards (SSR, WebAssembly support)

## Installation

```bash
pnpm add @rbee/sdk-loader
```

## Usage

### Basic Usage

```typescript
import { loadSDK } from '@rbee/sdk-loader'

const result = await loadSDK({
  packageName: '@rbee/queen-rbee-sdk',
  requiredExports: ['QueenClient', 'HeartbeatMonitor'],
  timeout: 15000,
  maxAttempts: 3,
})

const sdk = result.sdk
```

### Factory Pattern (Recommended)

```typescript
import { createSDKLoader } from '@rbee/sdk-loader'

const queenLoader = createSDKLoader({
  packageName: '@rbee/queen-rbee-sdk',
  requiredExports: ['QueenClient', 'HeartbeatMonitor', 'RhaiClient'],
})

// Load once (singleflight)
const { sdk } = await queenLoader.loadOnce()
```

### With WASM Initialization

```typescript
const { sdk } = await queenLoader.loadOnce({
  memory: new WebAssembly.Memory({ initial: 256 })
})
```

## API

See [types.ts](./src/types.ts) for full API documentation.
```

---

## Package 2: @rbee/react-hooks

### Purpose
Reusable React hooks for common patterns: async state management, SSE with health check, and CRUD operations.

### Files to Create

```
frontend/packages/react-hooks/
├── package.json
├── tsconfig.json
├── vitest.config.ts
├── README.md
├── src/
│   ├── index.ts              # Main exports
│   ├── useAsyncState.ts      # Async data loading hook
│   ├── useSSEWithHealthCheck.ts  # SSE with health check
│   └── useCRUD.ts            # CRUD operations hook (optional)
└── src/
    ├── useAsyncState.test.ts
    ├── useSSEWithHealthCheck.test.ts
    └── useCRUD.test.ts
```

### Implementation

**1. Create useAsyncState.ts:**
```typescript
/**
 * TEAM-356: Async state management hook
 */

import { useState, useEffect, useRef, useCallback, type DependencyList } from 'react'

export interface AsyncStateOptions {
  /** Skip initial load (default: false) */
  skip?: boolean
  
  /** Callback on success */
  onSuccess?: (data: any) => void
  
  /** Callback on error */
  onError?: (error: Error) => void
}

export interface AsyncStateResult<T> {
  data: T | null
  loading: boolean
  error: Error | null
  refetch: () => void
}

/**
 * Hook for async data loading with loading/error states
 * 
 * @example
 * const { data, loading, error, refetch } = useAsyncState(
 *   async () => {
 *     const response = await fetch('/api/data')
 *     return response.json()
 *   },
 *   []
 * )
 */
export function useAsyncState<T>(
  asyncFn: () => Promise<T>,
  deps: DependencyList,
  options: AsyncStateOptions = {}
): AsyncStateResult<T> {
  const { skip = false, onSuccess, onError } = options
  
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(!skip)
  const [error, setError] = useState<Error | null>(null)
  const mountedRef = useRef(true)

  const execute = useCallback(async () => {
    if (skip) return

    setLoading(true)
    setError(null)

    try {
      const result = await asyncFn()
      
      if (mountedRef.current) {
        setData(result)
        setLoading(false)
        onSuccess?.(result)
      }
    } catch (err) {
      const error = err as Error
      
      if (mountedRef.current) {
        setError(error)
        setLoading(false)
        onError?.(error)
      }
    }
  }, [asyncFn, skip, onSuccess, onError])

  useEffect(() => {
    mountedRef.current = true
    execute()

    return () => {
      mountedRef.current = false
    }
  }, [...deps, execute])

  const refetch = useCallback(() => {
    execute()
  }, [execute])

  return { data, loading, error, refetch }
}
```

**2. Create useSSEWithHealthCheck.ts:**
```typescript
/**
 * TEAM-356: SSE connection with health check
 * Prevents CORS errors by checking health before connecting
 */

import { useState, useEffect, useRef } from 'react'

export interface Monitor<T> {
  checkHealth: () => Promise<boolean>
  start: (onData: (data: T) => void) => void
  stop: () => void
}

export interface SSEHealthCheckOptions {
  /** Auto-retry on connection failure (default: true) */
  autoRetry?: boolean
  
  /** Retry delay in ms (default: 5000) */
  retryDelayMs?: number
  
  /** Max retry attempts (default: 3) */
  maxRetries?: number
}

export interface SSEHealthCheckResult<T> {
  data: T | null
  connected: boolean
  loading: boolean
  error: Error | null
  retry: () => void
}

/**
 * Hook for SSE connection with health check
 * 
 * @example
 * const { data, connected, error } = useSSEWithHealthCheck(
 *   (baseUrl) => new sdk.HeartbeatMonitor(baseUrl),
 *   'http://localhost:7833'
 * )
 */
export function useSSEWithHealthCheck<T>(
  createMonitor: (baseUrl: string) => Monitor<T>,
  baseUrl: string,
  options: SSEHealthCheckOptions = {}
): SSEHealthCheckResult<T> {
  const {
    autoRetry = true,
    retryDelayMs = 5000,
    maxRetries = 3,
  } = options

  const [data, setData] = useState<T | null>(null)
  const [connected, setConnected] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  
  const mountedRef = useRef(true)
  const monitorRef = useRef<Monitor<T> | null>(null)
  const retriesRef = useRef(0)

  const startMonitoring = async () => {
    if (!mountedRef.current) return

    setLoading(true)
    setError(null)

    try {
      const monitor = createMonitor(baseUrl)
      monitorRef.current = monitor

      // Check health before starting SSE
      const isHealthy = await monitor.checkHealth()

      if (!mountedRef.current) return

      if (!isHealthy) {
        throw new Error('Service is offline')
      }

      // Start SSE connection
      monitor.start((snapshot: T) => {
        if (!mountedRef.current) return
        setData(snapshot)
        setConnected(true)
        setError(null)
        setLoading(false)
        retriesRef.current = 0 // Reset retry count on success
      })
    } catch (err) {
      const error = err as Error

      if (!mountedRef.current) return

      setError(error)
      setConnected(false)
      setLoading(false)

      // Auto-retry if enabled
      if (autoRetry && retriesRef.current < maxRetries) {
        retriesRef.current++
        console.warn(
          `[useSSEWithHealthCheck] Retry ${retriesRef.current}/${maxRetries} in ${retryDelayMs}ms`
        )
        setTimeout(startMonitoring, retryDelayMs)
      }
    }
  }

  useEffect(() => {
    mountedRef.current = true
    retriesRef.current = 0
    startMonitoring()

    return () => {
      mountedRef.current = false
      monitorRef.current?.stop()
    }
  }, [baseUrl])

  const retry = () => {
    retriesRef.current = 0
    startMonitoring()
  }

  return { data, connected, loading, error, retry }
}
```

**3. Create index.ts:**
```typescript
/**
 * TEAM-356: React Hooks - Reusable hooks for common patterns
 */

export * from './useAsyncState'
export * from './useSSEWithHealthCheck'
// export * from './useCRUD'  // Optional
```

**4. Create package.json:**
```json
{
  "name": "@rbee/react-hooks",
  "version": "0.1.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": "./dist/index.js"
  },
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "peerDependencies": {
    "react": "^18.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "react": "^18.2.0",
    "typescript": "^5.2.2",
    "vitest": "^3.2.4",
    "@testing-library/react": "^14.0.0",
    "@testing-library/react-hooks": "^8.0.1"
  }
}
```

**5. Create README.md:**
```markdown
# @rbee/react-hooks

Reusable React hooks for common patterns in rbee UIs.

## Features

- ✅ `useAsyncState` - Async data loading with loading/error states
- ✅ `useSSEWithHealthCheck` - SSE connection with health check
- ✅ Automatic cleanup on unmount
- ✅ TypeScript support

## Installation

```bash
pnpm add @rbee/react-hooks
```

## Hooks

### useAsyncState

Load async data with loading/error states.

```typescript
import { useAsyncState } from '@rbee/react-hooks'

function MyComponent() {
  const { data, loading, error, refetch } = useAsyncState(
    async () => {
      const response = await fetch('/api/data')
      return response.json()
    },
    [] // dependencies
  )

  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error.message}</div>
  return <div>{JSON.stringify(data)}</div>
}
```

### useSSEWithHealthCheck

Connect to SSE stream with health check.

```typescript
import { useSSEWithHealthCheck } from '@rbee/react-hooks'

function HeartbeatMonitor() {
  const { data, connected, error } = useSSEWithHealthCheck(
    (baseUrl) => new sdk.HeartbeatMonitor(baseUrl),
    'http://localhost:7833'
  )

  return (
    <div>
      Status: {connected ? 'Connected' : 'Disconnected'}
      {data && <pre>{JSON.stringify(data, null, 2)}</pre>}
    </div>
  )
}
```

## API

See individual hook files for full API documentation.
```

---

## Migration: Update Queen UI

After creating packages, migrate Queen UI to use them.

### Step 1: Update Queen Dependencies

```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
pnpm add @rbee/sdk-loader @rbee/react-hooks
```

### Step 2: Migrate SDK Loader

**Before:** `src/loader.ts` (120 lines)
```typescript
// Custom loader with retry logic
export async function loadSDK() { ... }
```

**After:** Use `@rbee/sdk-loader`
```typescript
import { createSDKLoader } from '@rbee/sdk-loader'

export const queenSDKLoader = createSDKLoader({
  packageName: '@rbee/queen-rbee-sdk',
  requiredExports: ['QueenClient', 'HeartbeatMonitor', 'RhaiClient'],
  timeout: 15000,
  maxAttempts: 3,
})

export const loadSDK = queenSDKLoader.loadOnce
```

**Lines removed:** ~100 lines

### Step 3: Migrate useHeartbeat Hook

**Before:** `src/hooks/useHeartbeat.ts` (90 lines)
```typescript
export function useHeartbeat(baseUrl = 'http://localhost:7833') {
  const [data, setData] = useState(null)
  const [connected, setConnected] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const mounted = useRef(true)
  
  // 40 lines of health check + SSE logic
  // 20 lines of cleanup
  // 10 lines of error handling
}
```

**After:** Use `@rbee/react-hooks`
```typescript
import { useSSEWithHealthCheck } from '@rbee/react-hooks'
import { getServiceUrl } from '@rbee/shared-config'

export function useHeartbeat(
  baseUrl = getServiceUrl('queen', 'prod')  // Use shared config!
) {
  return useSSEWithHealthCheck(
    (url) => new sdk.HeartbeatMonitor(url),
    baseUrl
  )
}
```

**Lines removed:** ~60 lines

### Step 4: Migrate useRhaiScripts Hook

**Before:** `src/hooks/useRhaiScripts.ts` (250 lines)
```typescript
export function useRhaiScripts(baseUrl = 'http://localhost:7833') {
  const [scripts, setScripts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const mounted = useRef(true)
  
  // 80 lines of list/load logic
  // 60 lines of save logic
  // 40 lines of delete logic
  // 30 lines of select logic
}
```

**After:** Use `@rbee/react-hooks`
```typescript
import { useAsyncState } from '@rbee/react-hooks'
import { getServiceUrl } from '@rbee/shared-config'

export function useRhaiScripts(
  baseUrl = getServiceUrl('queen', 'prod')
) {
  const { data: sdk } = useSDK()
  
  const { data: scripts, loading, error, refetch } = useAsyncState(
    async () => {
      if (!sdk) return []
      const client = new sdk.RhaiClient(baseUrl)
      const result = await client.listScripts()
      return JSON.parse(JSON.stringify(result))
    },
    [sdk, baseUrl]
  )
  
  // Keep save/delete/select functions (30 lines)
  
  return { scripts, loading, error, refetch, save, delete, select }
}
```

**Lines removed:** ~150 lines

---

## Testing Requirements

### @rbee/sdk-loader Tests (~40 tests)

**Files:**
- `src/loader.test.ts` - 20 tests
- `src/singleflight.test.ts` - 10 tests
- `src/utils.test.ts` - 10 tests

**Coverage:**
- Retry logic with exponential backoff
- Timeout handling
- Export validation
- Singleflight pattern
- Environment guards
- Error handling

### @rbee/react-hooks Tests (~30 tests)

**Files:**
- `src/useAsyncState.test.ts` - 15 tests
- `src/useSSEWithHealthCheck.test.ts` - 15 tests

**Coverage:**
- Async data loading
- Loading/error states
- Cleanup on unmount
- Health check before SSE
- Auto-retry logic
- Refetch functionality

**Use @testing-library/react-hooks for testing.**

---

## Verification Checklist

### Package Creation
- [ ] `@rbee/sdk-loader` created with all files
- [ ] `@rbee/react-hooks` created with all files
- [ ] Both packages added to `pnpm-workspace.yaml`
- [ ] `pnpm install` runs successfully
- [ ] Both packages build without errors
- [ ] All tests written and passing

### Queen Migration
- [ ] Queen depends on `@rbee/sdk-loader`
- [ ] Queen depends on `@rbee/react-hooks`
- [ ] `loader.ts` migrated to use `@rbee/sdk-loader`
- [ ] `useHeartbeat.ts` migrated to use `@rbee/react-hooks`
- [ ] `useRhaiScripts.ts` migrated to use `@rbee/react-hooks`
- [ ] All hardcoded URLs replaced with `@rbee/shared-config`
- [ ] Queen UI builds without errors
- [ ] Queen UI runs in dev mode
- [ ] Queen UI runs in prod mode
- [ ] Narration still works
- [ ] Hot reload still works

### Code Quality
- [ ] All files have TEAM-356 signatures
- [ ] No TODO markers
- [ ] Comprehensive JSDoc comments
- [ ] README for each package
- [ ] TypeScript strict mode enabled
- [ ] No `any` types (except controlled cases)

### Documentation
- [ ] Migration guide created
- [ ] Before/after code examples
- [ ] Lines saved documented
- [ ] Handoff document (≤2 pages)

---

## Success Criteria

✅ Both packages created and tested  
✅ Queen UI migrated successfully  
✅ ~300-400 lines removed from Queen  
✅ All tests passing  
✅ No regression in functionality  
✅ Pattern validated for Hive/Worker  
✅ Documentation complete

---

## Estimated Effort

**Package Creation:** 4-6 hours
- `@rbee/sdk-loader`: 2-3 hours
- `@rbee/react-hooks`: 2-3 hours

**Queen Migration:** 2-3 hours
- Migrate loader: 30 minutes
- Migrate hooks: 1-2 hours
- Testing: 1 hour

**Total:** 6-9 hours

---

## ROI Analysis

**Investment:**
- Create packages: 500 lines
- Write tests: 70 tests
- Time: 6-9 hours

**Savings:**
- Queen: ~300 lines removed
- Hive (future): ~400 lines prevented
- Worker (future): ~400 lines prevented
- **Total: 1,100 lines saved**

**ROI:** 2.2x (1,100 saved / 500 written)

---

**TEAM-356: Extract reusable patterns and prevent duplication!** 🎯
