# TEAM-291: Hardened WASM SDK Bootstrap

**Status:** ✅ COMPLETE

**Mission:** Implement production-grade WASM SDK loading with singleflight, HMR safety, retries, timeouts, and Suspense support.

## Implementation Summary

Replaced the basic `useRbeeSDK` hook with a hardened implementation featuring:

1. **Singleflight loading** - Multiple concurrent calls share one load operation
2. **HMR-safe global singleton** - Survives hot module reloads via `globalThis.__rbeeSDKInit_v1__`
3. **Retry with jittered exponential backoff** - 3 attempts by default, configurable
4. **Per-attempt timeouts** - 15s default, prevents hanging on slow networks
5. **SSR-safe** - Explicit browser-only guard with clear error message
6. **Flexible init() handling** - Supports sync, async, or absent init functions
7. **Export validation** - Ensures all required exports are present
8. **Suspense variant** - `useRbeeSDKSuspense` for Suspense boundaries
9. **Full TypeScript support** - Proper types, no `any` leaks

## Architecture

### Global Singleton (HMR-Safe)

```typescript
// Slot on globalThis survives HMR
globalThis.__rbeeSDKInit_v1__ = {
  promise?: Promise<{ sdk: RbeeSDK }>,  // In-flight load
  value?: { sdk: RbeeSDK },              // Cached result
  error?: Error                          // Cached error
}
```

**Benefits:**
- Multiple components mounting simultaneously → one load
- HMR doesn't trigger re-initialization
- Singleflight: concurrent calls wait for same promise

### Retry Logic

```
Attempt 1: Load → timeout 15s → fail
  ↓ backoff: 300ms + jitter
Attempt 2: Load → timeout 15s → fail
  ↓ backoff: 600ms + jitter
Attempt 3: Load → timeout 15s → fail
  ↓ throw error
```

**Backoff formula:** `delay = 2^(attempt-1) * baseBackoffMs + jitter`
- Jitter: `[0, baseBackoffMs)` to avoid thundering herd

### Environment Guards

1. **SSR check:**
   ```typescript
   if (typeof window === 'undefined') {
     throw Error("rbee SDK can only be initialized in the browser (client component).");
   }
   ```

2. **WebAssembly check:**
   ```typescript
   if (typeof WebAssembly !== 'object' || typeof WebAssembly.instantiate !== 'function') {
     throw Error("WebAssembly is not supported in this environment.");
   }
   ```

### Init Flexibility

Handles all three cases:
```typescript
// Case 1: No init function
if (typeof wasmModule.init !== 'function') { /* skip */ }

// Case 2: Sync init
wasmModule.init(arg); // returns void

// Case 3: Async init
await wasmModule.init(arg); // returns Promise
```

## API

### Types

```typescript
export interface RbeeSDK {
  RbeeClient: typeof RbeeClient;
  HeartbeatMonitor: typeof HeartbeatMonitor;
  OperationBuilder: typeof OperationBuilder;
}

export type LoadOptions = {
  timeoutMs?: number;       // default: 15000
  maxAttempts?: number;     // default: 3
  baseBackoffMs?: number;   // default: 300
  initArg?: unknown;        // forwarded to init()
  onReady?: (sdk: RbeeSDK) => void;
};
```

### useRbeeSDK (Standard Hook)

```typescript
const { sdk, loading, error } = useRbeeSDK(options?);
```

**Behavior:**
- Returns `{ sdk: null, loading: true, error: null }` initially
- On success: `{ sdk: RbeeSDK, loading: false, error: null }`
- On error: `{ sdk: null, loading: false, error: Error }`
- Safe for concurrent mounts (singleflight)
- Survives HMR (global singleton)

**Example:**
```tsx
function Dashboard() {
  const { sdk, loading, error } = useRbeeSDK({
    timeoutMs: 20000,
    onReady: (s) => console.log('SDK ready!'),
  });

  if (loading) return <Spinner />;
  if (error) return <ErrorBanner error={error} />;
  
  const client = new sdk.RbeeClient('http://localhost:8500');
  // ...
}
```

### useRbeeSDKSuspense (Suspense Hook)

```typescript
const sdk = useRbeeSDKSuspense(options?);
```

**Behavior:**
- Throws promise until ready (Suspense catches it)
- Returns `RbeeSDK` when ready
- Throws error on failure (ErrorBoundary catches it)
- Same singleton as `useRbeeSDK`

**Example:**
```tsx
function App() {
  return (
    <Suspense fallback={<Spinner />}>
      <Dashboard />
    </Suspense>
  );
}

function Dashboard() {
  const sdk = useRbeeSDKSuspense();
  const client = new sdk.RbeeClient('http://localhost:8500');
  return <div>Ready!</div>;
}
```

## Implementation Details

### Core Functions

1. **`getGlobalSlot()`** - Creates/returns global singleton
2. **`withTimeout(p, ms, label)`** - Races promise with timeout
3. **`sleep(ms)`** - Async delay for backoff
4. **`actuallyLoadSDK(opts)`** - Core loader with retries
5. **`loadSDKOnce(options)`** - Singleflight wrapper

### Loader Flow

```
loadSDKOnce()
  ↓
Check slot.value → return immediately
  ↓
Check slot.error → reject immediately
  ↓
Check slot.promise → return it (singleflight)
  ↓
Create slot.promise = actuallyLoadSDK()
  ↓
For each attempt (1..maxAttempts):
  ↓
  Import @rbee/sdk with timeout
  ↓
  Handle ESM/CJS default shims
  ↓
  Call init(arg) if present (sync or async)
  ↓
  Validate exports (RbeeClient, HeartbeatMonitor, OperationBuilder)
  ↓
  On success: cache in slot.value, call onReady, return
  ↓
  On failure: backoff, retry
  ↓
Final failure: cache in slot.error, throw
```

### Hook Implementation

**useRbeeSDK:**
- Uses `useState` for sdk, loading, error
- Uses `useRef` to track mount status (avoid state updates after unmount)
- `useEffect` with empty deps (loads once per mount)
- Options not in deps (intentional: load once, not on option changes)

**useRbeeSDKSuspense:**
- No state, no effects
- Reads from global slot
- Throws promise or error for React to catch

## Error Messages

All error messages are user-facing and actionable:

1. **SSR usage:**
   ```
   "rbee SDK can only be initialized in the browser (client component)."
   ```

2. **No WebAssembly:**
   ```
   "WebAssembly is not supported in this environment."
   ```

3. **Missing exports:**
   ```
   "SDK exports missing: expected { RbeeClient, HeartbeatMonitor, OperationBuilder }."
   ```

4. **Timeout:**
   ```
   "SDK import (attempt 1/3) timed out after 15000ms"
   "SDK init (attempt 2/3) timed out after 15000ms"
   ```

5. **All retries exhausted:**
   ```
   "SDK load failed after all retry attempts."
   ```

## Acceptance Checklist

✅ **Singleflight:** Multiple components mounting simultaneously do one import/init  
✅ **HMR-safe:** slot.value persists across hot reloads  
✅ **Export validation:** Missing exports → deterministic error  
✅ **Init flexibility:** Handles absent, sync, and async init()  
✅ **Timeouts:** Per-attempt timeout fires when import/init hangs  
✅ **Retries:** Exponential backoff with jitter, maxAttempts configurable  
✅ **Suspense:** useRbeeSDKSuspense throws promise until ready  
✅ **SSR-safe:** Explicit browser-only error on SSR usage  
✅ **Type safety:** RbeeSDK, LoadOptions exported; no internal leaks  

## Files Changed

1. **MODIFIED:** `frontend/packages/rbee-react/src/useRbeeSDK.ts` (258 LOC)
   - Complete rewrite with hardened implementation
   - Added LoadOptions type
   - Added useRbeeSDKSuspense hook
   - Added global singleton with HMR safety
   - Added retry logic with backoff
   - Added timeout helpers
   - Added environment guards
   - Added export validation

2. **MODIFIED:** `frontend/packages/rbee-react/src/index.ts`
   - Exported useRbeeSDKSuspense
   - Exported RbeeSDK and LoadOptions types

3. **REBUILT:** `frontend/packages/rbee-react/dist/`
   - TypeScript compilation successful
   - All types properly exported

## Usage Patterns

### Basic Usage
```tsx
const { sdk, loading, error } = useRbeeSDK();
```

### With Options
```tsx
const { sdk, loading, error } = useRbeeSDK({
  timeoutMs: 20000,
  maxAttempts: 5,
  baseBackoffMs: 500,
  onReady: (sdk) => {
    console.log('SDK ready, prewarming...');
    // Optional: prewarm connections
  },
});
```

### Suspense Pattern
```tsx
<Suspense fallback={<LoadingSpinner />}>
  <ErrorBoundary fallback={<ErrorScreen />}>
    <DashboardWithSDK />
  </ErrorBoundary>
</Suspense>

function DashboardWithSDK() {
  const sdk = useRbeeSDKSuspense();
  // SDK is guaranteed to be loaded here
}
```

### Mixed Pattern
```tsx
// Some components use Suspense
<Suspense fallback={<Spinner />}>
  <ComponentA /> {/* uses useRbeeSDKSuspense */}
</Suspense>

// Others use standard hook
<ComponentB /> {/* uses useRbeeSDK */}

// Both share the same singleton!
```

## Performance Characteristics

**Cold start (first load):**
- Best case: ~100-500ms (fast network, small WASM)
- Typical: ~1-3s (normal network)
- Worst case: 3 attempts × 15s timeout = 45s max

**Subsequent mounts:**
- Instant (reads from slot.value)

**HMR:**
- No re-initialization (slot persists)

**Concurrent mounts:**
- N components → 1 load operation (singleflight)

## Testing Scenarios

### Scenario 1: Happy Path
```
Component mounts → loadSDKOnce() → import succeeds → init() succeeds → 
exports valid → slot.value cached → component renders
```

### Scenario 2: Slow Network
```
Component mounts → loadSDKOnce() → import takes 10s → succeeds → 
slot.value cached → component renders
```

### Scenario 3: Timeout
```
Attempt 1: import hangs → timeout at 15s → retry
Attempt 2: import succeeds → init() succeeds → slot.value cached
```

### Scenario 4: All Retries Fail
```
Attempt 1: timeout → backoff 300ms
Attempt 2: timeout → backoff 600ms
Attempt 3: timeout → slot.error cached → component shows error
```

### Scenario 5: Concurrent Mounts
```
Component A mounts → starts load → slot.promise created
Component B mounts → sees slot.promise → waits for same promise
Component C mounts → sees slot.promise → waits for same promise
Load completes → all three components update with same SDK
```

### Scenario 6: HMR
```
Component renders with SDK → HMR triggers → component re-mounts →
sees slot.value → instant render (no re-load)
```

### Scenario 7: SSR Attempt
```
Component renders on server → loadSDKOnce() → typeof window check →
throws "browser only" error → SSR fails gracefully
```

## Engineering Rules Compliance

- ✅ No TODO markers
- ✅ Complete implementation
- ✅ Full TypeScript types
- ✅ Minimal, focused solution
- ✅ No external dependencies added
- ✅ Team signature (TEAM-291)
- ✅ Self-documenting code with JSDoc

---

**TEAM-291 COMPLETE** - Production-grade WASM SDK bootstrap implemented.
