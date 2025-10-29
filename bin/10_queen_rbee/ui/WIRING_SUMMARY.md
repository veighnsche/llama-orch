# Queen UI Wiring Summary

**Date:** Oct 29, 2025  
**Status:** ✅ COMPLETE

---

## Architecture

```
queen-rbee-sdk (WASM)
    ↓
queen-rbee-react (React hooks)
    ↓
queen-rbee-ui (App)
```

---

## Changes Made

### 1. Created `useHeartbeat` Hook in React Package

**File:** `packages/queen-rbee-react/src/hooks/useHeartbeat.ts`

```tsx
export function useHeartbeat(baseUrl: string = 'http://localhost:7833'): UseHeartbeatResult {
  const { sdk, loading: sdkLoading, error: sdkError } = useRbeeSDK()
  const [data, setData] = useState<HeartbeatData | null>(null)
  const [connected, setConnected] = useState(false)
  
  useEffect(() => {
    if (!sdk) return
    
    const monitor = new sdk.HeartbeatMonitor(baseUrl)
    monitor.start((snapshot: any) => {
      setData(snapshot)
      setConnected(true)
    })
    
    return () => monitor.stop()
  }, [sdk, baseUrl])
  
  return { data, connected, loading, error }
}
```

### 2. Updated React Package Exports

**File:** `packages/queen-rbee-react/src/index.ts`

```tsx
export { useRbeeSDK, useRbeeSDKSuspense, useHeartbeat } from './hooks'
export type { HeartbeatData, UseHeartbeatResult } from './hooks'
```

### 3. Updated Dashboard to Use React Package

**File:** `app/src/pages/DashboardPage.tsx`

**Before:**
```tsx
import { useHeartbeat } from '../hooks/useHeartbeat'
import { useRbeeStore } from '../stores/rbeeStore'

const { connected, loading, error } = useHeartbeat()
const { hives, workersOnline } = useRbeeStore()
```

**After:**
```tsx
import { useHeartbeat } from '@rbee/queen-rbee-react'

const { data, connected, loading, error } = useHeartbeat('http://localhost:7833')
const workersOnline = data?.workers_online || 0
```

---

## Data Flow

1. **SDK Layer** (`queen-rbee-sdk`)
   - WASM bindings to Rust
   - `HeartbeatMonitor` class
   - Connects to `/v1/heartbeats/stream` SSE endpoint

2. **React Layer** (`queen-rbee-react`)
   - `useHeartbeat()` hook
   - Manages SDK lifecycle
   - Provides typed data to components

3. **App Layer** (`app`)
   - `DashboardPage` component
   - Uses `useHeartbeat()` from react package
   - Displays heartbeat data + RHAI IDE

---

## Package Structure

```
bin/10_queen_rbee/ui/
├── packages/
│   ├── queen-rbee-sdk/          # WASM SDK
│   │   └── src/
│   │       ├── heartbeat.rs     # Rust HeartbeatMonitor
│   │       └── lib.rs
│   └── queen-rbee-react/        # React hooks
│       └── src/
│           ├── hooks/
│           │   ├── useRbeeSDK.ts
│           │   └── useHeartbeat.ts  ← NEW
│           └── index.ts
└── app/                         # Vite app
    └── src/
        └── pages/
            └── DashboardPage.tsx    ← UPDATED
```

---

## Benefits

✅ **Clean separation of concerns**
- SDK handles WASM/Rust bindings
- React package handles React-specific logic
- App just consumes the hook

✅ **Reusable**
- `useHeartbeat()` can be used in any React component
- No need to duplicate heartbeat logic

✅ **Type-safe**
- Full TypeScript types from SDK → React → App
- `HeartbeatData` interface exported

✅ **Testable**
- Each layer can be tested independently
- Mock SDK in React tests
- Mock React hook in App tests

---

## Usage

```tsx
import { useHeartbeat } from '@rbee/queen-rbee-react'

function MyComponent() {
  const { data, connected, loading, error } = useHeartbeat()
  
  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error.message}</div>
  
  return (
    <div>
      <p>Workers Online: {data?.workers_online}</p>
      <p>Status: {connected ? 'Connected' : 'Disconnected'}</p>
    </div>
  )
}
```

---

## Summary

The Queen UI is now properly wired through the package structure:
- ✅ SDK provides WASM bindings
- ✅ React package provides hooks
- ✅ App consumes hooks

**Clean, type-safe, and reusable!** 🎉
