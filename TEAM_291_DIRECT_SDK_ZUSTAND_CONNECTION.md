# TEAM-291: Direct SDK-Zustand Connection

**Status:** ✅ COMPLETE

**Mission:** Connect rbee SDK directly to zustand store, removing intermediate hook layer and console logs.

## Problem

**Before:**
```
SDK → useHeartbeat (hook with local state) → zustand store → components
```

**Issues:**
- Unnecessary intermediate layer
- Console logs everywhere (debugging noise)
- Hook manages state that zustand should own
- Duplicate state management
- Complex data flow

## Solution

**After:**
```
SDK → zustand store → components
```

**Benefits:**
- Direct connection (no intermediate layer)
- Single source of truth (zustand owns all state)
- Cleaner code (no console logs)
- Simpler data flow
- Better performance (fewer re-renders)

## Architecture

### Zustand Store (State Owner)

The store now:
1. **Owns the HeartbeatMonitor instance**
2. **Manages the SSE connection lifecycle**
3. **Receives heartbeat updates directly**
4. **Provides actions to start/stop monitoring**

```typescript
interface RbeeState {
  // ... state fields
  monitor: HeartbeatMonitor | null;  // ← Owns the monitor
  
  // Actions
  startMonitoring: (monitor, baseUrl) => void;  // ← Start SSE
  stopMonitoring: () => void;                    // ← Stop SSE
}
```

### Hook (Initialization Only)

The hook now only:
1. **Waits for SDK to load**
2. **Creates HeartbeatMonitor instance**
3. **Passes it to zustand**
4. **Returns loading/error state**

```typescript
export function useHeartbeat(baseUrl) {
  const { sdk, loading, error } = useRbeeSDK();
  const { startMonitoring, stopMonitoring } = useRbeeStore();

  useEffect(() => {
    if (!sdk) return;
    const monitor = new sdk.HeartbeatMonitor(baseUrl);
    startMonitoring(monitor, baseUrl);
    return () => stopMonitoring();
  }, [sdk, baseUrl]);

  return { connected, loading, error };
}
```

## Implementation Details

### Store Changes

#### Added Fields
```typescript
monitor: HeartbeatMonitor | null;  // Owns the monitor instance
```

#### Added Actions

**1. startMonitoring(monitor, baseUrl)**
- Stops existing monitor if any
- Starts new monitor with callback
- Callback updates store directly with heartbeat data
- Stores monitor instance

**2. stopMonitoring()**
- Stops monitor if exists
- Clears monitor instance

**3. resetState() (updated)**
- Stops monitor before resetting
- Ensures cleanup

#### Heartbeat Callback (Inside Store)
```typescript
monitorInstance.start((snapshot: HeartbeatSnapshot) => {
  // Direct store update - no intermediate layer
  set({
    queen: { connected: true, lastUpdate: snapshot.timestamp, error: null },
    hives: snapshot.hive_ids.map(id => ({ id, status: 'online', lastSeen: snapshot.timestamp })),
    hivesOnline: snapshot.hives_online,
    hivesAvailable: snapshot.hives_available,
    workersOnline: snapshot.workers_online,
    workersAvailable: snapshot.workers_available,
    workerIds: snapshot.worker_ids,
    lastHeartbeat: snapshot,
  });
});
```

### Hook Changes

**Before (83 LOC with console logs):**
- Created monitor
- Managed monitor in useRef
- Called updateFromHeartbeat
- Called setQueenConnected
- Called setQueenError
- Manual cleanup
- Console logs everywhere

**After (41 LOC, no console logs):**
- Creates monitor
- Calls startMonitoring
- Calls stopMonitoring on cleanup
- Returns loading/error state
- Clean, minimal code

## Data Flow

### Before (Complex)
```
1. SDK loads
2. useHeartbeat creates monitor
3. Monitor fires callback
4. Callback calls updateFromHeartbeat(snapshot)
5. updateFromHeartbeat updates zustand
6. Components read from zustand
```

### After (Simple)
```
1. SDK loads
2. useHeartbeat creates monitor
3. useHeartbeat calls startMonitoring(monitor)
4. Monitor fires callback (inside store)
5. Callback updates zustand directly
6. Components read from zustand
```

## Removed Console Logs

Removed all debugging console logs:
- `🐝 [useHeartbeat] Effect running, sdk: ...`
- `🐝 [useHeartbeat] SDK not ready yet, waiting...`
- `🐝 [useHeartbeat] Creating HeartbeatMonitor with baseUrl: ...`
- `🐝 [useHeartbeat] HeartbeatMonitor created: ...`
- `🐝 [useHeartbeat] Starting monitor...`
- `🐝 [useHeartbeat] CALLBACK FIRED! Received snapshot: ...`
- `🐝 [useHeartbeat] Monitor.start() called`
- `🐝 [useHeartbeat] Connection check after 1s, isConnected: ...`
- `🐝 [useHeartbeat] ERROR starting monitor: ...`
- `🐝 [useHeartbeat] Cleanup: stopping monitor`

**Result:** Clean, production-ready code without debugging noise.

## Cleaned Up Files

### Deleted
1. ✅ `/home/vince/Projects/llama-orch/frontend/apps/web-ui/src/hooks/useRbeeSDK.ts`
   - Moved to `@rbee/react` package

2. ✅ `/home/vince/Projects/llama-orch/frontend/packages/rbee-react/src/useRbeeSDK.ts`
   - Replaced by modular structure

### Modified
1. **`src/stores/rbeeStore.ts`**
   - Added `monitor` field
   - Added `startMonitoring()` action
   - Added `stopMonitoring()` action
   - Updated `resetState()` to cleanup monitor
   - Heartbeat callback now inside store

2. **`src/hooks/useHeartbeat.ts`**
   - Removed all console logs (10 removed)
   - Simplified from 83 LOC → 41 LOC (50% reduction)
   - Removed intermediate state management
   - Now just initializes and delegates to store

## Benefits

### 1. Simpler Architecture
- One less layer of indirection
- Clear ownership (zustand owns monitor)
- Easier to understand

### 2. Better Performance
- Fewer function calls
- Direct store updates
- No intermediate state

### 3. Cleaner Code
- No console logs
- 50% less code in hook
- Single responsibility

### 4. Easier Testing
```typescript
// Test store directly
const { startMonitoring } = useRbeeStore.getState();
const mockMonitor = { start: jest.fn(), stop: jest.fn() };
startMonitoring(mockMonitor, 'http://localhost:8500');
expect(mockMonitor.start).toHaveBeenCalled();
```

### 5. Better Lifecycle Management
- Store owns monitor lifecycle
- Automatic cleanup on unmount
- No memory leaks

## Usage (Unchanged)

Components see no difference:

```tsx
function Dashboard() {
  const { connected, loading, error } = useHeartbeat();
  const { hives, workers } = useRbeeStore();
  
  if (loading) return <Spinner />;
  if (error) return <Error error={error} />;
  
  return <div>Connected: {connected ? 'Yes' : 'No'}</div>;
}
```

## State Management Pattern

This follows the **"Smart Store, Dumb Hook"** pattern:

**Smart Store:**
- Owns all state
- Owns side effects (SSE connection)
- Manages lifecycle
- Provides actions

**Dumb Hook:**
- Initializes only
- Delegates to store
- Returns derived state
- No business logic

## Comparison

### Before
```typescript
// Hook: 83 LOC, manages state, console logs
useEffect(() => {
  console.log('Effect running...');
  if (!sdk) {
    console.log('SDK not ready...');
    return;
  }
  console.log('Creating monitor...');
  const monitor = new sdk.HeartbeatMonitor(baseUrl);
  console.log('Monitor created:', monitor);
  
  monitor.start((snapshot) => {
    console.log('Callback fired:', snapshot);
    updateFromHeartbeat(snapshot);  // ← Indirect
  });
  
  setTimeout(() => {
    console.log('Checking connection...');
    if (monitor.isConnected()) {
      setQueenConnected(true);  // ← Manual
    }
  }, 1000);
  
  return () => {
    console.log('Cleanup...');
    monitor.stop();
  };
}, [sdk, baseUrl, updateFromHeartbeat, setQueenConnected]);
```

### After
```typescript
// Hook: 41 LOC, initialization only, no logs
useEffect(() => {
  if (!sdk) return;
  
  try {
    const monitor = new sdk.HeartbeatMonitor(baseUrl);
    startMonitoring(monitor, baseUrl);  // ← Direct to store
  } catch (err) {
    setQueenError((err as Error).message);
  }
  
  return () => stopMonitoring();
}, [sdk, baseUrl, startMonitoring, stopMonitoring, setQueenError]);
```

## Files Changed

1. **MODIFIED:** `frontend/apps/web-ui/src/stores/rbeeStore.ts`
   - Added `monitor: HeartbeatMonitor | null` field
   - Added `startMonitoring()` action (27 LOC)
   - Added `stopMonitoring()` action (7 LOC)
   - Updated `resetState()` to cleanup monitor
   - Heartbeat callback moved inside store

2. **MODIFIED:** `frontend/apps/web-ui/src/hooks/useHeartbeat.ts`
   - Removed 10 console.log statements
   - Simplified from 83 LOC → 41 LOC (50% reduction)
   - Removed intermediate state management
   - Now delegates to store

3. **DELETED:** `frontend/apps/web-ui/src/hooks/useRbeeSDK.ts`
   - Moved to `@rbee/react` package

4. **DELETED:** `frontend/packages/rbee-react/src/useRbeeSDK.ts`
   - Replaced by modular structure

## Engineering Rules Compliance

- ✅ No console logs in production code
- ✅ Single responsibility per module
- ✅ Direct connections (no unnecessary layers)
- ✅ Clean, maintainable code
- ✅ Team signature (TEAM-291)

---

**TEAM-291 COMPLETE** - SDK now connects directly to zustand, all console logs removed, code simplified by 50%.
