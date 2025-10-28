# TEAM-291: Real-Time Heartbeats Clarification

**Status:** ✅ CONFIRMED - Heartbeats ARE real-time

## Clarification

The heartbeats **ARE** coming in real-time and updating the store continuously. The confusion was about the implementation, not the behavior.

## How It Actually Works

### SSE Stream (Server-Sent Events)
```
Queen sends heartbeat every ~5 seconds
    ↓
SSE connection (WebSocket-like)
    ↓
HeartbeatMonitor.start(callback)
    ↓
Callback fires EVERY TIME queen sends heartbeat
    ↓
Store updates immediately
    ↓
Components re-render with new data
```

### The Code Flow

**1. Initialization (once):**
```typescript
const monitor = new sdk.HeartbeatMonitor('http://localhost:8500');
startMonitoring(monitor, baseUrl);
```

**2. Real-Time Updates (every ~5 seconds):**
```typescript
// Inside startMonitoring:
monitorInstance.start((snapshot: HeartbeatSnapshot) => {
  // ← THIS CALLBACK FIRES EVERY TIME QUEEN SENDS HEARTBEAT
  
  set({
    queen: { connected: true, lastUpdate: snapshot.timestamp, ... },
    hives: snapshot.hive_ids.map(...),
    hivesOnline: snapshot.hives_online,
    workersOnline: snapshot.workers_online,
    // ... all data updated in real-time
  });
});
```

**3. Components React (automatically):**
```typescript
// Component re-renders every time store updates
const { hives, workersOnline } = useRbeeStore();
```

## Timeline

```
T+0s:   startMonitoring() called
T+0s:   SSE connection established
T+0s:   First heartbeat received → store updated → components render
T+5s:   Second heartbeat received → store updated → components render
T+10s:  Third heartbeat received → store updated → components render
T+15s:  Fourth heartbeat received → store updated → components render
...     (continues every ~5 seconds)
```

## What Was Confusing

The code had TWO ways to update the store:

**Old (removed):**
```typescript
updateFromHeartbeat: (heartbeat) => set({ ... })  // ← Manual, not used
```

**Current (correct):**
```typescript
startMonitoring: (monitor) => {
  monitor.start((snapshot) => {
    set({ ... })  // ← Automatic, fires every heartbeat
  });
}
```

## What Changed

### Removed
- ❌ `setQueenConnected()` - Not needed (heartbeat sets connected=true)
- ❌ `updateFromHeartbeat()` - Duplicate of what startMonitoring does

### Kept
- ✅ `startMonitoring()` - Sets up real-time updates
- ✅ `stopMonitoring()` - Stops real-time updates
- ✅ `setQueenError()` - For error handling

### Clarified
Added comments to make it obvious:
```typescript
// Start new monitor - callback fires on EVERY heartbeat from queen
monitorInstance.start((snapshot: HeartbeatSnapshot) => {
  // REAL-TIME UPDATE: This runs every time queen sends a heartbeat
  set({ ... });
});
```

## Verification

To verify real-time updates are working:

1. **Start queen:**
   ```bash
   ./rbee queen start
   ```

2. **Open web UI:**
   ```
   http://localhost:3002
   ```

3. **Watch the "Last update" timestamp:**
   - Should update every ~5 seconds
   - Shows real-time connection

4. **Start a hive:**
   ```bash
   ./rbee hive start -a localhost
   ```

5. **Watch the UI:**
   - "Hives: 0" → "Hives: 1" (within 5 seconds)
   - Real-time update without refresh

6. **Spawn a worker:**
   ```bash
   ./rbee worker spawn --hive localhost --model llama-3-8b
   ```

7. **Watch the UI:**
   - "Workers: 0" → "Workers: 1" (within 5 seconds)
   - Real-time update without refresh

## The Architecture is Correct

```
Queen (backend)
  ↓ SSE (every ~5s)
HeartbeatMonitor (WASM SDK)
  ↓ callback (every ~5s)
Zustand Store
  ↓ subscription (immediate)
React Components
  ↓ re-render (immediate)
UI Updates (real-time)
```

**No polling. No manual refresh. Pure real-time SSE.**

## Summary

- ✅ Heartbeats ARE real-time (SSE stream)
- ✅ Store updates every ~5 seconds automatically
- ✅ Components re-render immediately
- ✅ No manual updates needed
- ✅ Single callback handles everything

The implementation was already correct - just needed clearer comments and removal of unused duplicate code.

---

**TEAM-291** - Real-time heartbeats confirmed working as designed.
