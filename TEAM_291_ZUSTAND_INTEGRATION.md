# TEAM-291: Zustand State Management Integration

**Status:** ✅ COMPLETE

**Mission:** Integrate zustand for centralized state management of Queen status and hives in the web-ui Next.js application.

## Deliverables

### 1. Zustand Store Created
**File:** `frontend/apps/web-ui/src/stores/rbeeStore.ts` (114 LOC)

**State Structure:**
```typescript
interface RbeeState {
  // Queen status
  queen: {
    connected: boolean;
    lastUpdate: string | null;
    error: string | null;
  }
  
  // Hives
  hives: HiveInfo[];
  hivesOnline: number;
  hivesAvailable: number;
  
  // Workers
  workersOnline: number;
  workersAvailable: number;
  workerIds: string[];
  
  // Raw heartbeat
  lastHeartbeat: HeartbeatSnapshot | null;
}
```

**Actions:**
- `setQueenConnected(connected: boolean)` - Update queen connection status
- `setQueenError(error: string | null)` - Set queen error state
- `updateFromHeartbeat(heartbeat: HeartbeatSnapshot)` - Update all state from heartbeat
- `resetState()` - Reset to initial state

### 2. Hook Integration
**File:** `frontend/apps/web-ui/src/hooks/useHeartbeat.ts` (Modified)

**Changes:**
- Removed local `useState` for heartbeat, connected, error
- Integrated zustand store actions
- Hook now updates store instead of local state
- Returns simplified interface: `{ connected, loading, error }`

**Key Pattern:**
```typescript
// TEAM-291: Update zustand store instead of local state
monitor.start((snapshot: HeartbeatSnapshot) => {
  updateFromHeartbeat(snapshot);
});
```

### 3. Page Component Update
**File:** `frontend/apps/web-ui/src/app/page.tsx` (Modified)

**Changes:**
- Imports zustand store: `useRbeeStore()`
- Reads state from store instead of hook return value
- All UI now driven by centralized state
- Hives displayed from `hives` array with structured `HiveInfo` objects

**Before:**
```typescript
const { heartbeat, connected, loading, error } = useHeartbeat();
// Access: heartbeat?.hives_online
```

**After:**
```typescript
const { connected, loading, error } = useHeartbeat();
const { queen, hives, hivesOnline, hivesAvailable, ... } = useRbeeStore();
// Access: hivesOnline, hives array
```

## Architecture Benefits

### 1. Centralized State
- Single source of truth for Queen and hives
- No prop drilling needed
- Easy to add new consumers (other pages/components)

### 2. Separation of Concerns
- `useHeartbeat` hook: SSE connection management only
- `useRbeeStore` store: State management only
- Components: Pure presentation logic

### 3. Scalability
- New components can access state via `useRbeeStore()`
- No need to pass props through component tree
- Easy to add new state slices (models, jobs, etc.)

### 4. Developer Experience
- Type-safe with TypeScript
- Minimal boilerplate
- React DevTools integration
- Easy to test (can mock store)

## Data Flow

```
SSE Stream (queen-rbee)
    ↓
HeartbeatMonitor (WASM)
    ↓
useHeartbeat hook
    ↓
updateFromHeartbeat(snapshot)
    ↓
Zustand Store (rbeeStore)
    ↓
React Components (page.tsx, future components)
```

## Files Changed

1. **NEW:** `frontend/apps/web-ui/src/stores/rbeeStore.ts` (114 LOC)
   - Zustand store with Queen and hives state
   - Actions for updating state
   - TypeScript interfaces

2. **MODIFIED:** `frontend/apps/web-ui/src/hooks/useHeartbeat.ts`
   - Integrated zustand store
   - Removed local state management
   - Simplified return interface

3. **MODIFIED:** `frontend/apps/web-ui/src/app/page.tsx`
   - Imports zustand store
   - Reads state from store
   - Updated all references to use store state

4. **MODIFIED:** `frontend/apps/web-ui/package.json`
   - Added `zustand: ^5.0.8` dependency

## Verification

### Compilation
```bash
cd frontend/apps/web-ui
pnpm build
```

### Runtime
- Queen status updates from SSE stream
- Hives displayed with structured data
- Workers count updates in real-time
- All state centralized in zustand store

## Next Steps (Future Teams)

### Potential Enhancements
1. **Persistence:** Add zustand middleware for localStorage persistence
2. **DevTools:** Add zustand DevTools middleware for debugging
3. **Models State:** Add models slice to store
4. **Jobs State:** Add active jobs tracking
5. **Optimistic Updates:** Add optimistic UI updates for actions
6. **Selectors:** Add memoized selectors for derived state

### Usage Example for Future Components
```typescript
// Any component can now access Queen/hives state
import { useRbeeStore } from '@/src/stores/rbeeStore';

function MyComponent() {
  const { queen, hives } = useRbeeStore();
  
  return (
    <div>
      Queen: {queen.connected ? 'Online' : 'Offline'}
      Hives: {hives.length}
    </div>
  );
}
```

## Code Signatures

All code marked with `// TEAM-291:` comments for traceability.

## Engineering Rules Compliance

- ✅ No TODO markers
- ✅ Complete implementation (no stubs)
- ✅ TypeScript types defined
- ✅ Follows existing patterns
- ✅ No breaking changes
- ✅ Team signature added

---

**TEAM-291 COMPLETE** - Zustand state management successfully integrated for Queen status and hives.
