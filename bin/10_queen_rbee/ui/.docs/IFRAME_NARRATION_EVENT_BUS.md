# iframe Narration Event Bus Architecture

**Problem Identified:** Oct 29, 2025

## The Issue

The Queen web UI is embedded in rbee-keeper via an **iframe** (`http://localhost:7833`), but narration events from Queen's SSE streams are isolated inside the iframe and cannot reach rbee-keeper's narration store.

### Current Architecture

```
┌─────────────────────────────────────────────────────────┐
│ rbee-keeper (localhost:7834)                            │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ narrationStore.ts (Zustand)                    │   │
│  │ - Listens to Tauri events                      │   │
│  │ - Persists narration entries                   │   │
│  │ - Currently: NO access to Queen iframe data    │   │
│  └────────────────────────────────────────────────┘   │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ <iframe src="http://localhost:7833" />         │   │
│  │                                                 │   │
│  │  ┌──────────────────────────────────────────┐ │   │
│  │  │ Queen UI (isolated context)              │ │   │
│  │  │                                          │ │   │
│  │  │ - JobClient connects to SSE              │ │   │
│  │  │ - Receives narration events              │ │   │
│  │  │ - Events TRAPPED in iframe               │ │   │
│  │  │ - Cannot reach parent narrationStore     │ │   │
│  │  └──────────────────────────────────────────┘ │   │
│  └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Why This Matters

1. **Queen operations** (RHAI scripts, worker management) generate narration via SSE
2. **SSE events** are received by Queen's JobClient inside the iframe
3. **rbee-keeper's narration panel** shows NO Queen events (isolated context)
4. **User sees no feedback** when using Queen features

## Solution: postMessage Event Bus

Use the browser's `window.postMessage()` API to bridge the iframe boundary.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│ rbee-keeper (parent window)                             │
│                                                         │
│  ┌────────────────────────────────────────────────┐   │
│  │ narrationStore.ts                              │   │
│  │                                                 │   │
│  │ 1. Listen to window.addEventListener()         │   │
│  │    'message' events                             │   │
│  │                                                 │   │
│  │ 2. Filter: event.data.type === 'NARRATION'    │   │
│  │                                                 │   │
│  │ 3. Call addEntry(event.data.payload)          │   │
│  └────────────────────────────────────────────────┘   │
│                          ▲                              │
│                          │ postMessage                  │
│                          │                              │
│  ┌────────────────────────────────────────────────┐   │
│  │ <iframe src="http://localhost:7833" />         │   │
│  │                                                 │   │
│  │  ┌──────────────────────────────────────────┐ │   │
│  │  │ Queen UI                                 │ │   │
│  │  │                                          │ │   │
│  │  │ JobClient receives SSE                   │ │   │
│  │  │        ↓                                 │ │   │
│  │  │ onNarration callback                     │ │   │
│  │  │        ↓                                 │ │   │
│  │  │ window.parent.postMessage({              │ │   │
│  │  │   type: 'NARRATION',                     │ │   │
│  │  │   payload: narrationEvent                │ │   │
│  │  │ }, 'http://localhost:7834')              │ │   │
│  │  └──────────────────────────────────────────┘ │   │
│  └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Implementation Plan

### 1. Queen UI: Send Events to Parent

**File:** `bin/10_queen_rbee/ui/app/src/App.tsx` (or wherever JobClient is used)

```typescript
// When JobClient receives narration via SSE
jobClient.submitAndStream(operation, (narrationLine) => {
  // Parse narration event
  const event = parseNarrationEvent(narrationLine);
  
  // Send to parent window (rbee-keeper)
  if (window.parent !== window) {
    window.parent.postMessage({
      type: 'QUEEN_NARRATION',
      payload: event,
      source: 'queen-rbee',
      timestamp: Date.now()
    }, 'http://localhost:7834'); // rbee-keeper origin
  }
  
  // Also handle locally if needed
  // ...
});
```

### 2. rbee-keeper: Listen for Events

**File:** `bin/00_rbee_keeper/ui/src/store/narrationStore.ts`

```typescript
// Add to the store creation
export const useNarrationStore = create<NarrationState>()(
  persist(
    immer((set) => ({
      entries: [],
      idCounter: 0,
      showNarration: true,

      addEntry: (event: NarrationEvent) => {
        set((state) => {
          state.entries.unshift({
            ...event,
            id: state.idCounter++,
          })
        })
      },

      // ... other actions
    })),
    { name: 'narration-store', /* ... */ }
  ),
)

// TEAM-XXX: Listen to iframe postMessage events
if (typeof window !== 'undefined') {
  window.addEventListener('message', (event) => {
    // Security: Verify origin
    if (event.origin !== 'http://localhost:7833') {
      return;
    }

    // Filter for narration events
    if (event.data?.type === 'QUEEN_NARRATION') {
      const narrationEvent = event.data.payload;
      
      // Add to store
      useNarrationStore.getState().addEntry(narrationEvent);
    }
  });
}
```

### 3. Alternative: Setup in App Component

**File:** `bin/00_rbee_keeper/ui/src/App.tsx`

```typescript
import { useEffect } from 'react';
import { useNarrationStore } from './store/narrationStore';

function App() {
  const addEntry = useNarrationStore((s) => s.addEntry);

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      // Security: Verify origin
      if (event.origin !== 'http://localhost:7833') {
        return;
      }

      // Filter for narration events from Queen
      if (event.data?.type === 'QUEEN_NARRATION') {
        addEntry(event.data.payload);
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [addEntry]);

  return (
    // ... rest of app
  );
}
```

## Security Considerations

1. **Origin Validation:** Always check `event.origin` matches expected Queen URL
2. **Message Type Filtering:** Only process known message types
3. **Data Validation:** Validate payload structure before using
4. **Localhost Only:** This pattern is safe for localhost development

```typescript
// Robust validation
function isValidNarrationMessage(event: MessageEvent): boolean {
  return (
    event.origin === 'http://localhost:7833' &&
    event.data?.type === 'QUEEN_NARRATION' &&
    typeof event.data?.payload === 'object' &&
    event.data?.payload !== null
  );
}
```

## Message Format

```typescript
interface NarrationMessage {
  type: 'QUEEN_NARRATION';
  payload: NarrationEvent;
  source: 'queen-rbee';
  timestamp: number;
}

// Example
{
  type: 'QUEEN_NARRATION',
  payload: {
    actor: 'queen-rbee',
    action: 'rhai_save_start',
    human: '💾 Saving RHAI script: my-script',
    level: 'Info',
    // ... other NarrationEvent fields
  },
  source: 'queen-rbee',
  timestamp: 1730217840000
}
```

## Testing

1. **Open rbee-keeper** at `http://localhost:7834`
2. **Navigate to Queen page** (iframe loads `http://localhost:7833`)
3. **Trigger Queen operation** (e.g., save RHAI script)
4. **Verify:** Narration appears in rbee-keeper's narration panel
5. **Check console:** No CORS errors, messages received

## Benefits

✅ **No CORS issues** - postMessage works across origins  
✅ **Standard browser API** - Well-supported, reliable  
✅ **Minimal changes** - Add listener in keeper, sender in queen  
✅ **Type-safe** - Can validate message structure  
✅ **Bidirectional** - Can send messages both ways if needed  

## Limitations

⚠️ **Localhost only** - Origin validation hardcoded  
⚠️ **Manual sync** - Not automatic like shared state  
⚠️ **Serialization** - Only JSON-serializable data  

## Next Steps

1. [ ] Implement postMessage sender in Queen UI
2. [ ] Add message listener in rbee-keeper narrationStore
3. [ ] Test with RHAI script operations
4. [ ] Add error handling and logging
5. [ ] Document message types and payloads
6. [ ] Consider bidirectional communication (keeper → queen)

## References

- [MDN: Window.postMessage()](https://developer.mozilla.org/en-US/docs/Web/API/Window/postMessage)
- [MDN: MessageEvent](https://developer.mozilla.org/en-US/docs/Web/API/MessageEvent)
- Current narration store: `bin/00_rbee_keeper/ui/src/store/narrationStore.ts`
- Queen iframe: `bin/00_rbee_keeper/ui/src/pages/QueenPage.tsx`
