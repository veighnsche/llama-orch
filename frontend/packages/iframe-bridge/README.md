# @rbee/iframe-bridge

Generic iframe ↔ parent window communication utilities.

## Installation

```bash
pnpm add @rbee/iframe-bridge
```

## Usage

### Child → Parent Communication

```typescript
// In iframe (Queen/Hive)
import { createMessageSender } from '@rbee/iframe-bridge'

const sendMessage = createMessageSender({
  targetOrigin: 'http://localhost:5173',
  debug: true
})

sendMessage({
  type: 'NARRATION_EVENT',
  payload: { /* ... */ },
  source: 'queen-rbee',
  timestamp: Date.now()
})
```

```typescript
// In parent (Keeper)
import { createMessageReceiver } from '@rbee/iframe-bridge'

const cleanup = createMessageReceiver({
  allowedOrigins: ['http://localhost:7833', 'http://localhost:7834'],
  onMessage: (message) => {
    console.log('Received from child:', message)
  }
})

// Later: cleanup()
```

### Parent → Child Communication

```typescript
// In parent (Keeper) - broadcast to ALL iframes
import { broadcastToIframes } from '@rbee/iframe-bridge'

broadcastToIframes({
  type: 'THEME_CHANGE',
  source: 'keeper',
  timestamp: Date.now(),
  theme: 'dark'
})
```

```typescript
// In iframe (Queen/Hive) - receive from parent
import { receiveFromParent } from '@rbee/iframe-bridge'

const cleanup = receiveFromParent((message) => {
  console.log('Received from parent:', message)
  // Handle message...
})

// Later: cleanup()
```

### Theme Synchronization Helpers

```typescript
// In parent (Keeper) - auto-broadcast theme changes
import { broadcastThemeChanges } from '@rbee/iframe-bridge'

useEffect(() => {
  const cleanup = broadcastThemeChanges()
  return cleanup
}, [])
```

```typescript
// In iframe (Queen/Hive) - auto-receive theme changes
import { receiveThemeChanges } from '@rbee/iframe-bridge'

useEffect(() => {
  const cleanup = receiveThemeChanges()
  return cleanup
}, [])
```

## Features

- ✅ **Bidirectional:** Child ↔ Parent communication
- ✅ **Generic:** `broadcastToIframes()` / `receiveFromParent()` work with any message
- ✅ **Type-safe:** Full TypeScript support
- ✅ **Origin validation:** Security built-in
- ✅ **Helper functions:** Theme sync out of the box
- ✅ **Cleanup:** All listeners return cleanup functions
