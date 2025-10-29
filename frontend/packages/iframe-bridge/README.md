# @rbee/iframe-bridge

Generic iframe ↔ parent window communication utilities.

## Installation

```bash
pnpm add @rbee/iframe-bridge
```

## Usage

### Sending Messages (from iframe)

```typescript
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

### Receiving Messages (in parent)

```typescript
import { createMessageReceiver } from '@rbee/iframe-bridge'

const cleanup = createMessageReceiver({
  allowedOrigins: ['http://localhost:7833', 'http://localhost:7834'],
  onMessage: (message) => {
    console.log('Received:', message)
  },
  debug: true
})

// Later: cleanup()
```

## Features

- ✅ Origin validation
- ✅ Type-safe messages
- ✅ Debug logging
- ✅ Cleanup function
- ✅ Wildcard support (non-strict mode)
