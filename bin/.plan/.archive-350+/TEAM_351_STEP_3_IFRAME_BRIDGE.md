# TEAM-351 Step 3: Create @rbee/iframe-bridge Package

**Estimated Time:** 30 minutes  
**Priority:** MEDIUM  
**Previous Step:** TEAM_351_STEP_2_NARRATION_CLIENT.md  
**Next Step:** TEAM_351_STEP_4_DEV_UTILS.md

---

## Mission

Create the `@rbee/iframe-bridge` package - generic iframe ↔ parent communication utilities.

**Why This Matters:**
- Reusable message sending/receiving
- Origin validation in one place
- Type-safe message handling
- Easy cleanup

---

## Deliverables Checklist

- [ ] Package structure created
- [ ] package.json created
- [ ] tsconfig.json created
- [ ] src/types.ts created
- [ ] src/validator.ts created
- [ ] src/sender.ts created
- [ ] src/receiver.ts created
- [ ] src/index.ts created
- [ ] README.md created
- [ ] Package builds successfully

---

## Step 1: Create Package Structure

```bash
mkdir -p frontend/packages/iframe-bridge/src
cd frontend/packages/iframe-bridge
```

---

## Step 2: Create package.json

```bash
cat > package.json << 'EOF'
{
  "name": "@rbee/iframe-bridge",
  "version": "0.1.0",
  "type": "module",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.js",
      "types": "./dist/index.d.ts"
    }
  },
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch"
  },
  "devDependencies": {
    "typescript": "^5.0.0"
  }
}
EOF
```

---

## Step 3: Create tsconfig.json

```bash
cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ES2020",
    "moduleResolution": "node",
    "declaration": true,
    "outDir": "./dist",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
EOF
```

---

## Step 4: Create src/types.ts

```bash
cat > src/types.ts << 'EOF'
export interface BaseMessage {
  type: string
  source: string
  timestamp: number
}

export interface NarrationMessage extends BaseMessage {
  type: 'NARRATION_EVENT'
  payload: any
}

export interface CommandMessage extends BaseMessage {
  type: 'COMMAND'
  command: string
  args?: any
}

export type IframeMessage = NarrationMessage | CommandMessage
EOF
```

---

## Step 5: Create src/validator.ts

```bash
cat > src/validator.ts << 'EOF'
export interface OriginConfig {
  allowedOrigins: string[]
  strictMode?: boolean
}

export function validateOrigin(
  origin: string,
  config: OriginConfig
): boolean {
  // Allow wildcard in non-strict mode
  if (!config.strictMode && config.allowedOrigins.includes('*')) {
    return true
  }
  
  return config.allowedOrigins.includes(origin)
}

export function createOriginValidator(config: OriginConfig) {
  return (origin: string) => validateOrigin(origin, config)
}
EOF
```

---

## Step 6: Create src/sender.ts

```bash
cat > src/sender.ts << 'EOF'
import type { IframeMessage } from './types'

export interface SenderConfig {
  targetOrigin: string
  debug?: boolean
}

export function createMessageSender(config: SenderConfig) {
  return (message: IframeMessage) => {
    if (typeof window === 'undefined' || window.parent === window) {
      return
    }

    try {
      if (config.debug) {
        console.log('[IframeBridge] Sending:', message.type, 'to', config.targetOrigin)
      }
      
      window.parent.postMessage(message, config.targetOrigin)
    } catch (error) {
      console.warn('[IframeBridge] Send failed:', error)
    }
  }
}
EOF
```

---

## Step 7: Create src/receiver.ts

```bash
cat > src/receiver.ts << 'EOF'
import type { IframeMessage } from './types'
import { createOriginValidator, type OriginConfig } from './validator'

export interface ReceiverConfig extends OriginConfig {
  onMessage: (message: IframeMessage) => void
  debug?: boolean
}

export function createMessageReceiver(config: ReceiverConfig) {
  const validateOrigin = createOriginValidator(config)
  
  const handleMessage = (event: MessageEvent) => {
    if (!validateOrigin(event.origin)) {
      if (config.debug) {
        console.warn('[IframeBridge] Rejected origin:', event.origin)
      }
      return
    }

    if (event.data && typeof event.data === 'object' && event.data.type) {
      config.onMessage(event.data as IframeMessage)
    }
  }

  window.addEventListener('message', handleMessage)
  
  return () => {
    window.removeEventListener('message', handleMessage)
  }
}
EOF
```

---

## Step 8: Create src/index.ts

```bash
cat > src/index.ts << 'EOF'
export * from './types'
export * from './validator'
export * from './sender'
export * from './receiver'
EOF
```

---

## Step 9: Create README.md

```bash
cat > README.md << 'EOF'
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
EOF
```

---

## Step 10: Build and Test

```bash
pnpm install
pnpm build
```

---

## Verification Checklist

- [ ] `dist/` folder created
- [ ] All files compiled
- [ ] No TypeScript errors
- [ ] All exports available

---

## Expected Output

```
dist/
├── index.js
├── index.d.ts
├── types.js
├── types.d.ts
├── validator.js
├── validator.d.ts
├── sender.js
├── sender.d.ts
├── receiver.js
└── receiver.d.ts
```

---

## Next Step

✅ **Step 3 Complete!**

**Next:** `TEAM_351_STEP_4_DEV_UTILS.md` - Create development utilities package

---

**TEAM-351 Step 3: Generic iframe bridge!** 🌉
