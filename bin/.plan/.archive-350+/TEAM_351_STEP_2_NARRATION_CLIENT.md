# TEAM-351 Step 2: Create @rbee/narration-client Package

**Estimated Time:** 45-60 minutes  
**Priority:** CRITICAL  
**Previous Step:** TEAM_351_STEP_1_SHARED_CONFIG.md  
**Next Step:** TEAM_351_STEP_3_IFRAME_BRIDGE.md

---

## Mission

Create the `@rbee/narration-client` package - reusable narration handling for all UIs.

**Why This Matters:**
- Eliminates ~100 LOC duplicate code per UI
- Single implementation of SSE parsing
- Single implementation of postMessage bridge
- Consistent narration handling everywhere

---

## Deliverables Checklist

- [ ] Package structure created
- [ ] package.json created
- [ ] tsconfig.json created
- [ ] src/types.ts created (type definitions)
- [ ] src/config.ts created (service configs)
- [ ] src/parser.ts created (SSE parsing)
- [ ] src/bridge.ts created (postMessage bridge)
- [ ] src/index.ts created (exports)
- [ ] README.md created
- [ ] Package builds successfully

---

## Step 1: Create Package Structure

```bash
mkdir -p frontend/packages/narration-client/src
cd frontend/packages/narration-client
```

---

## Step 2: Create package.json

```bash
cat > package.json << 'EOF'
{
  "name": "@rbee/narration-client",
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
/**
 * Narration event from backend SSE stream
 * This is the format ALL backends send (Queen, Hive, Worker)
 */
export interface BackendNarrationEvent {
  actor: string
  action: string
  human: string          // The message text
  formatted?: string     // Contains function name with ANSI codes
  level?: string
  timestamp?: number
  job_id?: string
  target?: string
}

/**
 * Message sent to parent window via postMessage
 */
export interface NarrationMessage {
  type: 'NARRATION_EVENT'
  payload: BackendNarrationEvent
  source: string           // 'queen-rbee' | 'rbee-hive' | 'llm-worker'
  timestamp: number
}
EOF
```

---

## Step 5: Create src/config.ts

```bash
cat > src/config.ts << 'EOF'
export interface ServiceConfig {
  name: string
  devPort: number
  prodPort: number
  keeperDevPort: number
  keeperProdOrigin: string
}

export const SERVICES: Record<string, ServiceConfig> = {
  queen: {
    name: 'queen-rbee',
    devPort: 7834,
    prodPort: 7833,
    keeperDevPort: 5173,
    keeperProdOrigin: '*',  // Tauri app
  },
  hive: {
    name: 'rbee-hive',
    devPort: 7836,
    prodPort: 7835,
    keeperDevPort: 5173,
    keeperProdOrigin: '*',
  },
  worker: {
    name: 'llm-worker',
    devPort: 7837,
    prodPort: 8080,
    keeperDevPort: 5173,
    keeperProdOrigin: '*',
  },
}

/**
 * Get parent origin based on current service location
 */
export function getParentOrigin(serviceConfig: ServiceConfig): string {
  const currentPort = window.location.port
  const isOnDevServer = currentPort === serviceConfig.devPort.toString()
  
  return isOnDevServer
    ? `http://localhost:${serviceConfig.keeperDevPort}`
    : serviceConfig.keeperProdOrigin
}
EOF
```

---

## Step 6: Create src/parser.ts

```bash
cat > src/parser.ts << 'EOF'
import type { BackendNarrationEvent } from './types'

/**
 * Parse SSE line into narration event
 * Handles [DONE] marker gracefully
 */
export function parseNarrationLine(line: string): BackendNarrationEvent | null {
  // Skip [DONE] marker gracefully (not an error)
  if (line === '[DONE]' || line.trim() === '[DONE]') {
    return null
  }
  
  try {
    // Remove SSE "data: " prefix if present
    const jsonStr = line.startsWith('data: ') ? line.slice(6) : line
    const event = JSON.parse(jsonStr.trim())
    
    if (event && typeof event === 'object' && event.actor && event.action) {
      return event as BackendNarrationEvent
    }
    return null
  } catch (error) {
    console.warn('[NarrationClient] Failed to parse line:', line, error)
    return null
  }
}
EOF
```

---

## Step 7: Create src/bridge.ts

```bash
cat > src/bridge.ts << 'EOF'
import type { BackendNarrationEvent, NarrationMessage } from './types'
import type { ServiceConfig } from './config'
import { getParentOrigin } from './config'
import { parseNarrationLine } from './parser'

/**
 * Send narration event to parent window (rbee-keeper)
 */
export function sendToParent(
  event: BackendNarrationEvent,
  serviceConfig: ServiceConfig
): void {
  if (typeof window === 'undefined' || window.parent === window) {
    return
  }

  const message: NarrationMessage = {
    type: 'NARRATION_EVENT',
    payload: event,
    source: serviceConfig.name,
    timestamp: Date.now(),
  }

  try {
    const parentOrigin = getParentOrigin(serviceConfig)
    
    console.log(`[${serviceConfig.name}] Sending to parent:`, {
      origin: parentOrigin,
      event: event.action,
    })
    
    window.parent.postMessage(message, parentOrigin)
  } catch (error) {
    console.warn(`[${serviceConfig.name}] Failed to send to parent:`, error)
  }
}

/**
 * Create a stream handler for SSE narration events
 * @param serviceConfig - Service configuration
 * @param onLocal - Optional local handler for events
 */
export function createStreamHandler(
  serviceConfig: ServiceConfig,
  onLocal?: (event: BackendNarrationEvent) => void
) {
  return (line: string) => {
    const event = parseNarrationLine(line)
    if (event) {
      sendToParent(event, serviceConfig)
      onLocal?.(event)
    }
  }
}
EOF
```

---

## Step 8: Create src/index.ts

```bash
cat > src/index.ts << 'EOF'
export * from './types'
export * from './config'
export * from './parser'
export * from './bridge'
EOF
```

---

## Step 9: Create README.md

```bash
cat > README.md << 'EOF'
# @rbee/narration-client

Shared narration client for handling backend SSE narration events and forwarding to parent window.

## Installation

```bash
pnpm add @rbee/narration-client
```

## Usage

### Basic Usage

```typescript
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

// Create stream handler for Queen
const handleNarration = createStreamHandler(SERVICES.queen)

// Use in SSE stream
for await (const line of stream) {
  handleNarration(line)
}
```

### With Local Handler

```typescript
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

const handleNarration = createStreamHandler(
  SERVICES.queen,
  (event) => {
    console.log('Local event:', event)
  }
)
```

### Manual Parsing

```typescript
import { parseNarrationLine } from '@rbee/narration-client'

const event = parseNarrationLine('{"actor":"queen","action":"test"}')
if (event) {
  console.log(event)
}
```

## Features

- ✅ Automatic SSE line parsing
- ✅ Automatic postMessage to parent
- ✅ Environment-aware origin detection
- ✅ Graceful [DONE] marker handling
- ✅ Type-safe event handling

## Services

- `SERVICES.queen` - Queen UI
- `SERVICES.hive` - Hive UI
- `SERVICES.worker` - Worker UI
EOF
```

---

## Step 10: Build and Test

```bash
# Install dependencies
pnpm install

# Build the package
pnpm build
```

---

## Verification Checklist

- [ ] `dist/` folder created
- [ ] All TypeScript files compiled
- [ ] No compilation errors
- [ ] Types exported correctly
- [ ] SERVICES constant includes queen, hive, worker

---

## Expected Output

### After `pnpm build`
```
dist/
├── index.js
├── index.d.ts
├── types.js
├── types.d.ts
├── config.js
├── config.d.ts
├── parser.js
├── parser.d.ts
├── bridge.js
└── bridge.d.ts
```

---

## Test the Package

Create a test file to verify exports:

```typescript
// test.ts
import { 
  createStreamHandler, 
  SERVICES, 
  parseNarrationLine,
  type BackendNarrationEvent 
} from '@rbee/narration-client'

console.log('✅ All imports work')
console.log('Services:', Object.keys(SERVICES))

// Test parser
const event = parseNarrationLine('{"actor":"test","action":"test","human":"test"}')
console.log('Parsed event:', event)

// Test stream handler
const handler = createStreamHandler(SERVICES.queen)
console.log('Handler created:', typeof handler)
```

Run: `npx tsx test.ts`

---

## Troubleshooting

### Issue: TypeScript errors

```bash
pnpm install
pnpm build
```

### Issue: Exports not found

Check `src/index.ts` exports all modules.

### Issue: Window not defined

This is normal - `window` is only available in browser. The code handles this with:
```typescript
if (typeof window === 'undefined' || window.parent === window) {
  return
}
```

---

## Next Step

✅ **Step 2 Complete!**

**Next:** `TEAM_351_STEP_3_IFRAME_BRIDGE.md` - Create the iframe bridge package

---

**TEAM-351 Step 2: Reusable narration handling!** 🎯
