# TEAM-350: Architecture Recommendations for Future Teams

**Status:** 📋 RECOMMENDATIONS

**Purpose:** Guide future teams on creating reusable packages and maintaining consistent port configuration across all UIs (Queen, Hive, Worker).

---

## Table of Contents

1. [Reusable Package Architecture](#reusable-package-architecture)
2. [Port Configuration Management](#port-configuration-management)
3. [Shared Logic Extraction](#shared-logic-extraction)
4. [Implementation Checklist](#implementation-checklist)

---

## Reusable Package Architecture

### Current Problem

**We have duplicate code across UIs:**
- Queen has `narrationBridge.ts` 
- Hive will need the same logic
- Worker will need the same logic
- Each UI duplicates environment detection, postMessage, type mapping

**This violates DRY (Don't Repeat Yourself).**

### Recommended Solution: Create Shared Packages

```
frontend/packages/
├── @rbee/ui/                    # ✅ Already exists (UI components)
├── @rbee/narration-client/      # 🔜 NEW: Narration client logic
├── @rbee/iframe-bridge/         # 🔜 NEW: iframe ↔ parent communication
└── @rbee/dev-utils/             # 🔜 NEW: Environment detection utilities
```

---

## 1. @rbee/narration-client Package

### Purpose
Handle narration events from backend SSE streams and forward to parent window.

### What to Extract

**From:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/utils/narrationBridge.ts`

**To:** `frontend/packages/narration-client/src/`

### Package Structure

```
frontend/packages/narration-client/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.ts                 # Main exports
│   ├── types.ts                 # Shared types
│   ├── bridge.ts                # postMessage bridge
│   ├── parser.ts                # SSE line parsing
│   └── config.ts                # Port configuration
└── README.md
```

### Implementation

**1. Create types.ts:**
```typescript
// frontend/packages/narration-client/src/types.ts

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
  type: 'NARRATION_EVENT'  // Generic, not 'QUEEN_NARRATION'
  payload: BackendNarrationEvent
  source: string           // 'queen-rbee' | 'rbee-hive' | 'llm-worker'
  timestamp: number
}
```

**2. Create config.ts:**
```typescript
// frontend/packages/narration-client/src/config.ts

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
```

**3. Create parser.ts:**
```typescript
// frontend/packages/narration-client/src/parser.ts

import type { BackendNarrationEvent } from './types'

/**
 * Parse SSE line into narration event
 * Handles [DONE] marker gracefully
 */
export function parseNarrationLine(line: string): BackendNarrationEvent | null {
  // Skip [DONE] marker gracefully
  if (line === '[DONE]' || line.trim() === '[DONE]') {
    return null
  }
  
  try {
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
```

**4. Create bridge.ts:**
```typescript
// frontend/packages/narration-client/src/bridge.ts

import type { BackendNarrationEvent, NarrationMessage } from './types'
import type { ServiceConfig } from './config'
import { getParentOrigin } from './config'

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
```

**5. Create index.ts:**
```typescript
// frontend/packages/narration-client/src/index.ts

export * from './types'
export * from './config'
export * from './parser'
export * from './bridge'
```

**6. Create package.json:**
```json
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
```

### Usage in Queen UI

```typescript
// bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useRhaiScripts.ts
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

const handleNarration = createStreamHandler(SERVICES.queen)

// Use in SSE stream
for await (const line of stream) {
  handleNarration(line)
}
```

### Usage in Hive UI (Future)

```typescript
// bin/25_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useHiveOperations.ts
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

const handleNarration = createStreamHandler(SERVICES.hive)

// Use in SSE stream
for await (const line of stream) {
  handleNarration(line)
}
```

---

## 2. @rbee/iframe-bridge Package

### Purpose
Handle all iframe ↔ parent window communication, not just narration.

### What to Extract

**Functionality:**
- Message sending (iframe → parent)
- Message receiving (parent → iframe)
- Origin validation
- Type-safe message types

### Package Structure

```
frontend/packages/iframe-bridge/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.ts
│   ├── types.ts
│   ├── sender.ts      # Send messages from iframe
│   ├── receiver.ts    # Receive messages in parent
│   └── validator.ts   # Origin validation
└── README.md
```

### Implementation

**1. Create types.ts:**
```typescript
// frontend/packages/iframe-bridge/src/types.ts

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
```

**2. Create validator.ts:**
```typescript
// frontend/packages/iframe-bridge/src/validator.ts

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
```

**3. Create sender.ts:**
```typescript
// frontend/packages/iframe-bridge/src/sender.ts

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
```

**4. Create receiver.ts:**
```typescript
// frontend/packages/iframe-bridge/src/receiver.ts

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
```

---

## 3. @rbee/dev-utils Package

### Purpose
Environment detection, port utilities, and development helpers.

### Package Structure

```
frontend/packages/dev-utils/
├── package.json
├── tsconfig.json
├── src/
│   ├── index.ts
│   ├── environment.ts
│   ├── ports.ts
│   └── logging.ts
└── README.md
```

### Implementation

**1. Create environment.ts:**
```typescript
// frontend/packages/dev-utils/src/environment.ts

export function isDevelopment(): boolean {
  return import.meta.env.DEV
}

export function isProduction(): boolean {
  return import.meta.env.PROD
}

export function getCurrentPort(): number {
  return parseInt(window.location.port, 10) || 80
}

export function isRunningOnPort(port: number): boolean {
  return getCurrentPort() === port
}
```

**2. Create ports.ts:**
```typescript
// frontend/packages/dev-utils/src/ports.ts

export const PORTS = {
  keeper: {
    dev: 5173,
    prod: null,  // Tauri app
  },
  queen: {
    dev: 7834,
    prod: 7833,
    backend: 7833,
  },
  hive: {
    dev: 7836,
    prod: 7835,
    backend: 7835,
  },
  worker: {
    dev: 7837,
    prod: 8080,
    backend: 8080,
  },
} as const

export type ServiceName = keyof typeof PORTS

export function getServiceUrl(
  service: ServiceName,
  mode: 'dev' | 'prod' = 'dev'
): string {
  const port = PORTS[service][mode]
  return port ? `http://localhost:${port}` : ''
}
```

**3. Create logging.ts:**
```typescript
// frontend/packages/dev-utils/src/logging.ts

export function logStartupMode(
  serviceName: string,
  isDev: boolean,
  port?: number
): void {
  const emoji = isDev ? '🔧' : '🚀'
  const mode = isDev ? 'DEVELOPMENT' : 'PRODUCTION'
  
  console.log(`${emoji} [${serviceName}] Running in ${mode} mode`)
  
  if (isDev && port) {
    console.log(`   - Vite dev server active (hot reload enabled)`)
    console.log(`   - Running on: http://localhost:${port}`)
  } else if (!isDev) {
    console.log(`   - Serving embedded static files`)
  }
}
```

---

## Port Configuration Management

### The Problem

**Every UI needs to know:**
- Its own dev/prod ports
- Keeper's dev/prod ports
- Backend's port
- Other services' ports (for iframe loading)

**This information is scattered across:**
- TypeScript files
- Rust build.rs files
- Backend main.rs files
- Documentation

### The Solution: Single Source of Truth

Create a **shared configuration file** that ALL code references.

### Implementation

**1. Create shared config file:**

```typescript
// frontend/packages/shared-config/src/ports.ts

/**
 * SINGLE SOURCE OF TRUTH for all port configurations
 * 
 * CRITICAL: When adding a new service, update ALL sections:
 * 1. Add to PORTS constant
 * 2. Add to ALLOWED_ORIGINS
 * 3. Update PORT_CONFIGURATION.md
 * 4. Update backend Cargo.toml default port
 */

export const PORTS = {
  keeper: {
    dev: 5173,
    prod: null,  // Tauri app, no HTTP port
  },
  queen: {
    dev: 7834,      // Vite dev server
    prod: 7833,     // Embedded in backend
    backend: 7833,  // Backend HTTP server
  },
  hive: {
    dev: 7836,
    prod: 7835,
    backend: 7835,
  },
  worker: {
    dev: 7837,
    prod: 8080,
    backend: 8080,
  },
} as const

/**
 * Generate allowed origins for message listener
 */
export function getAllowedOrigins(): string[] {
  const origins: string[] = []
  
  for (const [service, ports] of Object.entries(PORTS)) {
    if (service === 'keeper') continue  // Keeper doesn't send messages
    
    if (ports.dev) {
      origins.push(`http://localhost:${ports.dev}`)
    }
    if (ports.prod) {
      origins.push(`http://localhost:${ports.prod}`)
    }
  }
  
  return origins
}

/**
 * Get iframe URL for a service
 */
export function getIframeUrl(
  service: keyof typeof PORTS,
  isDev: boolean
): string {
  const ports = PORTS[service]
  const port = isDev ? ports.dev : ports.prod
  return port ? `http://localhost:${port}` : ''
}

/**
 * Get parent origin for postMessage
 */
export function getParentOrigin(
  currentPort: number
): string {
  // If we're on a dev port, send to keeper dev
  const isDevPort = Object.values(PORTS).some(p => p.dev === currentPort)
  
  return isDevPort
    ? `http://localhost:${PORTS.keeper.dev}`
    : '*'  // Prod: Tauri app
}
```

**2. Use in Keeper iframe:**

```typescript
// bin/00_rbee_keeper/ui/src/pages/QueenPage.tsx
import { getIframeUrl } from '@rbee/shared-config'

const isDev = import.meta.env.DEV
const queenUrl = getIframeUrl('queen', isDev)
```

**3. Use in narration bridge:**

```typescript
// Use @rbee/narration-client package (already includes this logic)
import { SERVICES } from '@rbee/narration-client'
```

**4. Use in message listener:**

```typescript
// bin/00_rbee_keeper/ui/src/utils/narrationListener.ts
import { getAllowedOrigins } from '@rbee/shared-config'

const allowedOrigins = getAllowedOrigins()
```

**5. Generate Rust constants:**

Create a build script that generates Rust constants from the TypeScript config:

```typescript
// frontend/packages/shared-config/scripts/generate-rust.ts

import { PORTS } from '../src/ports'
import fs from 'fs'

const rustCode = `
// AUTO-GENERATED from frontend/packages/shared-config/src/ports.ts
// DO NOT EDIT MANUALLY - Run 'pnpm generate:rust' to update

pub const QUEEN_DEV_PORT: u16 = ${PORTS.queen.dev};
pub const QUEEN_PROD_PORT: u16 = ${PORTS.queen.prod};
pub const HIVE_DEV_PORT: u16 = ${PORTS.hive.dev};
pub const HIVE_PROD_PORT: u16 = ${PORTS.hive.prod};
pub const WORKER_DEV_PORT: u16 = ${PORTS.worker.dev};
pub const WORKER_PROD_PORT: u16 = ${PORTS.worker.prod};
`

fs.writeFileSync('../../shared-constants.rs', rustCode)
```

**6. Use in Rust build.rs:**

```rust
// bin/10_queen_rbee/build.rs
include!("../../shared-constants.rs");

let vite_dev_running = std::net::TcpStream::connect(
    format!("127.0.0.1:{}", QUEEN_DEV_PORT)
).is_ok();
```

---

## Implementation Checklist

### Phase 1: Create Shared Packages

- [ ] Create `@rbee/narration-client` package
  - [ ] Extract types from queen narrationBridge
  - [ ] Create config with all service ports
  - [ ] Create parser for SSE lines
  - [ ] Create bridge for postMessage
  - [ ] Add to pnpm workspace
  - [ ] Build and test

- [ ] Create `@rbee/iframe-bridge` package
  - [ ] Create message types
  - [ ] Create origin validator
  - [ ] Create sender/receiver
  - [ ] Add to pnpm workspace
  - [ ] Build and test

- [ ] Create `@rbee/dev-utils` package
  - [ ] Create environment detection
  - [ ] Create port utilities
  - [ ] Create logging helpers
  - [ ] Add to pnpm workspace
  - [ ] Build and test

- [ ] Create `@rbee/shared-config` package
  - [ ] Create ports.ts (single source of truth)
  - [ ] Create Rust code generator
  - [ ] Add to pnpm workspace
  - [ ] Generate Rust constants
  - [ ] Build and test

### Phase 2: Migrate Queen UI

- [ ] Update Queen UI to use `@rbee/narration-client`
  - [ ] Replace narrationBridge with package
  - [ ] Update imports
  - [ ] Test narration flow
  - [ ] Remove old code

- [ ] Update Queen UI to use `@rbee/dev-utils`
  - [ ] Replace environment detection
  - [ ] Use logging helpers
  - [ ] Test startup logs

- [ ] Update Keeper to use `@rbee/shared-config`
  - [ ] Replace hardcoded ports
  - [ ] Use getIframeUrl()
  - [ ] Use getAllowedOrigins()
  - [ ] Test iframe loading

### Phase 3: Implement Hive UI

- [ ] Create Hive UI packages (mirror Queen structure)
- [ ] Use `@rbee/narration-client` from the start
- [ ] Use `@rbee/iframe-bridge` for all communication
- [ ] Use `@rbee/dev-utils` for environment detection
- [ ] Add Hive ports to `@rbee/shared-config`
- [ ] Update Keeper to load Hive iframe
- [ ] Test narration flow

### Phase 4: Implement Worker UI

- [ ] Same as Hive, but for Worker
- [ ] Reuse ALL shared packages
- [ ] No code duplication!

### Phase 5: Documentation

- [ ] Update PORT_CONFIGURATION.md with all services
- [ ] Document shared packages in README
- [ ] Create migration guide for future services
- [ ] Add architecture diagrams

---

## Critical Rules for Future Teams

### 🚨 RULE 1: Never Hardcode Ports

**❌ WRONG:**
```typescript
const url = "http://localhost:7834"
```

**✅ CORRECT:**
```typescript
import { getIframeUrl } from '@rbee/shared-config'
const url = getIframeUrl('queen', isDev)
```

### 🚨 RULE 2: Always Use Shared Packages

**❌ WRONG:**
```typescript
// Copy-paste narrationBridge.ts to hive UI
```

**✅ CORRECT:**
```typescript
import { createStreamHandler, SERVICES } from '@rbee/narration-client'
const handler = createStreamHandler(SERVICES.hive)
```

### 🚨 RULE 3: Update ALL Locations When Adding Service

When adding a new service (e.g., "scheduler"), update:

1. ✅ `frontend/packages/shared-config/src/ports.ts` - Add ports
2. ✅ `PORT_CONFIGURATION.md` - Document ports
3. ✅ `@rbee/narration-client/src/config.ts` - Add service config
4. ✅ Run `pnpm generate:rust` - Update Rust constants
5. ✅ Update Keeper iframe pages - Add new page
6. ✅ Update Keeper message listener - Will auto-include from getAllowedOrigins()

### 🚨 RULE 4: Test Both Dev and Prod Modes

**Always test:**
- [ ] Dev mode: Vite dev server, hot reload works
- [ ] Prod mode: Embedded files, correct ports
- [ ] Narration flows in both modes
- [ ] Console logs show correct mode
- [ ] No hardcoded ports anywhere

---

## Benefits of This Architecture

### ✅ DRY (Don't Repeat Yourself)
- Write narration logic once, use everywhere
- Write iframe bridge once, use everywhere
- Write environment detection once, use everywhere

### ✅ Type Safety
- Shared types across all UIs
- TypeScript catches port mismatches
- No magic strings

### ✅ Single Source of Truth
- All ports in one file
- Generate Rust constants from TypeScript
- Update once, applies everywhere

### ✅ Easy to Add New Services
- Import shared packages
- Add ports to config
- Done!

### ✅ Maintainability
- Fix bugs in one place
- Add features in one place
- Easy to understand

---

## Example: Adding a New Service

Let's say we want to add a "Scheduler" service:

**1. Add to shared-config:**
```typescript
// frontend/packages/shared-config/src/ports.ts
export const PORTS = {
  // ... existing services
  scheduler: {
    dev: 7838,
    prod: 7839,
    backend: 7839,
  },
}
```

**2. Add to narration-client:**
```typescript
// @rbee/narration-client/src/config.ts
export const SERVICES = {
  // ... existing services
  scheduler: {
    name: 'scheduler',
    devPort: 7838,
    prodPort: 7839,
    keeperDevPort: 5173,
    keeperProdOrigin: '*',
  },
}
```

**3. Generate Rust constants:**
```bash
cd frontend/packages/shared-config
pnpm generate:rust
```

**4. Use in Scheduler UI:**
```typescript
import { createStreamHandler, SERVICES } from '@rbee/narration-client'
const handler = createStreamHandler(SERVICES.scheduler)
```

**5. Use in Keeper:**
```typescript
import { getIframeUrl } from '@rbee/shared-config'
const schedulerUrl = getIframeUrl('scheduler', isDev)
```

**Done!** No code duplication, all type-safe, all consistent.

---

## Summary

**TEAM-350 Recommendations:**

1. **Create 4 shared packages:**
   - `@rbee/narration-client` - Narration logic
   - `@rbee/iframe-bridge` - iframe communication
   - `@rbee/dev-utils` - Environment utilities
   - `@rbee/shared-config` - Port configuration (single source of truth)

2. **Migrate Queen UI to use shared packages**

3. **Implement Hive/Worker UIs using shared packages from day 1**

4. **Never hardcode ports** - Always use shared config

5. **Test both dev and prod modes** - Always

**This architecture will save weeks of development time and prevent countless bugs!**

---

**TEAM-350 Final Recommendations** - Build it right, build it once! 🚀
