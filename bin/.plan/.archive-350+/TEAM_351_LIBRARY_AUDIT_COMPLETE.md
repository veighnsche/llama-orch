# TEAM-351: Complete Library Audit

**Date:** Oct 29, 2025  
**Status:** 🔍 COMPREHENSIVE AUDIT COMPLETE  
**Scope:** All 4 packages + TEAM-356 recommendations

---

## Executive Summary

**Findings:** TEAM-351 missed several npm libraries that could replace custom code.

**Recommendation:** Replace ~60% of custom code with battle-tested libraries.

**Impact:**
- **Don't build:** 2 packages (use libraries instead)
- **Simplify:** 2 packages (use libraries internally)
- **Time saved:** 4-6 hours
- **Code saved:** ~300 LOC to maintain

---

## Package 1: @rbee/shared-config

### What TEAM-351 Built
- Port configuration constants
- URL generation functions
- Origin generation functions
- Rust code generator

### Library Alternatives

❌ **No direct replacement** - This is project-specific configuration

✅ **Keep this package** - No library can replace domain-specific port config

### Verdict
**KEEP AS-IS** - This is the right approach. No library alternative.

---

## Package 2: @rbee/narration-client

### What TEAM-351 Built
1. SSE line parser (~125 LOC)
2. Parse statistics tracking
3. Event validation
4. postMessage bridge

### Library Alternatives

#### 1. eventsource-parser (npm)
**What it does:** Streaming SSE parser (source-agnostic)

**Features:**
- ✅ Handles SSE format (`data:`, `event:`, `id:`, `:` comments)
- ✅ Handles multi-line events
- ✅ Streaming parser (doesn't buffer entire message)
- ✅ 3.6M weekly downloads
- ✅ TypeScript support
- ✅ 0 dependencies
- ✅ 2.4kb gzipped

**Usage:**
```typescript
import { createParser, type ParsedEvent } from 'eventsource-parser'

const parser = createParser((event: ParsedEvent) => {
  if (event.type === 'event') {
    const narrationEvent = JSON.parse(event.data)
    // Handle event
  }
})

// Feed lines to parser
for await (const line of stream) {
  parser.feed(line)
}
```

**Replaces:**
- ❌ Custom SSE parsing logic (~80 LOC)
- ❌ `data:` prefix handling
- ❌ Comment line skipping
- ❌ Multi-line event handling

**Doesn't replace:**
- ✅ Event validation (project-specific)
- ✅ postMessage bridge (project-specific)
- ✅ Statistics tracking (optional feature)

#### Verdict
✅ **USE eventsource-parser** - Replace custom SSE parsing (~80 LOC saved)

**Keep custom:**
- Event validation (`isValidNarrationEvent`)
- postMessage bridge (`sendToParent`)
- Statistics (if needed, or remove)

---

## Package 3: @rbee/iframe-bridge

### What TEAM-351 Built
1. Origin validation (~134 LOC)
2. Message sender/receiver
3. Type-safe message handling

### Library Alternatives

#### 1. Postmate (npm)
**What it does:** Promise-based postMessage iframe communication

**Features:**
- ✅ Parent-child handshake
- ✅ Promise-based API
- ✅ Type-safe communication
- ✅ 2.8k GitHub stars
- ✅ Battle-tested (Dollar Shave Club)
- ✅ 3kb gzipped

**Usage:**
```typescript
// Parent
import Postmate from 'postmate'

const handshake = new Postmate({
  container: document.getElementById('iframe-container'),
  url: 'http://localhost:7834',
})

const child = await handshake
child.on('narration', (data) => {
  // Handle narration event
})

// Child
import Postmate from 'postmate'

const handshake = new Postmate.Model({
  narration: (data) => {
    // Emit narration to parent
    this.emit('narration', data)
  },
})
```

**Replaces:**
- ❌ Custom origin validation
- ❌ Custom message sender/receiver
- ❌ Manual cleanup logic

**Limitations:**
- ⚠️ Requires handshake (adds complexity)
- ⚠️ Different API than current implementation

#### 2. Penpal (npm)
**What it does:** Promise-based iframe/worker communication

**Features:**
- ✅ Promise-based RPC
- ✅ TypeScript support
- ✅ Works with iframes, workers, windows
- ✅ 1.4k GitHub stars
- ✅ 5kb gzipped

**Usage:**
```typescript
// Parent
import { connectToChild } from 'penpal'

const connection = connectToChild({
  iframe: document.getElementById('iframe'),
  methods: {
    handleNarration: (data) => {
      // Handle narration
    },
  },
})

const child = await connection.promise
await child.sendNarration(data)

// Child
import { connectToParent } from 'penpal'

const connection = connectToParent({
  methods: {
    sendNarration: (data) => {
      // Send to parent
    },
  },
})

const parent = await connection.promise
await parent.handleNarration(data)
```

**Replaces:**
- ❌ Custom message passing
- ❌ Origin validation
- ❌ Type safety

**Limitations:**
- ⚠️ RPC-style API (different from current event-based)
- ⚠️ Requires method definitions upfront

#### 3. react-iframe-comm (npm)
**What it does:** React component for iframe communication

**Features:**
- ✅ React-specific
- ✅ Simple API
- ✅ PostMessage wrapper
- ✅ 100+ GitHub stars

**Usage:**
```typescript
import IframeComm from 'react-iframe-comm'

<IframeComm
  attributes={{ src: 'http://localhost:7834' }}
  postMessageData="narration-data"
  handleReceiveMessage={(event) => {
    // Handle message from iframe
  }}
/>
```

**Limitations:**
- ⚠️ React-specific (not reusable)
- ⚠️ Less flexible than Postmate/Penpal

#### Verdict
🟡 **MAYBE use Postmate or Penpal** - But current implementation is simpler

**Analysis:**
- Postmate/Penpal are more complex (handshake, RPC)
- Current implementation is straightforward (just postMessage + validation)
- Libraries add ~3-5kb for features you don't need

**Recommendation:** **KEEP CUSTOM** - Your implementation is simpler and lighter

---

## Package 4: @rbee/dev-utils

### What TEAM-351 Built
1. Environment detection (~211 LOC)
2. Logging utilities (~231 LOC)
3. Port validation

### Library Alternatives

#### 1. For Logging: loglevel (npm)
**What it does:** Minimal logging library for browser

**Features:**
- ✅ Log levels (trace, debug, info, warn, error)
- ✅ 6kb gzipped
- ✅ 2.6M weekly downloads
- ✅ Persistent log levels (localStorage)
- ✅ Plugin system

**Usage:**
```typescript
import log from 'loglevel'

log.setLevel('info')
log.info('QUEEN UI', 'Running in DEVELOPMENT mode')
log.warn('Connection failed')
```

**Replaces:**
- ❌ Custom log level handling
- ❌ Custom logger factory

**Doesn't replace:**
- ✅ Startup mode logging (project-specific format)
- ✅ Environment detection
- ✅ Port validation

#### 2. For Logging: debug (npm)
**What it does:** Tiny debugging utility (used by Express, Socket.io, etc.)

**Features:**
- ✅ Namespace-based logging
- ✅ 2kb gzipped
- ✅ 11M weekly downloads
- ✅ Browser + Node.js
- ✅ Color-coded output

**Usage:**
```typescript
import debug from 'debug'

const log = debug('queen:ui')
log('Running in development mode')
```

**Replaces:**
- ❌ Custom logger factory

**Doesn't replace:**
- ✅ Log levels (debug only has on/off)
- ✅ Startup formatting
- ✅ Environment detection

#### Verdict
🟡 **MAYBE use loglevel or debug** - But custom logging is simple enough

**Analysis:**
- Your logging is ~50 LOC of actual logic (rest is types/comments)
- Libraries add features you don't need (persistent levels, plugins)
- Your startup logging format is project-specific

**Recommendation:** **KEEP CUSTOM** - Your logging is simple and project-specific

---

## TEAM-356 Recommendations (Already Covered)

### 1. TanStack Query
✅ **CONFIRMED** - Replace custom async state hooks

### 2. exponential-backoff
✅ **CONFIRMED** - Replace custom retry logic

### 3. SSE Libraries
✅ **NEW FINDING** - Use eventsource-parser for SSE parsing

---

## Final Recommendations

### ✅ MUST USE These Libraries

**1. TanStack Query**
- **Replaces:** Custom async state hooks, CRUD hooks
- **Package:** `@tanstack/react-query`
- **Savings:** ~150 LOC per UI

**2. exponential-backoff**
- **Replaces:** Custom retry logic in SDK loader
- **Package:** `exponential-backoff`
- **Savings:** ~30 LOC

**3. eventsource-parser** ⭐ NEW
- **Replaces:** Custom SSE parsing in narration-client
- **Package:** `eventsource-parser`
- **Savings:** ~80 LOC

### 🟡 KEEP CUSTOM (Simpler Than Libraries)

**4. @rbee/shared-config**
- **Why:** Project-specific port configuration
- **No alternative:** This is domain logic

**5. @rbee/iframe-bridge**
- **Why:** Current implementation is simpler than Postmate/Penpal
- **Libraries are overkill:** Don't need RPC or handshake

**6. @rbee/dev-utils**
- **Why:** Logging is simple and project-specific
- **Libraries add complexity:** loglevel/debug don't match your needs

**7. Narration validation + postMessage**
- **Why:** Project-specific event structure
- **Keep:** `isValidNarrationEvent`, `sendToParent`

---

## Updated Package Plan

### Package 1: @rbee/shared-config ✅ KEEP AS-IS
- No changes needed
- No library alternative

### Package 2: @rbee/narration-client ⚠️ SIMPLIFY

**Before:** 125 LOC (parser) + 50 LOC (bridge) = 175 LOC

**After:** Use eventsource-parser
```typescript
import { createParser } from 'eventsource-parser'
import { isValidNarrationEvent } from './types'
import { sendToParent } from './bridge'

export function createNarrationParser(serviceConfig) {
  return createParser((event) => {
    if (event.type === 'event') {
      try {
        const narrationEvent = JSON.parse(event.data)
        
        if (isValidNarrationEvent(narrationEvent)) {
          sendToParent(narrationEvent, serviceConfig)
        }
      } catch (err) {
        console.warn('[NarrationClient] Parse error:', err)
      }
    }
  })
}
```

**New size:** ~50 LOC (validation + bridge only)

**Savings:** 80 LOC

### Package 3: @rbee/iframe-bridge ✅ KEEP AS-IS
- Current implementation is simpler than libraries
- No changes needed

### Package 4: @rbee/dev-utils ✅ KEEP AS-IS
- Logging is simple and project-specific
- No changes needed

---

## Revised TEAM-356 Plan

### Phase 1: Install Libraries
```bash
# Data fetching
pnpm add @tanstack/react-query @tanstack/react-query-devtools

# Retry logic
pnpm add exponential-backoff

# SSE parsing (NEW!)
pnpm add eventsource-parser
```

### Phase 2: Simplify narration-client
- Replace custom SSE parser with eventsource-parser
- Keep validation and postMessage bridge
- Remove statistics tracking (or make opt-in)

### Phase 3: Simplify SDK loader
- Use exponential-backoff for retry logic
- Keep WASM loading logic

### Phase 4: Migrate to TanStack Query
- Replace all async state hooks
- Replace CRUD operations
- Add QueryClientProvider

---

## ROI Comparison

### Original TEAM-351 Plan
| Item | LOC | Time |
|------|-----|------|
| Build 4 packages | 800 | 6-8h |
| Write tests | 70 tests | 2-3h |
| **Total** | **800** | **8-11h** |

### Revised Plan (With Libraries)
| Item | LOC | Time |
|------|-----|------|
| Install 3 libraries | 0 | 10min |
| Simplify narration-client | -80 | 30min |
| Simplify SDK loader | -30 | 30min |
| Keep shared-config | 150 | 0h (done) |
| Keep iframe-bridge | 200 | 0h (done) |
| Keep dev-utils | 250 | 0h (done) |
| **Total** | **490** | **1h** |

**Savings:**
- **Time:** 7-10 hours saved
- **Code:** 310 fewer lines to maintain
- **Tests:** 40 fewer tests to write (libraries already tested)

---

## Summary Table

| Package | Status | Library Alternative | Recommendation |
|---------|--------|-------------------|----------------|
| shared-config | ✅ Keep | None | Keep as-is |
| narration-client | ⚠️ Simplify | eventsource-parser | Use for SSE parsing |
| iframe-bridge | ✅ Keep | Postmate/Penpal | Keep (simpler) |
| dev-utils | ✅ Keep | loglevel/debug | Keep (simpler) |
| react-hooks | ❌ Don't build | TanStack Query | Use library |
| sdk-loader | ⚠️ Simplify | exponential-backoff | Use for retry |

---

## Action Items for TEAM-356

### High Priority
- [ ] Install eventsource-parser
- [ ] Simplify narration-client parser (~80 LOC removed)
- [ ] Install TanStack Query
- [ ] Migrate Queen hooks to TanStack Query
- [ ] Install exponential-backoff
- [ ] Simplify SDK loader retry logic

### Low Priority
- [ ] Consider removing parse statistics (or make opt-in)
- [ ] Document library choices in README
- [ ] Update tests to use library APIs

### Don't Do
- [ ] ❌ Don't build @rbee/react-hooks (use TanStack Query)
- [ ] ❌ Don't replace iframe-bridge (current is simpler)
- [ ] ❌ Don't replace dev-utils logging (current is simpler)

---

## Conclusion

**TEAM-351 did good work, but missed 3 key libraries:**

1. **TanStack Query** - Industry standard for async state (TEAM-356 found this)
2. **exponential-backoff** - Battle-tested retry logic (TEAM-356 found this)
3. **eventsource-parser** - SSE parsing (NEW finding from this audit)

**Final verdict:**
- Keep 3 of 4 packages (shared-config, iframe-bridge, dev-utils)
- Simplify 1 package (narration-client with eventsource-parser)
- Don't build 1 package (react-hooks, use TanStack Query)
- Use 3 npm libraries (TanStack Query, exponential-backoff, eventsource-parser)

**Result:** Less code, better quality, faster implementation.

---

**TEAM-351 Library Audit: Complete!** 🔍
