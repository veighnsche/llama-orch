# TEAM-351: Testing Guide for Steps 2, 3, 4

**Based on:** Step 1 test patterns  
**Target:** Write tests for narration-client, iframe-bridge, dev-utils

---

## Step 2: @rbee/narration-client Tests

### File 1: `src/parser.test.ts` (~35 tests)

```typescript
import { describe, it, expect, beforeEach } from 'vitest'
import {
  parseNarrationLine,
  getParseStats,
  resetParseStats,
} from './parser'

describe('@rbee/narration-client - parser', () => {
  beforeEach(() => {
    resetParseStats()
  })

  describe('parseNarrationLine()', () => {
    // Valid JSON parsing (5 tests)
    it('should parse valid narration event')
    it('should parse event with data: prefix')
    it('should parse event without data: prefix')
    it('should handle formatted field')
    it('should handle optional fields')

    // [DONE] marker handling (3 tests)
    it('should skip [DONE] marker')
    it('should skip [DONE] with whitespace')
    it('should not count [DONE] as error')

    // Empty/whitespace handling (4 tests)
    it('should skip empty string')
    it('should skip whitespace-only line')
    it('should skip empty data: line')
    it('should handle null/undefined')

    // SSE format handling (5 tests)
    it('should skip SSE comment lines (:)')
    it('should skip event: lines')
    it('should skip id: lines')
    it('should handle multi-line SSE')
    it('should strip data: prefix correctly')

    // Validation (6 tests)
    it('should validate event structure by default')
    it('should reject event without actor')
    it('should reject event without action')
    it('should reject event without human')
    it('should skip validation when disabled')
    it('should handle malformed JSON')

    // Error handling (4 tests)
    it('should handle JSON parse errors')
    it('should log warnings by default')
    it('should suppress warnings when silent')
    it('should not throw on invalid input')

    // Statistics (8 tests)
    it('should track total lines')
    it('should track successful parses')
    it('should track failed parses')
    it('should track done markers')
    it('should track empty lines')
    it('should reset stats correctly')
    it('should return readonly stats')
    it('should increment stats correctly')
  })
})
```

### File 2: `src/config.test.ts` (~15 tests)

```typescript
import { describe, it, expect } from 'vitest'
import { SERVICES, getParentOrigin, isValidServiceConfig } from './config'
import { PORTS } from '@rbee/shared-config'

describe('@rbee/narration-client - config', () => {
  describe('SERVICES constant', () => {
    // Structure tests (4 tests)
    it('should have queen config')
    it('should have hive config')
    it('should have worker config')
    it('should import ports from shared-config')
  })

  describe('getParentOrigin()', () => {
    // Mock window.location.port for these tests
    // Dev ports (3 tests)
    it('should return keeper dev for queen dev port')
    it('should return keeper dev for hive dev port')
    it('should return keeper dev for worker dev port')

    // Prod ports (3 tests)
    it('should return wildcard for queen prod port')
    it('should return wildcard for hive prod port')
    it('should return wildcard for worker prod port')

    // Edge cases (2 tests)
    it('should handle missing port (default 80)')
    it('should handle unknown port')
  })

  describe('isValidServiceConfig()', () => {
    // Validation tests (3 tests)
    it('should validate correct config')
    it('should reject invalid config')
    it('should reject missing fields')
  })
})
```

### File 3: `src/bridge.test.ts` (~10 tests)

```typescript
import { describe, it, expect, vi } from 'vitest'
import { sendToParent } from './bridge'

describe('@rbee/narration-client - bridge', () => {
  describe('sendToParent()', () => {
    // Mock window.parent.postMessage
    // Basic functionality (3 tests)
    it('should send message to parent')
    it('should use correct origin')
    it('should include timestamp')

    // Edge cases (4 tests)
    it('should handle no parent window')
    it('should handle postMessage errors')
    it('should not throw on failure')
    it('should log debug info')

    // Service config (3 tests)
    it('should use service dev port')
    it('should use service prod port')
    it('should use keeper dev origin')
  })
})
```

**Total Step 2:** ~60 tests

---

## Step 3: @rbee/iframe-bridge Tests

### File 1: `src/validator.test.ts` (~30 tests)

```typescript
import { describe, it, expect } from 'vitest'
import {
  isValidOriginFormat,
  isLocalhostOrigin,
  validateOrigin,
  isValidOriginConfig,
  createOriginValidator,
} from './validator'

describe('@rbee/iframe-bridge - validator', () => {
  describe('isValidOriginFormat()', () => {
    // Valid formats (5 tests)
    it('should accept wildcard')
    it('should accept http://localhost:3000')
    it('should accept https://example.com')
    it('should accept IPv4 origins')
    it('should accept IPv6 origins')

    // Invalid formats (6 tests)
    it('should reject null/undefined')
    it('should reject empty string')
    it('should reject without protocol')
    it('should reject with path')
    it('should reject with query')
    it('should reject with hash')
  })

  describe('isLocalhostOrigin()', () => {
    // Localhost detection (5 tests)
    it('should detect localhost')
    it('should detect 127.0.0.1')
    it('should detect [::1]')
    it('should reject non-localhost')
    it('should reject wildcard')
  })

  describe('validateOrigin()', () => {
    // Wildcard mode (3 tests)
    it('should allow all in wildcard mode')
    it('should reject in strict mode with wildcard')
    it('should validate format even with wildcard')

    // Exact match (4 tests)
    it('should allow exact match')
    it('should reject non-match')
    it('should be case-sensitive')
    it('should require exact protocol')

    // Localhost mode (3 tests)
    it('should allow any localhost port when enabled')
    it('should still require one localhost in allowed list')
    it('should reject localhost when disabled')

    // Edge cases (4 tests)
    it('should reject empty allowed list')
    it('should reject invalid config')
    it('should handle null origin')
    it('should handle malformed origin')
  })

  describe('createOriginValidator()', () => {
    // Factory tests (2 tests)
    it('should create validator function')
    it('should throw on invalid config')
  })
})
```

### File 2: `src/sender.test.ts` (~12 tests)

```typescript
import { describe, it, expect, vi } from 'vitest'
import { createMessageSender } from './sender'

describe('@rbee/iframe-bridge - sender', () => {
  describe('createMessageSender()', () => {
    // Basic functionality (4 tests)
    it('should create sender function')
    it('should send message to parent')
    it('should use target origin')
    it('should handle debug logging')

    // Error handling (4 tests)
    it('should handle no parent window')
    it('should handle postMessage errors')
    it('should not throw on failure')
    it('should log errors when debug enabled')

    // Edge cases (4 tests)
    it('should handle SSR (no window)')
    it('should handle same window (no parent)')
    it('should serialize message correctly')
    it('should handle complex message objects')
  })
})
```

### File 3: `src/receiver.test.ts` (~15 tests)

```typescript
import { describe, it, expect, vi } from 'vitest'
import { createMessageReceiver } from './receiver'

describe('@rbee/iframe-bridge - receiver', () => {
  describe('createMessageReceiver()', () => {
    // Basic functionality (5 tests)
    it('should create receiver and add listener')
    it('should call onMessage for valid events')
    it('should validate origin')
    it('should handle debug logging')
    it('should return cleanup function')

    // Origin validation (4 tests)
    it('should reject invalid origins')
    it('should accept allowed origins')
    it('should handle wildcard')
    it('should log rejected origins when debug enabled')

    // Message validation (3 tests)
    it('should require message.type')
    it('should reject non-object messages')
    it('should handle malformed events')

    // Cleanup (3 tests)
    it('should remove listener on cleanup')
    it('should not error on double cleanup')
    it('should handle cleanup before messages')
  })
})
```

**Total Step 3:** ~57 tests

---

## Step 4: @rbee/dev-utils Tests

### File 1: `src/environment.test.ts` (~40 tests)

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  isDevelopment,
  isProduction,
  isSSR,
  getCurrentPort,
  getProtocol,
  getHostname,
  validatePort,
  isRunningOnPort,
  isLocalhost,
  isHTTPS,
  getEnvironmentInfo,
} from './environment'

describe('@rbee/dev-utils - environment', () => {
  describe('isDevelopment()', () => {
    // Environment detection (2 tests)
    it('should detect dev mode')
    it('should return false in prod')
  })

  describe('isProduction()', () => {
    // Environment detection (2 tests)
    it('should detect prod mode')
    it('should return false in dev')
  })

  describe('isSSR()', () => {
    // SSR detection (2 tests)
    it('should return true when no window')
    it('should return false in browser')
  })

  describe('getCurrentPort()', () => {
    // Port detection (6 tests)
    it('should return 0 in SSR')
    it('should parse port from window.location')
    it('should default to 80 for HTTP')
    it('should default to 443 for HTTPS')
    it('should handle invalid port string')
    it('should validate port range')
  })

  describe('getProtocol()', () => {
    // Protocol detection (4 tests)
    it('should detect http')
    it('should detect https')
    it('should return unknown in SSR')
    it('should return unknown for other protocols')
  })

  describe('getHostname()', () => {
    // Hostname detection (3 tests)
    it('should return hostname')
    it('should return empty in SSR')
    it('should handle IPv6')
  })

  describe('validatePort()', () => {
    // Port validation (7 tests)
    it('should validate valid port')
    it('should reject non-number')
    it('should reject NaN')
    it('should reject port < 1')
    it('should reject port > 65535')
    it('should return error message')
    it('should return port in result')
  })

  describe('isRunningOnPort()', () => {
    // Port checking (4 tests)
    it('should return true for current port')
    it('should return false for different port')
    it('should validate port first')
    it('should log warning for invalid port')
  })

  describe('isLocalhost()', () => {
    // Localhost detection (4 tests)
    it('should detect localhost')
    it('should detect 127.0.0.1')
    it('should detect [::1]')
    it('should return false for other hosts')
  })

  describe('isHTTPS()', () => {
    // HTTPS detection (2 tests)
    it('should return true for HTTPS')
    it('should return false for HTTP')
  })

  describe('getEnvironmentInfo()', () => {
    // Comprehensive info (4 tests)
    it('should return all environment data')
    it('should include port and protocol')
    it('should include hostname and URL')
    it('should handle SSR correctly')
  })
})
```

### File 2: `src/logging.test.ts` (~20 tests)

```typescript
import { describe, it, expect, vi } from 'vitest'
import { logStartupMode, createLogger } from './logging'

describe('@rbee/dev-utils - logging', () => {
  describe('logStartupMode()', () => {
    // Basic logging (4 tests)
    it('should log dev mode with emoji')
    it('should log prod mode with emoji')
    it('should include service name')
    it('should include port when provided')

    // Edge cases (3 tests)
    it('should handle missing port')
    it('should handle long service names')
    it('should not throw on console errors')
  })

  describe('createLogger()', () => {
    // Logger creation (5 tests)
    it('should create logger with prefix')
    it('should support log levels')
    it('should include timestamps')
    it('should handle debug mode')
    it('should return logger object')

    // Log methods (5 tests)
    it('should log info messages')
    it('should log warn messages')
    it('should log error messages')
    it('should log debug when enabled')
    it('should skip debug when disabled')

    // Edge cases (3 tests)
    it('should handle missing console')
    it('should serialize objects')
    it('should handle errors in logging')
  })
})
```

**Total Step 4:** ~60 tests

---

## Testing Patterns from Step 1

### Pattern 1: Test Structure
```typescript
describe('Package - Module', () => {
  describe('functionName()', () => {
    it('should do X')
    it('should handle edge case Y')
  })
})
```

### Pattern 2: Edge Cases
- null/undefined inputs
- Empty strings/arrays
- Invalid formats
- SSR scenarios (no window)
- Error conditions

### Pattern 3: Mocking
```typescript
// Mock window.location
vi.stubGlobal('window', {
  location: {
    port: '7834',
    protocol: 'http:',
    hostname: 'localhost',
  },
  parent: {
    postMessage: vi.fn(),
  },
})
```

### Pattern 4: Statistics/State
```typescript
beforeEach(() => {
  resetStats() // Clean state between tests
})
```

---

## Execution Plan

**TEAM-351: Follow this order:**

1. **Step 2 (narration-client):**
   - Create `src/parser.test.ts` (35 tests)
   - Create `src/config.test.ts` (15 tests)
   - Create `src/bridge.test.ts` (10 tests)
   - Run: `pnpm test`
   - Target: 60/60 passing

2. **Step 3 (iframe-bridge):**
   - Create `src/validator.test.ts` (30 tests)
   - Create `src/sender.test.ts` (12 tests)
   - Create `src/receiver.test.ts` (15 tests)
   - Run: `pnpm test`
   - Target: 57/57 passing

3. **Step 4 (dev-utils):**
   - Create `src/environment.test.ts` (40 tests)
   - Create `src/logging.test.ts` (20 tests)
   - Run: `pnpm test`
   - Target: 60/60 passing

**Total: ~237 tests across all packages**

---

## Key Reminders

1. **Follow Step 1 patterns** - Same structure, same style
2. **Test edge cases** - null, empty, invalid, SSR
3. **Mock browser APIs** - window, postMessage, console
4. **Keep tests fast** - <100ms total per package
5. **Use beforeEach** - Clean state between tests
6. **Test validation** - Type guards, config validation
7. **Test error handling** - Don't throw, log warnings
8. **Use descriptive names** - "should X when Y"

---

**TEAM-351: Use this guide to write tests for steps 2, 3, 4!**
