# @rbee/narration-client

Shared narration client for handling backend SSE narration events and forwarding to parent window.

**TEAM-351:** Bug fixes - Validation, type safety, production logging, monitoring  
**TEAM-351 CORRECTION:** Ports imported from `@rbee/shared-config` (no duplication)

## Installation

```bash
pnpm add @rbee/narration-client
```

**Dependencies:**
- `@rbee/shared-config` - Port configuration (single source of truth)

## Features

✅ **Validation** - Runtime validation of events and configs  
✅ **Type Safety** - Full TypeScript type inference  
✅ **SSE Parsing** - Handles all SSE formats (data:, event:, id:, comments)  
✅ **Production Ready** - Conditional logging, error handling  
✅ **Monitoring** - Parse statistics tracking  
✅ **Retry Logic** - Optional retry for failed postMessage  
✅ **Edge Cases** - Empty lines, malformed JSON, missing fields  
✅ **Protocol Version** - Future-proof message format

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

### With Options

```typescript
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

const handleNarration = createStreamHandler(
  SERVICES.queen,
  (event) => {
    console.log('Local event:', event)
  },
  {
    debug: true,      // Enable debug logging
    silent: false,    // Show parse warnings
    validate: true,   // Validate event structure
    retry: true,      // Retry failed postMessage
  }
)
```

### Manual Parsing

```typescript
import { parseNarrationLine } from '@rbee/narration-client'

// Basic parsing
const event = parseNarrationLine('data: {"actor":"queen","action":"test","human":"Testing"}')
if (event) {
  console.log(event)
}

// With options
const event2 = parseNarrationLine(line, {
  silent: true,     // Suppress warnings
  validate: false,  // Skip validation (faster)
})
```

### Monitoring

```typescript
import { getParseStats, resetParseStats } from '@rbee/narration-client'

// Get parse statistics
const stats = getParseStats()
console.log('Parse stats:', {
  total: stats.total,
  success: stats.success,
  failed: stats.failed,
  successRate: `${(stats.success / stats.total * 100).toFixed(1)}%`,
})

// Reset statistics
resetParseStats()
```

### Direct Sending

```typescript
import { sendToParent, SERVICES } from '@rbee/narration-client'

const event = {
  actor: 'queen',
  action: 'test',
  human: 'Test message',
}

const success = sendToParent(event, SERVICES.queen, {
  debug: true,
  retry: true,
})

console.log('Sent:', success)
```

## Services

Type-safe service configurations:

```typescript
import { SERVICES } from '@rbee/narration-client'

SERVICES.queen  // Queen UI (port 7834 dev, 7833 prod)
SERVICES.hive   // Hive UI (port 7836 dev, 7835 prod)
SERVICES.worker // Worker UI (port 7837 dev, 8080 prod)
```

## Validation

All events are validated for required fields:

```typescript
// Valid event (all required fields present)
{
  actor: 'queen',    // ✅ Required
  action: 'start',   // ✅ Required
  human: 'Starting', // ✅ Required
  job_id: '123',     // ✅ Optional
}

// Invalid event (missing required fields)
{
  actor: 'queen',
  action: 'start',
  // ❌ Missing 'human' field
}
```

## SSE Format Support

Handles all SSE line types:

```typescript
// Data lines (parsed)
parseNarrationLine('data: {"actor":"queen",...}')

// Event lines (skipped)
parseNarrationLine('event: message')

// ID lines (skipped)
parseNarrationLine('id: 123')

// Comment lines (skipped)
parseNarrationLine(': this is a comment')

// [DONE] marker (skipped gracefully)
parseNarrationLine('[DONE]')

// Empty lines (skipped)
parseNarrationLine('')
parseNarrationLine('   ')
```

## Production Mode

Automatic production detection:

```typescript
// Development: Full logging
// Production: Minimal logging (performance)

// Override with options
createStreamHandler(SERVICES.queen, null, {
  debug: false,  // Force production mode
})
```

## Error Handling

Comprehensive error handling:

```typescript
// Parser errors (malformed JSON)
parseNarrationLine('invalid json')  // Returns null, logs warning

// Validation errors (missing fields)
parseNarrationLine('{"actor":"queen"}')  // Returns null, logs warning

// postMessage errors (caught and logged)
sendToParent(event, config)  // Returns false on error

// Local handler errors (caught and logged)
createStreamHandler(config, (event) => {
  throw new Error('Handler error')  // Caught, logged, doesn't crash
})
```

## Type Safety

Full TypeScript support:

```typescript
import type { BackendNarrationEvent, ServiceName } from '@rbee/narration-client'

// Type-safe service names
const service: ServiceName = 'queen'  // ✅ 'queen' | 'hive' | 'worker'

// Type-safe events
const event: BackendNarrationEvent = {
  actor: 'queen',
  action: 'start',
  human: 'Starting',
}
```

## Performance

- ✅ Minimal overhead in production (no debug logging)
- ✅ Efficient parsing (early returns for skipped lines)
- ✅ Optional validation (can disable for speed)
- ✅ Statistics tracking (negligible overhead)

## Compatibility

- ✅ Browser environments (uses import.meta.env)
- ✅ SSR compatible (checks for window)
- ✅ TypeScript 5.0+
- ✅ ES2020+
