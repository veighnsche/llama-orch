# @rbee/dev-utils

Development utilities for environment detection and logging.

**TEAM-351:** Bug fixes - Validation, SSR support, log levels, timestamps, types

## Installation

```bash
pnpm add @rbee/dev-utils
```

## Features

✅ **Environment Detection** - Dev/prod, SSR, protocol, hostname  
✅ **Port Validation** - Validate port numbers (1-65535)  
✅ **HTTPS Detection** - Detect HTTPS protocol  
✅ **Localhost Detection** - Check if running on localhost  
✅ **Startup Logging** - Consistent, configurable logging  
✅ **Log Levels** - Debug, info, warn, error with emojis  
✅ **Timestamps** - Optional timestamps on all logs  
✅ **Logger Factory** - Create prefixed loggers  
✅ **SSR Safe** - All functions work in SSR environments  
✅ **Type Safe** - Full TypeScript support

## Usage

### Startup Logging

```typescript
import { logStartupMode, isDevelopment, getCurrentPort } from '@rbee/dev-utils'

// Basic usage
logStartupMode('QUEEN UI', isDevelopment(), getCurrentPort())

// With options
logStartupMode('QUEEN UI', isDevelopment(), getCurrentPort(), {
  timestamp: true,
  showProtocol: true,
  showHostname: true,
})
```

**Output (dev mode):**
```
🔧 [QUEEN UI] Running in DEVELOPMENT mode
   - Vite dev server active (hot reload enabled)
   - Running on: http://localhost:7834
```

**Output (prod mode):**
```
🚀 [QUEEN UI] Running in PRODUCTION mode
   - Serving embedded static files
```

### Environment Detection

```typescript
import {
  isDevelopment,
  isProduction,
  isSSR,
  getCurrentPort,
  isRunningOnPort,
  isLocalhost,
  isHTTPS,
  getEnvironmentInfo,
} from '@rbee/dev-utils'

// Simple checks
if (isDevelopment()) {
  console.log('Dev mode')
}

if (isRunningOnPort(7834)) {
  console.log('Running on Queen dev port')
}

if (isLocalhost()) {
  console.log('Running on localhost')
}

if (isHTTPS()) {
  console.log('Using HTTPS')
}

// Get complete environment info
const env = getEnvironmentInfo()
console.log(env)
// {
//   isDev: true,
//   isProd: false,
//   isSSR: false,
//   port: 7834,
//   protocol: 'http',
//   hostname: 'localhost',
//   url: 'http://localhost:7834'
// }
```

### Port Validation

```typescript
import { validatePort, getCurrentPort } from '@rbee/dev-utils'

const validation = validatePort(7834)
if (validation.valid) {
  console.log('Port is valid:', validation.port)
} else {
  console.error('Invalid port:', validation.error)
}

// Get current port (with validation)
const port = getCurrentPort()
// Returns: 80 (HTTP), 443 (HTTPS), or specified port
```

### Generic Logging

```typescript
import { log } from '@rbee/dev-utils'

// Log with level
log('info', 'Application started')
log('warn', 'Deprecated API used')
log('error', 'Failed to load resource')

// With options
log('info', 'User logged in', {
  timestamp: true,
  prefix: 'AUTH',
  color: true,
})
// Output: [12:34:56] ℹ️ [AUTH] User logged in
```

### Logger Factory

```typescript
import { createLogger } from '@rbee/dev-utils'

const logger = createLogger('MyComponent', { timestamp: true })

logger.debug('Debugging info')
logger.info('Information')
logger.warn('Warning message')
logger.error('Error occurred')

// Output:
// [12:34:56] 🐛 [MyComponent] Debugging info
// [12:34:56] ℹ️ [MyComponent] Information
// [12:34:56] ⚠️ [MyComponent] Warning message
// [12:34:56] ❌ [MyComponent] Error occurred
```

### Environment Information Logging

```typescript
import { logEnvironmentInfo, getEnvironmentInfo } from '@rbee/dev-utils'

const env = getEnvironmentInfo()
logEnvironmentInfo('QUEEN UI', env, { timestamp: true })

// Output:
// [12:34:56] 🌍 [QUEEN UI] Environment Information:
//    - Mode: Development
//    - SSR: No
//    - Protocol: http
//    - Hostname: localhost
//    - Port: 7834
//    - URL: http://localhost:7834
```

## API Reference

### Environment Detection

```typescript
isDevelopment(): boolean
isProduction(): boolean
isSSR(): boolean
getCurrentPort(): number
getProtocol(): 'http' | 'https' | 'unknown'
getHostname(): string
isRunningOnPort(port: number): boolean
isLocalhost(): boolean
isHTTPS(): boolean
getEnvironmentInfo(): EnvironmentInfo
```

### Port Validation

```typescript
validatePort(port: number): PortValidation

interface PortValidation {
  valid: boolean
  port: number
  error?: string
}
```

### Logging

```typescript
log(level: LogLevel, message: string, options?: LogOptions): void
logStartupMode(serviceName: string, isDev: boolean, port?: number, options?: StartupLogOptions): void
logEnvironmentInfo(serviceName: string, envInfo: EnvironmentInfo, options?: LogOptions): void
createLogger(prefix: string, options?: LogOptions): Logger

type LogLevel = 'debug' | 'info' | 'warn' | 'error'

interface LogOptions {
  timestamp?: boolean
  level?: LogLevel
  prefix?: string
  color?: boolean
}

interface StartupLogOptions extends LogOptions {
  showUrl?: boolean
  showProtocol?: boolean
  showHostname?: boolean
}
```

## Types

```typescript
interface EnvironmentInfo {
  isDev: boolean
  isProd: boolean
  isSSR: boolean
  port: number
  protocol: 'http' | 'https' | 'unknown'
  hostname: string
  url: string
}

interface PortValidation {
  valid: boolean
  port: number
  error?: string
}

type LogLevel = 'debug' | 'info' | 'warn' | 'error'
```

## SSR Support

All functions are SSR-safe:

```typescript
// In SSR environment
isSSR() // true
getCurrentPort() // 0
getProtocol() // 'unknown'
getHostname() // ''
isLocalhost() // false
```

## Edge Cases

### Invalid Port Handling

```typescript
// NaN handling
validatePort(NaN) // { valid: false, error: 'Port is NaN' }

// Out of range
validatePort(99999) // { valid: false, error: 'Port must be between 1 and 65535' }

// getCurrentPort() with invalid port
// Returns 0 and logs warning
```

### HTTPS Default Port

```typescript
// On HTTPS without explicit port
getCurrentPort() // 443 (not 80)
```

### Empty Service Name

```typescript
logStartupMode('', true, 7834) // Logs warning, returns early
```

## Performance

- ✅ Minimal overhead (simple checks)
- ✅ No external dependencies
- ✅ Tree-shakeable (ES modules)
- ✅ TypeScript type checking (compile-time)

## Compatibility

- ✅ Browser environments
- ✅ SSR compatible
- ✅ TypeScript 5.0+
- ✅ ES2020+
