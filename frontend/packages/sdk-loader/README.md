# @rbee/sdk-loader

**TEAM-356:** Generic WASM/SDK loader with exponential backoff, retry logic, and singleflight pattern.

## Features

- ✅ Exponential backoff with jitter
- ✅ Configurable retry attempts
- ✅ Timeout handling
- ✅ Singleflight pattern (one load at a time)
- ✅ Export validation
- ✅ Environment guards (SSR, WebAssembly support)
- ✅ TypeScript support with strict mode
- ✅ Comprehensive test coverage (40 tests)

## Installation

```bash
pnpm add @rbee/sdk-loader
```

## Usage

### Basic Usage

```typescript
import { loadSDK } from '@rbee/sdk-loader'

const result = await loadSDK({
  packageName: '@rbee/queen-rbee-sdk',
  requiredExports: ['QueenClient', 'HeartbeatMonitor'],
  timeout: 15000,
  maxAttempts: 3,
})

const sdk = result.sdk
console.log(`Loaded in ${result.loadTime}ms after ${result.attempts} attempts`)
```

### Factory Pattern (Recommended)

```typescript
import { createSDKLoader } from '@rbee/sdk-loader'

const queenLoader = createSDKLoader({
  packageName: '@rbee/queen-rbee-sdk',
  requiredExports: ['QueenClient', 'HeartbeatMonitor', 'RhaiClient'],
  timeout: 15000,
  maxAttempts: 3,
})

// Load once (singleflight - multiple calls share same load)
const { sdk } = await queenLoader.loadOnce()

// Or load fresh each time
const { sdk: freshSDK } = await queenLoader.load()
```

### With WASM Initialization

```typescript
const { sdk } = await queenLoader.loadOnce({
  memory: new WebAssembly.Memory({ initial: 256 })
})
```

### Singleflight Pattern

The singleflight pattern ensures only one load operation happens at a time per package:

```typescript
// Multiple concurrent calls - only one load executes
const [result1, result2, result3] = await Promise.all([
  loadSDKOnce({ packageName: '@rbee/sdk', requiredExports: ['Client'] }),
  loadSDKOnce({ packageName: '@rbee/sdk', requiredExports: ['Client'] }),
  loadSDKOnce({ packageName: '@rbee/sdk', requiredExports: ['Client'] }),
])

// All results share the same SDK instance
console.log(result1.sdk === result2.sdk) // true
```

## API

### `loadSDK<T>(options: LoadOptions): Promise<SDKLoadResult<T>>`

Load SDK with retry logic and timeout.

**Options:**
- `packageName` (required): Package name to import (e.g., '@rbee/queen-rbee-sdk')
- `requiredExports` (required): Array of required export names to validate
- `timeout` (optional): Timeout in milliseconds (default: 15000)
- `maxAttempts` (optional): Max retry attempts (default: 3)
- `baseBackoffMs` (optional): Base backoff delay in ms (default: 300)
- `initArg` (optional): Initialization argument for WASM init function

**Returns:**
- `sdk`: Loaded SDK module
- `loadTime`: Total load time in milliseconds
- `attempts`: Number of attempts required

### `loadSDKOnce<T>(options: LoadOptions): Promise<SDKLoadResult<T>>`

Load SDK once using singleflight pattern. Subsequent calls return cached result.

### `createSDKLoader<T>(defaultOptions: Omit<LoadOptions, 'initArg'>)`

Create SDK loader factory with default options.

**Returns:**
- `load(initArg?)`: Load SDK (may load multiple times)
- `loadOnce(initArg?)`: Load SDK once (singleflight pattern)

## Error Handling

The loader handles several error cases:

```typescript
try {
  const { sdk } = await loadSDK({
    packageName: '@rbee/sdk',
    requiredExports: ['Client'],
  })
} catch (error) {
  if (error.message.includes('Timeout')) {
    // Timeout exceeded
  } else if (error.message.includes('missing required export')) {
    // Export validation failed
  } else if (error.message.includes('browser environment')) {
    // Running in SSR/Node.js
  } else if (error.message.includes('WebAssembly not supported')) {
    // Browser doesn't support WASM
  } else {
    // Network error or other failure
  }
}
```

## Retry Logic

The loader uses exponential backoff with jitter:

- **Attempt 1:** Immediate
- **Attempt 2:** 300ms + jitter (0-300ms)
- **Attempt 3:** 600ms + jitter (0-300ms)

Backoff formula: `2^(attempt-1) * baseBackoffMs + random(0, baseBackoffMs)`

## Testing

Run tests:

```bash
pnpm test
```

Run tests in watch mode:

```bash
pnpm test:watch
```

## Development

Build the package:

```bash
pnpm build
```

Watch mode:

```bash
pnpm dev
```

## License

GPL-3.0-or-later
