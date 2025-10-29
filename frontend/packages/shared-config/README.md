# @rbee/shared-config

Single source of truth for port configuration across the entire rbee project.

**TEAM-351:** Bug fixes - Validation, type safety, edge cases, HTTPS support

## Installation

```bash
pnpm add @rbee/shared-config
```

## Features

✅ **Port validation** - All ports validated at module load (1-65535)  
✅ **Type safety** - Full TypeScript type inference  
✅ **Deduplication** - No duplicate origins in getAllowedOrigins()  
✅ **HTTPS support** - Optional HTTPS for production  
✅ **Error handling** - Clear error messages for invalid usage  
✅ **Rust codegen** - Auto-generates constants from TypeScript source  
✅ **Edge cases** - Handles null ports, keeper prod, etc.

## Usage

### Get iframe URL

```typescript
import { getIframeUrl } from '@rbee/shared-config'

const isDev = import.meta.env.DEV

// Basic usage
const queenUrl = getIframeUrl('queen', isDev)
// Dev: http://localhost:7834
// Prod: http://localhost:7833

// With HTTPS (production)
const queenUrlHttps = getIframeUrl('queen', false, true)
// https://localhost:7833

// Error handling
try {
  getIframeUrl('keeper', false)  // Throws: Keeper has no prod HTTP port
} catch (error) {
  console.error(error.message)
}
```

### Get allowed origins

```typescript
import { getAllowedOrigins } from '@rbee/shared-config'

// HTTP only (default)
const origins = getAllowedOrigins()
// ["http://localhost:7833", "http://localhost:7834", "http://localhost:7835", ...]

// Include HTTPS for production
const originsWithHttps = getAllowedOrigins(true)
// ["http://localhost:7833", "https://localhost:7833", ...]
```

### Get parent origin

```typescript
import { getParentOrigin } from '@rbee/shared-config'

const currentPort = parseInt(window.location.port)

try {
  const parentOrigin = getParentOrigin(currentPort)
  // Dev ports (7834, 7836, 7837, 5173): http://localhost:5173
  // Prod ports: '*' (Tauri app)
} catch (error) {
  // Invalid port (< 1 or > 65535)
  console.error(error.message)
}
```

### Get service URL

```typescript
import { getServiceUrl } from '@rbee/shared-config'

// Development mode
const queenDev = getServiceUrl('queen', 'dev')
// http://localhost:7834

// Production mode
const queenProd = getServiceUrl('queen', 'prod')
// http://localhost:7833

// Backend mode (uses backend port)
const queenBackend = getServiceUrl('queen', 'backend')
// http://localhost:7833

// With HTTPS
const queenHttps = getServiceUrl('queen', 'prod', true)
// https://localhost:7833
```

## Generate Rust Constants

```bash
pnpm generate:rust
```

This creates `frontend/shared-constants.rs` for use in build.rs scripts.

**Features:**
- ✅ Imports from TypeScript source (single source of truth)
- ✅ Validates all ports (1-65535)
- ✅ Handles null ports with comments
- ✅ Error handling with exit codes
- ✅ Creates output directory if needed

## Port Validation

All ports are validated at module load time:

```typescript
// Valid ports: 1-65535
// null is valid (e.g., keeper.prod - Tauri app)

// This would throw an error:
export const PORTS = {
  invalid: {
    dev: 99999,  // ❌ Error: Invalid port (must be 1-65535)
  }
}
```

## Adding a New Service

1. Update `src/ports.ts` - Add service to PORTS constant
2. Run `pnpm build` to compile TypeScript
3. Run `pnpm generate:rust` to update Rust constants
4. Update `PORT_CONFIGURATION.md`
5. Update backend Cargo.toml with default port

## Error Handling

All functions include proper error handling:

```typescript
// Invalid port throws error
getParentOrigin(99999)  // ❌ Error: Invalid port

// Keeper prod throws error (no HTTP port)
getIframeUrl('keeper', false)  // ❌ Error: Keeper has no production HTTP port

// Invalid service caught by TypeScript
getIframeUrl('invalid', true)  // ❌ TypeScript error: Type '"invalid"' is not assignable
```

## Type Safety

Full TypeScript support with type inference:

```typescript
import { PORTS, ServiceName } from '@rbee/shared-config'

// ServiceName = 'keeper' | 'queen' | 'hive' | 'worker'
const service: ServiceName = 'queen'  // ✅ Type-safe

// PORTS is readonly
PORTS.queen.dev = 9999  // ❌ TypeScript error: Cannot assign to 'dev'
```
