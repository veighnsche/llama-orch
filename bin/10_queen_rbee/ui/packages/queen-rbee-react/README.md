# @rbee/queen-rbee-react

**TEAM-294: Migrated from `frontend/packages/rbee-react`**

React hooks for the queen-rbee WASM SDK.

**Location:** `bin/10_queen_rbee/ui/packages/queen-rbee-react`

## Installation

```bash
pnpm add @rbee/queen-rbee-react
```

## Usage

### useRbeeSDK

Hook for loading and initializing the queen-rbee WASM SDK.

```tsx
import { useRbeeSDK } from '@rbee/queen-rbee-react';

function MyComponent() {
  const { sdk, loading, error } = useRbeeSDK();

  if (loading) {
    return <div>Loading queen-rbee SDK...</div>;
  }

  if (error) {
    return <div>Error loading SDK: {error.message}</div>;
  }

  // SDK is ready - use sdk.RbeeClient, sdk.HeartbeatMonitor, etc.
  const client = new sdk.RbeeClient('http://localhost:7833');
  
  return <div>SDK loaded!</div>;
}
```

## API

### `useRbeeSDK()`

Returns an object with:

- `sdk: RbeeSDK | null` - The initialized SDK object containing:
  - `RbeeClient` - Client for submitting operations (class constructor)
  - `HeartbeatMonitor` - Monitor for real-time system status (class constructor)
  - `OperationBuilder` - Builder for creating operations (class with static methods)
- `loading: boolean` - Whether the SDK is still loading
- `error: Error | null` - Any error that occurred during loading

### Types

All WASM types are re-exported for convenience:

```typescript
import type { RbeeClient, HeartbeatMonitor, OperationBuilder } from '@rbee/queen-rbee-react';
```

These are the actual class types from the WASM package, providing full TypeScript support.

## Architecture

This package provides a thin React wrapper around `@rbee/queen-rbee-sdk` (the WASM package). It handles:

1. Dynamic WASM module loading
2. SDK initialization
3. React state management
4. Error handling

The actual SDK functionality is implemented in Rust and compiled to WASM in the `@rbee/sdk` package.

## License

GPL-3.0-or-later
