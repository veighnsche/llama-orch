# @rbee/react-hooks

**TEAM-356:** Reusable React hooks for common patterns in rbee UIs.

## Features

- ✅ `useAsyncState` - Async data loading with loading/error states
- ✅ `useSSEWithHealthCheck` - SSE connection with health check
- ✅ Automatic cleanup on unmount
- ✅ TypeScript support with strict mode
- ✅ Comprehensive test coverage (30 tests)

## Installation

```bash
pnpm add @rbee/react-hooks
```

## Hooks

### useAsyncState

Load async data with automatic loading/error state management.

**Features:**
- Automatic loading state
- Error handling
- Cleanup on unmount (prevents state updates after unmount)
- Refetch functionality
- Skip option
- Success/error callbacks

**Usage:**

```typescript
import { useAsyncState } from '@rbee/react-hooks'

function MyComponent() {
  const { data, loading, error, refetch } = useAsyncState(
    async () => {
      const response = await fetch('/api/data')
      return response.json()
    },
    [] // dependencies
  )

  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error.message}</div>
  
  return (
    <div>
      <pre>{JSON.stringify(data, null, 2)}</pre>
      <button onClick={refetch}>Refresh</button>
    </div>
  )
}
```

**With options:**

```typescript
const { data, loading, error } = useAsyncState(
  async () => fetchData(),
  [userId], // Re-fetch when userId changes
  {
    skip: !userId, // Skip if no userId
    onSuccess: (data) => console.log('Loaded:', data),
    onError: (error) => console.error('Failed:', error),
  }
)
```

### useSSEWithHealthCheck

Connect to SSE stream with health check before connection.

**Features:**
- Health check before SSE connection (prevents CORS errors)
- Automatic retry on failure
- Connection state tracking
- Cleanup on unmount
- Manual retry function

**Usage:**

```typescript
import { useSSEWithHealthCheck } from '@rbee/react-hooks'

function HeartbeatMonitor() {
  const { data, connected, loading, error, retry } = useSSEWithHealthCheck(
    (baseUrl) => new sdk.HeartbeatMonitor(baseUrl),
    'http://localhost:7833'
  )

  if (loading) return <div>Connecting...</div>
  if (error) return <div>Error: {error.message} <button onClick={retry}>Retry</button></div>
  
  return (
    <div>
      <div>Status: {connected ? 'Connected' : 'Disconnected'}</div>
      {data && <pre>{JSON.stringify(data, null, 2)}</pre>}
    </div>
  )
}
```

**With options:**

```typescript
const { data, connected, error } = useSSEWithHealthCheck(
  (baseUrl) => new sdk.Monitor(baseUrl),
  'http://localhost:7833',
  {
    autoRetry: true,      // Auto-retry on failure (default: true)
    retryDelayMs: 5000,   // Retry delay (default: 5000)
    maxRetries: 3,        // Max retry attempts (default: 3)
  }
)
```

## API

### useAsyncState

```typescript
function useAsyncState<T>(
  asyncFn: () => Promise<T>,
  deps: DependencyList,
  options?: AsyncStateOptions
): AsyncStateResult<T>
```

**Options:**
- `skip?: boolean` - Skip initial load (default: false)
- `onSuccess?: (data: T) => void` - Callback on success
- `onError?: (error: Error) => void` - Callback on error

**Returns:**
- `data: T | null` - Loaded data
- `loading: boolean` - Loading state
- `error: Error | null` - Error if failed
- `refetch: () => void` - Manually trigger refetch

### useSSEWithHealthCheck

```typescript
function useSSEWithHealthCheck<T>(
  createMonitor: (baseUrl: string) => Monitor<T>,
  baseUrl: string,
  options?: SSEHealthCheckOptions
): SSEHealthCheckResult<T>
```

**Monitor Interface:**
```typescript
interface Monitor<T> {
  checkHealth: () => Promise<boolean>
  start: (onData: (data: T) => void) => void
  stop: () => void
}
```

**Options:**
- `autoRetry?: boolean` - Auto-retry on failure (default: true)
- `retryDelayMs?: number` - Retry delay in ms (default: 5000)
- `maxRetries?: number` - Max retry attempts (default: 3)

**Returns:**
- `data: T | null` - Latest SSE data
- `connected: boolean` - Connection state
- `loading: boolean` - Initial connection loading
- `error: Error | null` - Error if failed
- `retry: () => void` - Manually trigger retry

## Examples

### CRUD Operations

```typescript
import { useAsyncState } from '@rbee/react-hooks'

function ScriptManager() {
  const { data: scripts, loading, error, refetch } = useAsyncState(
    async () => {
      const client = new RhaiClient(baseUrl)
      return client.listScripts()
    },
    [baseUrl]
  )

  const handleSave = async (script) => {
    const client = new RhaiClient(baseUrl)
    await client.saveScript(script)
    refetch() // Reload list
  }

  return (
    <div>
      {loading && <div>Loading scripts...</div>}
      {error && <div>Error: {error.message}</div>}
      {scripts?.map(script => (
        <div key={script.id}>{script.name}</div>
      ))}
    </div>
  )
}
```

### Real-time Monitoring

```typescript
import { useSSEWithHealthCheck } from '@rbee/react-hooks'

function SystemMonitor() {
  const { data, connected } = useSSEWithHealthCheck(
    (url) => new sdk.HeartbeatMonitor(url),
    getServiceUrl('queen', 'prod')
  )

  return (
    <div>
      <StatusIndicator connected={connected} />
      {data && (
        <div>
          <div>CPU: {data.cpu}%</div>
          <div>Memory: {data.memory}%</div>
          <div>Uptime: {data.uptime}s</div>
        </div>
      )}
    </div>
  )
}
```

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
