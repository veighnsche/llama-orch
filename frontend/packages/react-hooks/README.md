# @rbee/react-hooks

**TEAM-356:** Reusable React hooks for common patterns in rbee UIs.


## Installation

```bash
# Install TanStack Query (for data fetching)
pnpm add @tanstack/react-query @tanstack/react-query-devtools

# Install this package (for custom hooks)
pnpm add @rbee/react-hooks
```

## Hooks

### Data Fetching: TanStack Query

For async data fetching, we use **TanStack Query** (re-exported for convenience).

**Why TanStack Query?**
- ✅ Industry standard (47k+ GitHub stars)
- ✅ Automatic caching and deduplication
- ✅ DevTools for debugging
- ✅ Background refetching
- ✅ Consistent with rest of codebase

**Setup (one-time):**
```typescript
// App.tsx
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000, // 5 seconds
      retry: 1,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <YourApp />
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  )
}
```

**Usage:**
```typescript
import { useQuery } from '@tanstack/react-query'

function MyComponent({ userId }: { userId: string }) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['user-data', userId],
    queryFn: async () => {
      const response = await fetch(`/api/users/${userId}`)
      return response.json()
    },
  })

  if (isLoading) return <div>Loading...</div>
  if (error) return <div>Error: {error.message}</div>
  
  return <div>{JSON.stringify(data)}</div>
}
```

**Mutations (Create/Update/Delete):**
```typescript
import { useMutation, useQueryClient } from '@tanstack/react-query'

function MyComponent() {
  const queryClient = useQueryClient()
  
  const mutation = useMutation({
    mutationFn: (newData) => fetch('/api/data', {
      method: 'POST',
      body: JSON.stringify(newData),
    }),
    onSuccess: () => {
      // Invalidate and refetch
      queryClient.invalidateQueries({ queryKey: ['data'] })
    },
  })

  return (
    <button onClick={() => mutation.mutate({ name: 'New Item' })}>
      {mutation.isPending ? 'Saving...' : 'Save'}
    </button>
  )
}
```

**See [TanStack Query docs](https://tanstack.com/query/latest) for full API.**

---

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
