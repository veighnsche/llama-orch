# TEAM-381: ModelManagement Loading Issue - Root Cause

**Date:** 2025-11-01  
**Status:** 🐛 DIAGNOSED

## Problem
`ModelManagement.tsx` hangs at "Loading models..." indefinitely.

## Root Cause: Backend Not Running

### Investigation
```bash
# Check if rbee-hive is running
ps aux | grep rbee-hive
# Result: No process found ❌

# Check if port 7835 is listening
curl http://localhost:7835/v1/capabilities
# Result: Connection refused ❌
```

**Diagnosis:** The `useModels()` hook is trying to connect to `http://localhost:7835` but the rbee-hive backend isn't running.

### Code Flow
```tsx
// ModelManagement.tsx
const { models, loading, error } = useModels()
// ↓
// packages/rbee-hive-react/src/index.ts
export function useModels() {
  return useQuery({
    queryFn: async () => {
      const client = new HiveClient('http://localhost:7835', 'localhost')
      await client.submitAndStream(op, callback)
      // ↑ This hangs because backend isn't running!
    }
  })
}
```

**Why it hangs:**
1. `useQuery` calls `queryFn`
2. `client.submitAndStream()` tries to connect to `http://localhost:7835`
3. Connection fails (backend not running)
4. TanStack Query retries 3 times (with exponential backoff)
5. Each retry takes longer (1s, 2s, 4s...)
6. Total hang time: ~7-10 seconds before error
7. But if network timeout is long, it can hang for 30+ seconds

---

## Solution: Start the Backend

### Option 1: Run rbee-hive Binary
```bash
# Build and run
cargo run -p rbee-hive

# Or with auto-reload
cargo watch -x 'run -p rbee-hive'
```

**Expected output:**
```
🐝 rbee-hive starting on http://0.0.0.0:7835
✅ Serving UI at http://localhost:7835
```

### Option 2: Use the Daemon Script
```bash
# If you have a start script
./rbee start hive

# Or
pnpm start:hive  # (if configured)
```

### Verify Backend is Running
```bash
# Check process
ps aux | grep rbee-hive
# Should show: rbee-hive running

# Check endpoint
curl http://localhost:7835/v1/capabilities
# Should return: JSON with hive capabilities

# Check UI
curl http://localhost:7835/
# Should return: HTML (the embedded UI)
```

---

## Better Error Handling

### Current Issue
The UI doesn't show a helpful error message when the backend is down. It just says "Loading models..." forever.

### Recommended Fix

**Update ModelManagement.tsx:**
```tsx
export function ModelManagement() {
  const { models, loading, error } = useModels()
  
  // ... existing code ...

  return (
    <Card className="col-span-2">
      {/* ... header ... */}
      
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2 space-y-2">
          {loading && (
            <div className="text-center py-12 text-muted-foreground">
              <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4" />
              Loading models...
            </div>
          )}

          {error && (
            <div className="text-center py-12">
              <div className="text-destructive mb-4">
                ❌ Failed to connect to rbee-hive backend
              </div>
              <div className="text-sm text-muted-foreground mb-4">
                {error.message}
              </div>
              <div className="text-sm bg-muted p-4 rounded-md text-left max-w-md mx-auto">
                <p className="font-semibold mb-2">To fix this:</p>
                <ol className="list-decimal list-inside space-y-1">
                  <li>Start the backend: <code className="bg-background px-1 rounded">cargo run -p rbee-hive</code></li>
                  <li>Verify it's running: <code className="bg-background px-1 rounded">curl http://localhost:7835/v1/capabilities</code></li>
                  <li>Refresh this page</li>
                </ol>
              </div>
            </div>
          )}

          {!loading && !error && (
            // ... existing model list ...
          )}
        </div>
      </div>
    </Card>
  )
}
```

### Update useModels Hook (Optional)

**Add better error messages:**
```ts
// packages/rbee-hive-react/src/index.ts
export function useModels() {
  return useQuery({
    queryKey: ['hive-models'],
    queryFn: async () => {
      try {
        await ensureWasmInit()
        const hiveId = client.hiveId
        const op = OperationBuilder.modelList(hiveId)
        const lines: string[] = []
        
        await client.submitAndStream(op, (line: string) => {
          if (line !== '[DONE]') {
            lines.push(line)
          }
        })
        
        const jsonLine = lines.reverse().find(line => {
          const trimmed = line.trim()
          return trimmed.startsWith('[') || trimmed.startsWith('{')
        })
        
        return jsonLine ? JSON.parse(jsonLine) : []
      } catch (err) {
        // Add context to the error
        if (err instanceof Error) {
          throw new Error(
            `Failed to fetch models from rbee-hive (http://localhost:7835). ` +
            `Is the backend running? Original error: ${err.message}`
          )
        }
        throw err
      }
    },
    staleTime: 30000,
    retry: 2, // Reduce retries (was 3)
    retryDelay: 1000, // Fixed 1s delay (was exponential)
  })
}
```

---

## Development Workflow

### Recommended Setup

**Terminal 1: Backend (Rust)**
```bash
cargo watch -x 'run -p rbee-hive'
```

**Terminal 2: Frontend (Vite)**
```bash
cd bin/20_rbee_hive/ui/app
pnpm dev
```

**Terminal 3: SDK Watcher (if editing Rust SDK)**
```bash
cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk
cargo watch -i 'pkg/*' -s 'wasm-pack build --target bundler --out-dir pkg/bundler'
```

### Or Use Turbo (Frontend Only)
```bash
# Terminal 1: Backend
cargo watch -x 'run -p rbee-hive'

# Terminal 2: All frontend packages
turbo dev --filter=@rbee/rbee-hive-ui
```

---

## Quick Checklist

When ModelManagement hangs, check:

- [ ] **Backend running?** `ps aux | grep rbee-hive`
- [ ] **Port 7835 listening?** `lsof -i :7835`
- [ ] **Backend healthy?** `curl http://localhost:7835/v1/capabilities`
- [ ] **UI accessible?** Open `http://localhost:7835` in browser
- [ ] **Network errors in console?** Check browser DevTools → Network tab
- [ ] **WASM initialized?** Check browser Console for errors

### Common Issues

**1. Backend not running**
```bash
# Fix: Start it
cargo run -p rbee-hive
```

**2. Port 7835 already in use**
```bash
# Check what's using it
lsof -i :7835

# Kill the process
kill -9 <PID>

# Or use a different port (update code)
```

**3. WASM not loading**
```bash
# Check browser console for:
# "Failed to fetch rbee_hive_sdk_bg.wasm"

# Fix: Rebuild SDK
cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk
pnpm build
```

**4. CORS errors**
```bash
# Check browser console for:
# "CORS policy: No 'Access-Control-Allow-Origin' header"

# Fix: Backend should allow CORS from localhost:7836
# Check rbee-hive/src/main.rs for CORS config
```

---

## Summary

**Root Cause:** rbee-hive backend not running on port 7835

**Immediate Fix:**
```bash
cargo run -p rbee-hive
```

**Long-term Fix:**
1. Add better error messages in UI
2. Add backend health check
3. Add SDK watchers (see TEAM_381_WASM_SDK_WATCHERS.md)
4. Document development workflow

**Prevention:**
- Always start backend before opening UI
- Add health check endpoint
- Show connection status in UI
- Add retry with user feedback
