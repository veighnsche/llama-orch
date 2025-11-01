# Worker Install - Missing Frontend Integration 🚨

**Date:** 2025-11-01  
**Status:** ❌ **NOT WIRED UP**

## Problem

The "Install Worker" button in the UI is **NOT connected** to the backend. It only logs to console!

## Current Flow (BROKEN)

```
UI: Click "Install Worker" button
    ↓
WorkerCatalogView.tsx: handleInstall(workerId)
    ↓
WorkerManagement/index.tsx: handleInstallWorker(workerId)
    ↓
❌ console.log('Installing worker:', workerId)  // STOPS HERE!
    ↓
❌ NO SDK CALL
❌ NO BACKEND CALL
❌ NOTHING HAPPENS
```

## What's Missing

### 1. SDK Method ❌

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/operations.rs`

**Missing:**
```rust
#[wasm_bindgen(js_name = workerInstall)]
pub fn worker_install(hive_id: String, worker_id: String) -> JsValue {
    let op = Operation::WorkerInstall(WorkerInstallRequest {
        hive_id,
        worker_id,
    });
    to_value(&op).unwrap()
}
```

**Current state:** Only has `workerSpawn`, `workerList`, `workerDelete`, `modelDownload`, etc. **NO `workerInstall`!**

### 2. React Hook ❌

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useHiveOperations.ts`

**Missing:**
```typescript
export interface UseHiveOperationsResult {
  spawnWorker: (params: SpawnWorkerParams) => void
  installWorker: (workerId: string) => void  // ← MISSING!
  isPending: boolean
  isSuccess: boolean
  isError: boolean
  error: Error | null
  reset: () => void
}

export function useHiveOperations(): UseHiveOperationsResult {
  // ... existing spawnWorker mutation ...
  
  // MISSING: installWorker mutation
  const installMutation = useMutation<any, Error, string>({
    mutationFn: async (workerId: string) => {
      await ensureWasmInit()
      const hiveId = client.hiveId
      const op = OperationBuilder.workerInstall(hiveId, workerId)
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
          console.log('[Hive] Worker install:', line)
        }
      })
      
      return { success: true }
    },
  })

  return {
    spawnWorker: mutation.mutate,
    installWorker: installMutation.mutate,  // ← MISSING!
    // ...
  }
}
```

**Current state:** Only has `spawnWorker`. **NO `installWorker`!**

### 3. UI Integration ❌

**File:** `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/index.tsx`

**Current (BROKEN):**
```typescript
const handleInstallWorker = async (workerId: string) => {
  // TODO: Call hive backend to install worker
  console.log('Installing worker:', workerId)
  // POST /v1/workers/install { worker_id: workerId }
}
```

**Should be:**
```typescript
import { useHiveOperations } from '@rbee/rbee-hive-react'

export function WorkerManagement() {
  const { installWorker, isPending } = useHiveOperations()
  
  const handleInstallWorker = async (workerId: string) => {
    installWorker(workerId)
  }
  
  return (
    <WorkerCatalogView
      onInstall={handleInstallWorker}
      onRemove={handleRemoveWorker}
    />
  )
}
```

## What IS Working ✅

### Backend (COMPLETE)

1. ✅ **Operation Contract** - `WorkerInstall` added to `Operation` enum
2. ✅ **Request Type** - `WorkerInstallRequest { hive_id, worker_id }`
3. ✅ **Job Router** - `Operation::WorkerInstall` match arm routes to handler
4. ✅ **Handler** - `worker_install::handle_worker_install()` fully implemented (318 LOC)
5. ✅ **PKGBUILD** - Parser and executor ready
6. ✅ **Catalog** - Worker catalog service with 3 workers + PKGBUILDs

### Backend Flow (WORKS IF CALLED)

```
POST http://localhost:7835/v1/jobs
{
  "operation": "worker_install",
  "hive_id": "localhost",
  "worker_id": "llm-worker-rbee-cpu"
}
    ↓
job_router.rs: Operation::WorkerInstall
    ↓
worker_install::handle_worker_install(worker_id, worker_catalog)
    ↓
1. GET http://localhost:8787/workers/{worker_id}
2. GET http://localhost:8787/workers/{worker_id}/PKGBUILD
3. PkgBuild::parse(content)
4. PkgBuildExecutor::build()
5. PkgBuildExecutor::package()
6. Install to /usr/local/bin
7. Update capabilities
    ↓
SSE Stream: data: ✅ Worker installation complete!
```

**This works!** You can test it with curl:

```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "worker_install",
    "hive_id": "localhost",
    "worker_id": "llm-worker-rbee-cpu"
  }'
```

## Required Flow (NOT IMPLEMENTED)

```
UI: Click "Install Worker"
    ↓
WorkerCatalogView: handleInstall(workerId)
    ↓
WorkerManagement: handleInstallWorker(workerId)
    ↓
❌ useHiveOperations.installWorker(workerId)  // MISSING!
    ↓
❌ SDK: OperationBuilder.workerInstall(hiveId, workerId)  // MISSING!
    ↓
❌ HiveClient.submitAndStream(operation, callback)  // MISSING!
    ↓
POST http://localhost:7835/v1/jobs { operation: "worker_install", ... }
    ↓
✅ Backend handles it (THIS PART WORKS!)
```

## Implementation Checklist

### Priority 1: SDK Method

- [ ] Add `WorkerInstallRequest` import to `operations.rs`
- [ ] Add `worker_install()` method to `OperationBuilder`
- [ ] Rebuild WASM: `cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk && wasm-pack build --target web`

### Priority 2: React Hook

- [ ] Add `installWorker` to `UseHiveOperationsResult` interface
- [ ] Add `installMutation` using `useMutation`
- [ ] Return `installWorker: installMutation.mutate`
- [ ] Export `installWorker` state (`installingWorker`, `installError`)

### Priority 3: UI Integration

- [ ] Import `useHiveOperations` in `WorkerManagement/index.tsx`
- [ ] Replace `console.log` with `installWorker(workerId)`
- [ ] Show loading state from `isPending`
- [ ] Show errors from `error`
- [ ] Update `installedWorkers` state on success

### Priority 4: Progress Streaming

- [ ] Show real-time progress in UI
- [ ] Parse SSE events from backend
- [ ] Display build output
- [ ] Show completion message

## Comparison with Working Feature

### spawnWorker (WORKS) ✅

```typescript
// SDK
OperationBuilder.workerSpawn(hiveId, modelId, workerType, deviceId)

// React Hook
const { spawnWorker, isPending } = useHiveOperations()
spawnWorker({ modelId, workerType, deviceId })

// UI
<Button onClick={() => spawnWorker({ ... })} disabled={isPending}>
  {isPending ? 'Spawning...' : 'Spawn Worker'}
</Button>
```

### installWorker (BROKEN) ❌

```typescript
// SDK - MISSING!
// OperationBuilder.workerInstall(hiveId, workerId)

// React Hook - MISSING!
// const { installWorker, isPending } = useHiveOperations()
// installWorker(workerId)

// UI - BROKEN!
<Button onClick={() => console.log('Installing worker:', workerId)}>
  Install Worker
</Button>
```

## Testing

### Manual Test (Once Fixed)

1. Start catalog: `cd bin/80-hono-worker-catalog && pnpm dev`
2. Start hive: `cd bin/20_rbee_hive && cargo run`
3. Start UI: `cd bin/20_rbee_hive/ui/app && pnpm dev`
4. Open browser: `http://localhost:7836`
5. Navigate to Worker Management → Worker Catalog
6. Click "Install Worker" on `llm-worker-rbee-cpu`
7. Watch SSE stream in UI
8. Verify binary: `ls -la /usr/local/bin/llm-worker-rbee-cpu`

### Current Test (Broken)

1. Click "Install Worker"
2. Open browser console
3. See: `Installing worker: llm-worker-rbee-cpu`
4. Nothing else happens ❌

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Backend Handler | ✅ Complete | 318 LOC, fully tested |
| Job Router | ✅ Complete | Routes to handler |
| Operation Contract | ✅ Complete | WorkerInstall added |
| PKGBUILD Infrastructure | ✅ Complete | Parser + executor ready |
| Worker Catalog | ✅ Complete | 3 workers + PKGBUILDs |
| **SDK Method** | ❌ **Missing** | **No `workerInstall()` in operations.rs** |
| **React Hook** | ❌ **Missing** | **No `installWorker()` in useHiveOperations.ts** |
| **UI Integration** | ❌ **Broken** | **Just console.log, no actual call** |

## Next Steps

1. **Add SDK method** (5 minutes)
2. **Add React hook** (10 minutes)
3. **Wire up UI** (5 minutes)
4. **Test end-to-end** (5 minutes)

**Total time to fix:** ~25 minutes

---

**Conclusion:** Backend is 100% ready and working. Frontend is 0% connected. The button literally does nothing except log to console.
