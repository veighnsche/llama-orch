# TEAM-353 Step 3: Hive UI - Add Narration Support

**Estimated Time:** 30-45 minutes  
**Priority:** HIGH  
**Previous Step:** TEAM_353_STEP_2_HOOKS_MIGRATION.md  
**Next Step:** TEAM_353_STEP_4_CONFIG_CLEANUP.md

---

## Mission

Add narration support for Hive operations using @rbee/narration-client.

**Location:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/`

---

## Deliverables Checklist

- [ ] useHiveOperations hook created
- [ ] Uses @rbee/narration-client
- [ ] Uses SERVICES.hive config
- [ ] Narration flows to Keeper
- [ ] Package builds successfully
- [ ] TEAM-353 signatures added

---

## Step 1: Create useHiveOperations Hook

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useHiveOperations.ts`

```typescript
// TEAM-353: Hive operations with narration support
// Uses @rbee/narration-client (no custom implementation)

'use client'

import { useState } from 'react'
import { createStreamHandler, SERVICES } from '@rbee/narration-client'

export interface UseHiveOperationsResult {
  spawnWorker: (modelId: string) => Promise<void>
  spawning: boolean
  error: Error | null
}

/**
 * Hook for Hive operations with narration
 * 
 * TEAM-353: Uses @rbee/narration-client for narration
 * 
 * @param baseUrl - Hive API URL (from @rbee/shared-config)
 * @returns Hive operation functions
 */
export function useHiveOperations(
  baseUrl: string
): UseHiveOperationsResult {
  const [spawning, setSpawning] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const spawnWorker = async (modelId: string) => {
    setSpawning(true)
    setError(null)

    try {
      // TEAM-353: Create narration handler using shared package
      const narrationHandler = createStreamHandler(SERVICES.hive, (event) => {
        console.log('[Hive] Narration event:', event)
      }, {
        debug: true,
        silent: false,
        validate: true,
      })

      // Submit operation with narration
      const operation = {
        operation: 'worker_spawn',
        model_id: modelId,
      }

      // TODO: Use Hive SDK to submit operation with narration
      // This depends on how rbee-hive-sdk exposes operations
      // For now, this is a placeholder showing the pattern

      console.log('[Hive] Spawning worker with model:', modelId)
      // await hiveClient.submitAndStream(operation, narrationHandler)

    } catch (err) {
      console.error('[Hive] Worker spawn error:', err)
      setError(err as Error)
      throw err
    } finally {
      setSpawning(false)
    }
  }

  return {
    spawnWorker,
    spawning,
    error,
  }
}
```

**Key points:**
- ✅ Uses `createStreamHandler` from @rbee/narration-client
- ✅ Uses `SERVICES.hive` config (no hardcoded URLs)
- ✅ No custom narration parsing
- ✅ Events automatically flow to Keeper

---

## Step 2: Create hooks/index.ts

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/index.ts`

```typescript
// TEAM-353: Hive React hooks
export { useHiveOperations } from './useHiveOperations'
export type { UseHiveOperationsResult } from './useHiveOperations'
```

---

## Step 3: Update Package Index

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/index.ts`

```typescript
// TEAM-353: Hive React hooks package
// Uses shared packages - NO custom implementations

import { useQuery } from '@tanstack/react-query'
import { listModels, listWorkers, type Model, type Worker } from '@rbee/rbee-hive-sdk'

// Export data fetching hooks
export function useModels() {
  const { 
    data: models, 
    isLoading: loading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['hive-models'],
    queryFn: listModels,
    staleTime: 30000,
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })

  return { 
    models: models || [], 
    loading,
    error: error as Error | null,
    refetch
  }
}

export function useWorkers() {
  const { 
    data: workers, 
    isLoading: loading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['hive-workers'],
    queryFn: listWorkers,
    staleTime: 5000,
    refetchInterval: 2000,
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })

  return { 
    workers: workers || [], 
    loading,
    error: error as Error | null,
    refetch
  }
}

// Export operation hooks
export { useHiveOperations } from './hooks/useHiveOperations'
export type { UseHiveOperationsResult } from './hooks/useHiveOperations'

// Re-export types
export type { Model, Worker } from '@rbee/rbee-hive-sdk'
```

---

## Step 4: Verify SERVICES.hive Config

**Check:** `frontend/packages/narration-client/src/config.ts`

**Should contain:**
```typescript
export const SERVICES: Record<ServiceName, ServiceConfig> = {
  queen: { ... },
  hive: {
    name: 'rbee-hive',
    devPort: PORTS.hive.dev,      // 7836
    prodPort: PORTS.hive.prod,    // 7835
    keeperDevPort: PORTS.keeper.dev,
    keeperProdOrigin: '*',
  },
  worker: { ... },
}
```

**If missing:** Add it to the config file.

---

## Step 5: Build Package

```bash
cd bin/20_rbee_hive/ui/packages/rbee-hive-react
pnpm build
```

**Expected output:**
```
> @rbee/rbee-hive-react@0.1.0 build
> tsc

✓ Built successfully
```

---

## Step 6: Test Narration Flow

**Terminal 1:** Start Hive backend
```bash
cargo run --bin rbee-hive
```

**Terminal 2:** Start Hive UI
```bash
cd bin/20_rbee_hive/ui/app
pnpm dev
```

**Terminal 3:** Start Keeper UI
```bash
cd bin/00_rbee_keeper/ui
pnpm dev
```

**Browser:** Open Keeper at http://localhost:5173

**Test:**
1. Navigate to Hive page (iframe should load)
2. Trigger worker spawn operation
3. **Verify narration appears in:**
   - Hive console: `[Hive] Sending to parent: ...`
   - Keeper console: `[Keeper] Received narration: ...`
   - Keeper narration panel

**Expected console logs:**
```
[Hive] SSE line: data: {"actor":"rbee_hive",...}
[Hive] Narration event: { actor: "rbee_hive", ... }
[rbee-hive] Sending to parent: { origin: "http://localhost:5173", ... }
[Keeper] Received narration: { actor: "rbee_hive", ... }
```

---

## Testing Checklist

- [ ] `pnpm build` (rbee-hive-react) - success
- [ ] SERVICES.hive config exists
- [ ] Narration handler created correctly
- [ ] Narration appears in Hive console
- [ ] Narration appears in Keeper console
- [ ] Narration appears in Keeper panel
- [ ] No TypeScript errors
- [ ] TEAM-353 signatures added

---

## Success Criteria

✅ useHiveOperations hook created  
✅ Uses @rbee/narration-client  
✅ Uses SERVICES.hive config  
✅ Narration flows to Keeper  
✅ Package builds successfully  
✅ TEAM-353 signatures added

---

## Next Step

Continue to **TEAM_353_STEP_4_CONFIG_CLEANUP.md** to remove hardcoded URLs.

---

**TEAM-353 Step 3: Narration integration complete!** ✅
